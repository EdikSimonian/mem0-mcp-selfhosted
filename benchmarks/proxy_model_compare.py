"""Head-to-head benchmark across 5 LiteLLM-proxied models for mem0 extraction.

Targets the actual production path: a LiteLLM-compatible proxy (set
BENCH_PROXY_BASE_URL) via OpenAI-compatible client, exercising mem0ai's real
prompts and tool schemas.

Vector path: FACT_RETRIEVAL_PROMPT (json_object response).
Graph path:  EXTRACT_ENTITIES_TOOL → RELATIONS_TOOL with EXTRACT_RELATIONS_PROMPT,
             benchmarked both with and without the local _AUGMENT_RULES patch.

No Qdrant/Neo4j writes. No MCP restarts.

Usage:
    python3 benchmarks/proxy_model_compare.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Pull mem0ai's actual prompts + tool schemas to mirror production exactly.
from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT
from mem0.graphs.tools import EXTRACT_ENTITIES_TOOL, RELATIONS_TOOL

from mem0_mcp_selfhosted.helpers import _AUGMENT_RULES

import openai

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PROXY_BASE_URL = os.environ.get(
    "BENCH_PROXY_BASE_URL", "https://your-litellm-proxy.example.com/v1"
)
KEYCHAIN_SERVICE = "mem0-mcp.openai-api-key"
USER_ID = "bench_user"
RUN_ID = "bench"

MODELS = [
    "gemini-3.1-flash-lite-preview",
    "gpt-5.4-nano",
    "claude-haiku",
    "gemini-3-flash-preview",
    "gpt-5.4-mini",
]


@dataclass
class TestCase:
    name: str
    text: str
    # Quality oracle: edge predicates that should hold (source_substr, rel_substr, dest_substr)
    expected_edges: list[tuple[str, str, str]]
    # Entity-level invariants the model should respect
    must_preserve_compound: list[str] = field(default_factory=list)
    forbidden_entity_substrings: list[str] = field(default_factory=list)
    # Edge-level invariants: triples that MUST NOT appear
    # (source_substr, rel_substr, dest_substr). Use for false-positive detection.
    forbidden_edges: list[tuple[str, str, str]] = field(default_factory=list)


CASES = [
    TestCase(
        name="family_pronoun",
        text=(
            "Sam's mom is Pat. Sam's sister Riley is married to Casey, "
            "and their son is Quinn."
        ),
        expected_edges=[
            ("pat", "mother", "sam"),
            ("riley", "married", "casey"),
            ("quinn", "son", "riley"),
            ("quinn", "son", "casey"),
        ],
        forbidden_entity_substrings=["run_id", "user_id"],
    ),
    TestCase(
        name="compound_org",
        text=("TestService runs on AWS us-east-1. Maya leads TestService at TestCorp."),
        expected_edges=[
            ("testservice", "run", "aws us-east-1"),
            ("maya", "lead", "testservice"),
            ("testservice", "owned_by", "testcorp"),
        ],
        must_preserve_compound=["aws us-east-1"],
    ),
    TestCase(
        name="person_via_artifact",
        text="Alice reviewed Bob's PR for the auth refactor.",
        expected_edges=[
            ("alice", "review", "bob"),
        ],
    ),
    TestCase(
        name="parenthetical_apposition",
        text=(
            "Avery (Sam's brother) is married to Morgan, and they "
            "have a daughter named Drew."
        ),
        # Subject is Avery. Sam must NOT receive any spouse/parent edges.
        expected_edges=[
            ("avery", "married", "morgan"),
            ("avery", "parent", "drew"),
            ("morgan", "parent", "drew"),
            ("avery", "brother", "sam"),
        ],
        forbidden_edges=[
            ("sam", "married", "morgan"),
            ("sam", "parent", "drew"),
            ("morgan", "married", "sam"),
        ],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt assembly (mirrors mem0/memory/graph_memory.py:252-275)
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_SYSTEM_TMPL = (
    "You are a smart assistant who understands entities and their types in a "
    "given text. If user message contains self reference such as 'I', 'me', "
    "'my' etc. then use {user_id} as the source entity. Extract all the "
    "entities from the text. ***DO NOT*** answer the question itself if the "
    "given text is a question."
)


def relations_system(user_identity: str, augment: bool) -> str:
    base = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
    if augment:
        base = base + _AUGMENT_RULES
    return base


def user_identity_string() -> str:
    return f"user_id: {USER_ID}, run_id: {RUN_ID}"


# ─────────────────────────────────────────────────────────────────────────────
# Proxy client
# ─────────────────────────────────────────────────────────────────────────────


def get_api_key() -> str:
    out = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-a",
            os.environ["USER"],
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def make_client() -> openai.OpenAI:
    return openai.OpenAI(base_url=PROXY_BASE_URL, api_key=get_api_key())


# ─────────────────────────────────────────────────────────────────────────────
# Phase callers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CallResult:
    ok: bool
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: Any = None
    error: str | None = None


def call_vector(
    client: openai.OpenAI, model: str, text: str
) -> tuple[CallResult, list[str]]:
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FACT_RETRIEVAL_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        dt = time.time() - t0
        content = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
            facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
        except json.JSONDecodeError:
            facts = []
        usage = resp.usage
        return CallResult(
            ok=True,
            latency_s=dt,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            raw=content,
        ), facts
    except Exception as e:
        return CallResult(
            ok=False, latency_s=time.time() - t0, error=f"{type(e).__name__}: {e}"
        ), []


def call_entities(
    client: openai.OpenAI, model: str, text: str
) -> tuple[CallResult, list[dict]]:
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": ENTITY_SYSTEM_TMPL.format(user_id=USER_ID),
                },
                {"role": "user", "content": text},
            ],
            tools=[EXTRACT_ENTITIES_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_entities"}},
            temperature=0.0,
        )
        dt = time.time() - t0
        msg = resp.choices[0].message
        entities: list[dict] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "extract_entities":
                    try:
                        args = json.loads(tc.function.arguments)
                        entities.extend(args.get("entities", []))
                    except json.JSONDecodeError:
                        pass
        usage = resp.usage
        return CallResult(
            ok=True,
            latency_s=dt,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            raw=[asdict_safe(tc) for tc in (msg.tool_calls or [])],
        ), entities
    except Exception as e:
        return CallResult(
            ok=False, latency_s=time.time() - t0, error=f"{type(e).__name__}: {e}"
        ), []


def call_relations(
    client: openai.OpenAI,
    model: str,
    text: str,
    entities: list[dict],
    augment: bool,
) -> tuple[CallResult, list[dict]]:
    t0 = time.time()
    try:
        entity_names = [e.get("entity", "") for e in entities]
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": relations_system(
                        user_identity_string(), augment=augment
                    ),
                },
                {
                    "role": "user",
                    "content": f"List of entities: {entity_names}.\n\nText: {text}",
                },
            ],
            tools=[RELATIONS_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "establish_relationships"},
            },
            temperature=0.0,
        )
        dt = time.time() - t0
        msg = resp.choices[0].message
        rels: list[dict] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "establish_relationships":
                    try:
                        args = json.loads(tc.function.arguments)
                        rels.extend(args.get("entities", []))
                    except json.JSONDecodeError:
                        pass
        usage = resp.usage
        return CallResult(
            ok=True,
            latency_s=dt,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            raw=[asdict_safe(tc) for tc in (msg.tool_calls or [])],
        ), rels
    except Exception as e:
        return CallResult(
            ok=False, latency_s=time.time() - t0, error=f"{type(e).__name__}: {e}"
        ), []


def asdict_safe(tc: Any) -> dict:
    try:
        return {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        }
    except Exception:
        return {"repr": repr(tc)}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


def edge_matches(rels: list[dict], expected: tuple[str, str, str]) -> bool:
    """A relation matches if source, relationship, and destination each contain the expected substrings (case-insensitive)."""
    src_e, rel_e, dst_e = (s.lower() for s in expected)
    for r in rels:
        src = (r.get("source") or "").lower()
        rel = (r.get("relationship") or "").lower()
        dst = (r.get("destination") or "").lower()
        if src_e in src and rel_e in rel and dst_e in dst:
            return True
    return False


def score_case(case: TestCase, entities: list[dict], rels: list[dict]) -> dict:
    entity_names = [(e.get("entity") or "").lower() for e in entities]

    expected_hits = sum(1 for ee in case.expected_edges if edge_matches(rels, ee))
    missing_edges = [ee for ee in case.expected_edges if not edge_matches(rels, ee)]

    compound_ok = (
        all(
            any(comp.lower() in n for n in entity_names)
            for comp in case.must_preserve_compound
        )
        if case.must_preserve_compound
        else True
    )

    forbidden_present = []
    for sub in case.forbidden_entity_substrings:
        for n in entity_names:
            if sub.lower() in n:
                forbidden_present.append(n)
                break
        for r in rels:
            for fld in ("source", "destination"):
                v = (r.get(fld) or "").lower()
                if sub.lower() in v:
                    forbidden_present.append(v)
                    break

    self_loops = sum(
        1
        for r in rels
        if (r.get("source") or "").lower() == (r.get("destination") or "").lower()
    )

    forbidden_edge_hits = [fe for fe in case.forbidden_edges if edge_matches(rels, fe)]

    return {
        "expected_edges_hit": expected_hits,
        "expected_edges_total": len(case.expected_edges),
        "missing_edges": missing_edges,
        "compound_preserved": compound_ok,
        "forbidden_substring_hits": forbidden_present,
        "forbidden_edge_hits": forbidden_edge_hits,
        "self_loops": self_loops,
        "entity_count": len(entities),
        "relation_count": len(rels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


def run_one(client: openai.OpenAI, model: str, case: TestCase) -> dict:
    print(f"  [{model}] {case.name} ...", flush=True)
    out: dict = {"model": model, "case": case.name}

    vec_call, facts = call_vector(client, model, case.text)
    out["vector"] = {**asdict(vec_call), "facts": facts}

    ent_call, entities = call_entities(client, model, case.text)
    out["entities_call"] = {**asdict(ent_call), "entities": entities}

    rel_aug_call, rel_aug = call_relations(
        client, model, case.text, entities, augment=True
    )
    out["relations_aug"] = {
        **asdict(rel_aug_call),
        "relations": rel_aug,
        "score": score_case(case, entities, rel_aug),
    }

    rel_no_call, rel_no = call_relations(
        client, model, case.text, entities, augment=False
    )
    out["relations_noaug"] = {
        **asdict(rel_no_call),
        "relations": rel_no,
        "score": score_case(case, entities, rel_no),
    }

    return out


def aggregate(results: list[dict]) -> dict:
    by_model: dict[str, dict] = {}
    for r in results:
        m = r["model"]
        agg = by_model.setdefault(
            m,
            {
                "vector_facts_total": 0,
                "vector_calls_ok": 0,
                "vector_calls_total": 0,
                "entity_calls_ok": 0,
                "entity_calls_total": 0,
                "expected_edges_hit_aug": 0,
                "expected_edges_hit_noaug": 0,
                "expected_edges_total": 0,
                "compound_preserved_aug": 0,
                "compound_preserved_noaug": 0,
                "compound_total": 0,
                "forbidden_hits_aug": 0,
                "forbidden_hits_noaug": 0,
                "forbidden_edges_aug": 0,
                "forbidden_edges_noaug": 0,
                "self_loops_aug": 0,
                "self_loops_noaug": 0,
                "latency_total_s": 0.0,
                "latency_calls": 0,
                "errors": [],
            },
        )
        agg["vector_calls_total"] += 1
        if r["vector"]["ok"]:
            agg["vector_calls_ok"] += 1
            agg["vector_facts_total"] += len(r["vector"]["facts"])
        else:
            agg["errors"].append(f"{r['case']}/vector: {r['vector']['error']}")
        agg["entity_calls_total"] += 1
        if r["entities_call"]["ok"]:
            agg["entity_calls_ok"] += 1
        else:
            agg["errors"].append(f"{r['case']}/entities: {r['entities_call']['error']}")

        for tag, key in (("aug", "relations_aug"), ("noaug", "relations_noaug")):
            block = r[key]
            if block["ok"]:
                s = block["score"]
                agg[f"expected_edges_hit_{tag}"] += s["expected_edges_hit"]
                agg["expected_edges_total"] += (
                    s["expected_edges_total"] if tag == "aug" else 0
                )
                agg[f"compound_preserved_{tag}"] += 1 if s["compound_preserved"] else 0
                agg["compound_total"] += 1 if tag == "aug" else 0
                agg[f"forbidden_hits_{tag}"] += len(s["forbidden_substring_hits"])
                agg[f"forbidden_edges_{tag}"] += len(s.get("forbidden_edge_hits", []))
                agg[f"self_loops_{tag}"] += s["self_loops"]
            else:
                agg["errors"].append(f"{r['case']}/{key}: {block['error']}")

        for k in ("vector", "entities_call", "relations_aug", "relations_noaug"):
            if r[k]["ok"]:
                agg["latency_total_s"] += r[k]["latency_s"]
                agg["latency_calls"] += 1

    return by_model


def render_markdown(by_model: dict, results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# LiteLLM proxy model benchmark")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append(f"Test cases: {[c.name for c in CASES]}")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append(
        "| Model | Vector facts (total) | Edges hit (aug) | Edges hit (no-aug) | Compound preserved (aug/no-aug) | Forbidden hits (aug/no-aug) | Bad edges (aug/no-aug) | Self-loops (aug/no-aug) | Avg latency/call (s) | Errors |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        a = by_model.get(m, {})
        if not a:
            lines.append(f"| {m} | (no data) | | | | | | | |")
            continue
        avg_lat = (
            (a["latency_total_s"] / a["latency_calls"]) if a["latency_calls"] else 0
        )
        lines.append(
            f"| `{m}` "
            f"| {a['vector_facts_total']} ({a['vector_calls_ok']}/{a['vector_calls_total']} OK) "
            f"| {a['expected_edges_hit_aug']}/{a['expected_edges_total']} "
            f"| {a['expected_edges_hit_noaug']}/{a['expected_edges_total']} "
            f"| {a['compound_preserved_aug']}/{a['compound_total']} · {a['compound_preserved_noaug']}/{a['compound_total']} "
            f"| {a['forbidden_hits_aug']} · {a['forbidden_hits_noaug']} "
            f"| {a['forbidden_edges_aug']} · {a['forbidden_edges_noaug']} "
            f"| {a['self_loops_aug']} · {a['self_loops_noaug']} "
            f"| {avg_lat:.2f} "
            f"| {len(a['errors'])} |"
        )
    lines.append("")
    lines.append("## Per-case relation extraction (augmented prompt)")
    lines.append("")
    for case in CASES:
        lines.append(f"### `{case.name}`")
        lines.append(f"_Input_: {case.text}")
        lines.append("")
        lines.append("| Model | Entities | Relations | Hits | Missing |")
        lines.append("|---|---|---|---|---|")
        for m in MODELS:
            r = next(
                (x for x in results if x["model"] == m and x["case"] == case.name), None
            )
            if not r:
                continue
            ents = r["entities_call"].get("entities", [])
            rels = r["relations_aug"].get("relations", [])
            score = r["relations_aug"].get("score", {})
            ent_str = ", ".join(e.get("entity", "?") for e in ents)[:120]
            rel_str = "; ".join(
                f"{x.get('source')}-[{x.get('relationship')}]->{x.get('destination')}"
                for x in rels
            )[:200]
            missing = score.get("missing_edges", [])
            lines.append(
                f"| `{m}` | {ent_str} | {rel_str} | {score.get('expected_edges_hit', 0)}/{score.get('expected_edges_total', 0)} | {len(missing)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    client = make_client()
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results: list[dict] = []
    for model in MODELS:
        print(f"=== {model} ===", flush=True)
        for case in CASES:
            try:
                results.append(run_one(client, model, case))
            except Exception as e:
                print(f"  [{model}] {case.name} FATAL: {e}", flush=True)
                results.append(
                    {
                        "model": model,
                        "case": case.name,
                        "vector": {
                            "ok": False,
                            "latency_s": 0,
                            "error": str(e),
                            "facts": [],
                        },
                        "entities_call": {
                            "ok": False,
                            "latency_s": 0,
                            "error": str(e),
                            "entities": [],
                        },
                        "relations_aug": {
                            "ok": False,
                            "latency_s": 0,
                            "error": str(e),
                            "relations": [],
                            "score": score_case(case, [], []),
                        },
                        "relations_noaug": {
                            "ok": False,
                            "latency_s": 0,
                            "error": str(e),
                            "relations": [],
                            "score": score_case(case, [], []),
                        },
                    }
                )

    by_model = aggregate(results)

    raw_path = out_dir / f"proxy_compare_{stamp}.json"
    md_path = out_dir / f"proxy_compare_{stamp}.md"
    raw_path.write_text(json.dumps({"results": results, "summary": by_model}, indent=2))
    md_path.write_text(render_markdown(by_model, results))

    print()
    print(render_markdown(by_model, results))
    print()
    print(f"Raw results: {raw_path}")
    print(f"Markdown:    {md_path}")


if __name__ == "__main__":
    main()
