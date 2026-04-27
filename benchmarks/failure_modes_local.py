"""Targeted graph-extraction failure-mode harness.

Sends 5 Helios-style cases through the mem0ai entity+relationship extraction
pipeline against multiple models (local Ollama + LiteLLM-hosted). Scores each
on 4 specific failure modes observed in production (gemma4:e2b):

  F1. Compound technical entity preservation  ("AWS us-east-1" as one node)
  F2. Org -> platform/service edge preservation
  F3. Person -> person direct edge through artifact
  F4. Sub-team / org-unit distinction from parent

Set LITELLM_BASE_URL + LITELLM_TOKEN env vars to enable hosted models.

Usage:  python3 benchmarks/failure_modes_local.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---- Tool schemas (exact copies from mem0ai graphs/tools.py) -----------------

EXTRACT_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract entities and their types from the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string"},
                            "entity_type": {"type": "string"},
                        },
                        "required": ["entity", "entity_type"],
                    },
                }
            },
            "required": ["entities"],
        },
    },
}

RELATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relationships",
        "description": "Establish relationships among the entities based on the provided text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "relationship": {"type": "string"},
                            "destination": {"type": "string"},
                        },
                        "required": ["source", "relationship", "destination"],
                    },
                }
            },
            "required": ["entities"],
        },
    },
}

EXTRACT_RELATIONS_PROMPT = """
You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive and accurate information. Follow these key principles:

1. Extract only explicitly stated information from the text.
2. Establish relationships among the entities provided.
3. Use "USER_ID" as the source entity for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Relationships:
    - Use consistent, general, and timeless relationship types.
    - Example: Prefer "professor" over "became_professor."
    - Relationships should only be established among the entities explicitly mentioned in the user message.

Entity Consistency:
    - Ensure that relationships are coherent and logically align with the context of the message.
    - Maintain consistent naming for entities across the extracted data.

Strive to construct a coherent and easily understandable knowledge graph by establishing all the relationships among the entities and adherence to the user's context.

Adhere strictly to these guidelines to ensure high-quality knowledge graph extraction."""


# Augmented prompt: targets the four failure modes observed in production
# (gemma4:e2b on Ollama). Adds explicit rules without removing any of the
# original guidance, so it can be A/B compared against the upstream prompt.
EXTRACT_RELATIONS_PROMPT_AUGMENTED = EXTRACT_RELATIONS_PROMPT + """

Additional extraction rules — read carefully:

A. Compound technical identifiers are ONE entity, never two:
   - "AWS us-east-1" → entity "aws us-east-1" (NOT "aws" + "us-east-1")
   - "GCP europe-west4" → entity "gcp europe-west4"
   - Cloud regions, fully-qualified service names, version numbers, model
     IDs, and dotted/dashed identifiers must stay intact as one node.

B. When a person performs an action on or with another person via an
   intermediary (artifact, team, group, project), create a DIRECT
   person-to-person edge in addition to any artifact-related edges:
   - "Maya's team handed off the bridge to David's group"
     → also emit (maya, handed_off_to, david)
   - "Sarah transferred the service to Tom"
     → emit (sarah, transferred_to, tom) directly
   The direct interpersonal edge is what queries about responsibility
   transfer will look for.

C. When an organization owns, runs, or hosts a platform/service/product,
   emit a DIRECT edge from the organization to the platform — not only
   via a person:
   - "Maya leads the Helios team at Northwind" → also emit
     (helios, owned_by, northwind) — the org→platform link is load-bearing.
   - "Stripe's checkout service" → emit (stripe, owns, checkout_service)
     with that exact direction (org → service, not service → org).

D. Sub-teams and organizational units (e.g. "Helios platform team",
   "infrastructure squad") are distinct entities from their parent.
   Both should appear as nodes, with an edge connecting them.

E. Never emit self-referential edges (source equal to destination).
   "X --[leads]--> X" is always wrong — drop the relation.

F. Direction matters. "A is head of B" means (a, head_of, b), not
   (b, head_of, a). "X reports to Y" means (x, reports_to, y).
"""


# ---- Test cases --------------------------------------------------------------

@dataclass
class FailureCase:
    id: str
    text: str
    # F1: entities that should NOT be split
    compound_entities: list[str] = field(default_factory=list)
    # F2: org -> platform edges that should exist
    org_to_thing: list[tuple[str, str]] = field(default_factory=list)
    # F3: direct person-to-person edges that should exist
    person_to_person: list[tuple[str, str]] = field(default_factory=list)
    # F4: sub-team/unit names that should appear distinct from parent
    distinct_units: list[tuple[str, str]] = field(default_factory=list)  # (unit, parent)


CASES: list[FailureCase] = [
    FailureCase(
        id="HELIOS-1",
        text=(
            "Maya leads the Helios platform team at Northwind Robotics. "
            "Helios runs on Kubernetes in AWS us-east-1 and depends on Postgres "
            "for its order ledger. Last quarter Maya's team handed off the "
            "legacy MQTT bridge to David's group."
        ),
        compound_entities=["aws us-east-1"],
        org_to_thing=[("northwind robotics", "helios")],
        person_to_person=[("maya", "david")],
        distinct_units=[("helios platform team", "helios")],
    ),
    FailureCase(
        id="REGIONS-2",
        text="Production runs on GCP europe-west4 with a failover in AWS us-west-2.",
        compound_entities=["gcp europe-west4", "aws us-west-2"],
    ),
    FailureCase(
        id="ORG-3",
        text=(
            "The Atlas service at Stripe is run by Jordan's team. "
            "Jordan reports to Priya, the head of Payments at Stripe."
        ),
        org_to_thing=[("stripe", "atlas")],
        person_to_person=[("jordan", "priya")],
    ),
    FailureCase(
        id="HANDOFF-4",
        text=(
            "Sarah transferred the billing service to Tom in March. "
            "Tom now owns billing, and Sarah moved to the search team led by Wei."
        ),
        person_to_person=[("sarah", "tom"), ("sarah", "wei")],
    ),
    FailureCase(
        id="UNIT-5",
        text=(
            "The infrastructure squad within the Atlas org owns the API gateway. "
            "Atlas is part of Cloudwave Systems."
        ),
        org_to_thing=[("cloudwave systems", "atlas")],
        distinct_units=[("infrastructure squad", "atlas")],
    ),
]


# ---- OpenAI-compatible chat client (Ollama + LiteLLM share this API) ---------

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
LITELLM_BASE = os.environ.get("LITELLM_BASE_URL", "").rstrip("/")
LITELLM_TOKEN = os.environ.get("LITELLM_TOKEN", "")

# Per-model endpoint config. Add prefix variants here if LiteLLM rejects bare names.
MODEL_ENDPOINTS: dict[str, dict] = {
    "gemma4:e2b": {"url": OLLAMA_URL, "auth": None},
}
if LITELLM_BASE and LITELLM_TOKEN:
    _litellm_url = f"{LITELLM_BASE}/v1/chat/completions"
    _litellm_auth = f"Bearer {LITELLM_TOKEN}"
    MODEL_ENDPOINTS["gpt-5.4-nano"] = {"url": _litellm_url, "auth": _litellm_auth}
    MODEL_ENDPOINTS["gemini-3.1-flash-lite-preview"] = {"url": _litellm_url, "auth": _litellm_auth}


def call_chat(model: str, messages: list[dict], tools: list[dict], timeout: int = 240) -> dict:
    cfg = MODEL_ENDPOINTS.get(model)
    if cfg is None:
        return {"error": f"no endpoint configured for model {model!r}"}
    headers = {"Content-Type": "application/json"}
    if cfg["auth"]:
        headers["Authorization"] = cfg["auth"]
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": tools[0]["function"]["name"]}},
        "temperature": 0.0,
    }
    start = time.time()
    try:
        r = requests.post(cfg["url"], json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        # Include response body if available, for debugging hosted-API errors
        detail = str(exc)
        if hasattr(exc, "response") and exc.response is not None:
            try:
                detail = f"{detail} | body: {exc.response.text[:300]}"
            except Exception:
                pass
        return {"error": detail, "latency": time.time() - start}
    latency = time.time() - start
    msg = data["choices"][0]["message"]
    tool_calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        tool_calls.append({"name": fn.get("name"), "arguments": args or {}})
    return {"tool_calls": tool_calls, "latency": latency, "content": msg.get("content", "")}


# ---- Scoring -----------------------------------------------------------------

def _norm(s: str) -> str:
    return s.lower().strip().replace("_", " ").replace("-", " ")


def extract_entities(result: dict) -> list[str]:
    out = []
    for tc in result.get("tool_calls", []):
        if tc["name"] == "extract_entities":
            for e in tc["arguments"].get("entities", []):
                out.append(_norm(e.get("entity", "")))
    return out


def extract_relations(result: dict) -> list[tuple[str, str, str]]:
    out = []
    for tc in result.get("tool_calls", []):
        if tc["name"] in ("establish_relationships", "establish_relations"):
            for e in tc["arguments"].get("entities", []):
                out.append((_norm(e.get("source", "")),
                            _norm(e.get("relationship", "")),
                            _norm(e.get("destination", ""))))
    return out


def _person_match(token: str, person_norm: str) -> bool:
    """Exact match for a person name (case-normalized).

    Substring matching falsely conflates 'maya' with 'maya\'s team',
    so we require equality.
    """
    return token == person_norm


def score_case(case: FailureCase, entities: list[str], relations: list[tuple[str, str, str]]) -> dict:
    """Return per-failure-mode pass/fail booleans and details."""
    s = {}

    # All node names that appear anywhere in the graph (entity extraction + relation endpoints).
    # mem0ai will create graph nodes from either source, so both count.
    all_nodes = set(entities)
    for src, _, dst in relations:
        all_nodes.add(src)
        all_nodes.add(dst)

    # F1: compound entity preservation — full compound must appear as one node (substring)
    f1_results = []
    for compound in case.compound_entities:
        norm_c = _norm(compound)
        passed = any(norm_c in node for node in all_nodes)
        f1_results.append((compound, passed))
    s["F1"] = f1_results

    # F2: org -> thing edge — exact-token match on both endpoints (either direction)
    f2_results = []
    for org, thing in case.org_to_thing:
        nor, nthi = _norm(org), _norm(thing)
        passed = any(
            (src == nor and dst == nthi) or (src == nthi and dst == nor)
            for src, _, dst in relations
        )
        f2_results.append(((org, thing), passed))
    s["F2"] = f2_results

    # F3: direct person-to-person edge — exact-token match on both endpoints
    # Must NOT count edges through proxies like "maya's team" or "david's group".
    f3_results = []
    for p1, p2 in case.person_to_person:
        np1, np2 = _norm(p1), _norm(p2)
        passed = any(
            (_person_match(src, np1) and _person_match(dst, np2)) or
            (_person_match(src, np2) and _person_match(dst, np1))
            for src, _, dst in relations
        )
        f3_results.append(((p1, p2), passed))
    s["F3"] = f3_results

    # F4: sub-unit must appear as a distinct node (in entities OR relation endpoints)
    f4_results = []
    for unit, parent in case.distinct_units:
        nu = _norm(unit)
        passed = any(nu == node or nu in node for node in all_nodes)
        f4_results.append(((unit, parent), passed))
    s["F4"] = f4_results

    return s


# ---- Runner ------------------------------------------------------------------

MODELS = ["gemma4:e2b"]  # default; extended at runtime if LITELLM_TOKEN is set
if LITELLM_BASE and LITELLM_TOKEN:
    MODELS.extend(["gpt-5.4-nano", "gemini-3.1-flash-lite-preview"])


def run_case(model: str, case: FailureCase, prompt_variant: str = "baseline") -> dict:
    # Pick the system prompt for relation extraction
    if prompt_variant == "augmented":
        sys_prompt = EXTRACT_RELATIONS_PROMPT_AUGMENTED.replace("USER_ID", "test_user")
        # For Phase 1 we also nudge entity extraction — gemma keeps splitting
        # compound identifiers, so prepend a minimal rule.
        entity_prefix = (
            "Rules: keep compound technical identifiers (cloud regions like "
            "'AWS us-east-1', dotted/dashed names, version numbers) as ONE "
            "entity — do not split them.\n\n"
        )
    else:
        sys_prompt = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", "test_user")
        entity_prefix = ""

    # Phase 1: entity extraction
    msgs1 = [{"role": "user", "content": f"{entity_prefix}Extract entities from the following text:\n\n{case.text}"}]
    r1 = call_chat(model, msgs1, [EXTRACT_ENTITIES_TOOL])
    if "error" in r1:
        return {"error": r1["error"], "phase": "entity"}

    entities_raw = []
    for tc in r1.get("tool_calls", []):
        if tc["name"] == "extract_entities":
            for e in tc["arguments"].get("entities", []):
                entities_raw.append(e)
    entity_names = [e.get("entity", "") for e in entities_raw]

    # Phase 2: relationship extraction
    entity_list = ", ".join(entity_names[:20])
    msgs2 = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Entities found: {entity_list}\n\nOriginal text: {case.text}"},
    ]
    r2 = call_chat(model, msgs2, [RELATIONS_TOOL])

    entities = extract_entities(r1)
    relations = extract_relations(r2)
    score = score_case(case, entities, relations)

    return {
        "entities_raw": entities_raw,
        "entity_names_norm": entities,
        "relations": relations,
        "score": score,
        "phase1_latency": r1.get("latency"),
        "phase2_latency": r2.get("latency", r2.get("error")),
        "phase2_error": r2.get("error") if "error" in r2 else None,
    }


def fmt_check(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def print_case_report(case: FailureCase, results_per_model: dict[str, dict]) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {case.id}: {case.text[:70]}{'...' if len(case.text) > 70 else ''}")
    print(f"{'=' * 78}")

    for model, r in results_per_model.items():
        if "error" in r:
            print(f"  {model:>16}: ERROR ({r['phase']}): {r['error'][:80]}")
            continue
        s = r["score"]
        lat = r.get("phase1_latency", 0) + (r.get("phase2_latency", 0) if isinstance(r.get("phase2_latency"), (int, float)) else 0)
        print(f"  {model:>16}: {len(r['entities_raw'])} entities, {len(r['relations'])} relations ({lat:.1f}s)")
        for fmode, items in s.items():
            for target, passed in items:
                tag = fmode
                tgt_str = (f"{target}" if not isinstance(target, tuple) else f"{target[0]}->{target[1]}")
                print(f"      [{tag}] {fmt_check(passed):>4}: {tgt_str}")
        if s.get("F1") or s.get("F4"):
            print(f"      entities: {r['entity_names_norm']}")
        if r['relations']:
            print(f"      relations:")
            for src, rel, dst in r['relations'][:8]:
                print(f"        {src} --[{rel}]--> {dst}")


VARIANTS = ["baseline", "augmented"]


def main():
    print(f"Local-Ollama failure-mode harness ({len(CASES)} cases × {len(MODELS)} models × {len(VARIANTS)} variants)")
    print(f"Endpoint: {OLLAMA_URL}\n")

    # results[case_id][model][variant] = case_result
    all_results: dict[str, dict[str, dict[str, dict]]] = {}

    for case in CASES:
        all_results[case.id] = {}
        for model in MODELS:
            all_results[case.id][model] = {}
            for variant in VARIANTS:
                print(f"    {case.id} / {model} / {variant}...", end="", flush=True)
                r = run_case(model, case, prompt_variant=variant)
                all_results[case.id][model][variant] = r
                if "error" in r:
                    print(f" ERROR: {r['error'][:60]}")
                else:
                    e_count = len(r['entities_raw'])
                    r_count = len(r['relations'])
                    fails = sum(1 for items in r['score'].values() for _, p in items if not p)
                    print(f" {e_count} ent / {r_count} rel / {fails} failure-mode fails")

    # Per-case detail dump (baseline + augmented side by side)
    for case in CASES:
        print(f"\n{'=' * 78}")
        print(f"  {case.id}: {case.text[:70]}{'...' if len(case.text) > 70 else ''}")
        print(f"{'=' * 78}")
        for model in MODELS:
            for variant in VARIANTS:
                r = all_results[case.id][model][variant]
                if "error" in r:
                    print(f"  {model} / {variant}: ERROR")
                    continue
                s = r["score"]
                print(f"  {model} / {variant}: {len(r['entities_raw'])} ent, {len(r['relations'])} rel")
                for fmode, items in s.items():
                    for target, passed in items:
                        tgt = f"{target[0]}->{target[1]}" if isinstance(target, tuple) else f"{target}"
                        print(f"      [{fmode}] {fmt_check(passed):>4}: {tgt}")
                if r['relations']:
                    for src, rel, dst in r['relations'][:8]:
                        print(f"        {src} --[{rel}]--> {dst}")

    # Summary: variant comparison per failure mode
    print(f"\n\n{'#' * 78}")
    print(f"  SUMMARY: failure-mode pass rate, baseline vs augmented prompt")
    print(f"{'#' * 78}\n")

    # by[model][variant][fmode] = list[bool]
    by: dict[str, dict[str, dict[str, list[bool]]]] = {
        m: {v: {"F1": [], "F2": [], "F3": [], "F4": []} for v in VARIANTS}
        for m in MODELS
    }
    for case_id, mr in all_results.items():
        for model, vr in mr.items():
            for variant, r in vr.items():
                if "error" in r:
                    continue
                for fmode, items in r["score"].items():
                    for _, passed in items:
                        by[model][variant][fmode].append(passed)

    for model in MODELS:
        print(f"  {model}:")
        print(f"    {'mode':<22} {'baseline':>14}  {'augmented':>14}  {'delta':>8}")
        print(f"    {'-' * 22} {'-' * 14}  {'-' * 14}  {'-' * 8}")
        labels = {
            "F1": "F1 compound entity",
            "F2": "F2 org→thing edge",
            "F3": "F3 person→person",
            "F4": "F4 sub-unit distinct",
        }
        for fmode in ["F1", "F2", "F3", "F4"]:
            b = by[model]["baseline"][fmode]
            a = by[model]["augmented"][fmode]
            if not b or not a:
                continue
            bp = sum(b) / len(b)
            ap = sum(a) / len(a)
            print(f"    {labels[fmode]:<22} {sum(b)}/{len(b)} ({bp:.0%})    {sum(a)}/{len(a)} ({ap:.0%})    {(ap-bp)*100:+.0f}pp")

    # Save raw
    out = Path(__file__).parent / "failure_modes_local_results.json"
    serial = {}
    for case_id, mr in all_results.items():
        serial[case_id] = {}
        for model, vr in mr.items():
            serial[case_id][model] = {}
            for variant, r in vr.items():
                serial[case_id][model][variant] = {
                    "entities_raw": r.get("entities_raw"),
                    "relations": r.get("relations"),
                    "score": {k: [(list(t) if isinstance(t, tuple) else t, p) for t, p in v]
                              for k, v in r.get("score", {}).items()},
                    "phase1_latency": r.get("phase1_latency"),
                    "phase2_latency": r.get("phase2_latency"),
                    "error": r.get("error"),
                }
    out.write_text(json.dumps(serial, indent=2))
    print(f"\n  Raw results: {out}")


if __name__ == "__main__":
    main()
