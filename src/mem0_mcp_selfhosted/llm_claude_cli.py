"""Claude CLI subprocess LLM provider for mem0ai.

Shells out to `claude -p` for fact extraction, memory update, and graph
extraction calls, authenticating via the Claude Code OAT (the user's existing
subscription) rather than a separate ANTHROPIC_API_KEY. Useful for testing
extraction quality with Sonnet/Haiku without burning API credits.

Trade-offs vs the SDK-based AnthropicOATLLM provider:
- Slower: subprocess startup adds ~3-5 s per call.
- Cap-shared: every extraction counts against the same Claude Code rolling
  window as interactive sessions. A busy daemon will rate-limit your CLI.
- Tool-calling is simulated, not native. The Claude CLI has no tool-call
  surface, so when `tools` are passed (graph extraction) we coerce the call
  into a `--json-schema`-validated structured-output call against the chosen
  tool's `parameters` schema, then synthesize the
  ``{content, tool_calls: [{name, arguments}]}`` envelope mem0ai expects.
  Single-tool flows (mem0ai's graph extraction is always single-tool) are
  exact; multi-tool/auto-choice falls back to the first tool.

Cleanliness: invokes claude with --tools "" --strict-mcp-config
--mcp-config '{"mcpServers":{}}' --setting-sources "" so the call
behaves as a pure LLM round-trip with no MCP servers, hooks, plugins,
or auto-context loaded. The system prompt is fully replaced via
--system-prompt; --json-schema enforces structured output for both
response_format and tool-calling paths.

Usage tracking: session persistence is left ON so each call writes a
JSONL transcript under ~/.claude/projects/<encoded-cwd>/, which local
trackers (ccusage etc.) read to count tokens. The subprocess cwd is
pinned via MEM0_CLAUDE_CLI_CWD (default ~/.cache/mem0-extractor) so
those one-shot transcripts cluster in a dedicated project folder
instead of polluting whichever cwd the MCP server happened to inherit.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase

from mem0_mcp_selfhosted.env import env

logger = logging.getLogger(__name__)


FACT_RETRIEVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["facts"],
    "additionalProperties": False,
}

MEMORY_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "memory": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "event": {
                        "type": "string",
                        "enum": ["ADD", "UPDATE", "DELETE", "NONE"],
                    },
                    "old_memory": {"type": "string"},
                },
                "required": ["id", "text", "event"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["memory"],
    "additionalProperties": False,
}


def _default_transcript_cwd() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "mem0-extractor")


class ClaudeCliConfig(BaseLlmConfig):
    """Config for the Claude CLI subprocess provider."""

    def __init__(
        self,
        cli_path: str | None = None,
        timeout_seconds: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cli_path = cli_path or env("MEM0_CLAUDE_CLI_PATH", "claude")
        self.timeout_seconds = timeout_seconds or int(
            env("MEM0_CLAUDE_CLI_TIMEOUT", "120")
        )
        self.cwd = env("MEM0_CLAUDE_CLI_CWD", "") or _default_transcript_cwd()


class ClaudeCliLLM(LLMBase):
    """LLM provider that invokes the Claude Code CLI as a subprocess."""

    def __init__(self, config: ClaudeCliConfig | None = None):
        super().__init__(config)
        if not isinstance(self.config, ClaudeCliConfig):
            base = self.config
            self.config = ClaudeCliConfig(
                model=base.model,
                api_key=base.api_key,
                max_tokens=base.max_tokens,
                temperature=base.temperature,
                top_p=base.top_p,
                top_k=base.top_k,
            )

    @staticmethod
    def _select_schema(messages: list[dict]) -> dict:
        """Match AnthropicOATLLM's heuristic — system message → fact extraction."""
        has_system = any(m.get("role") == "system" for m in messages)
        return FACT_RETRIEVAL_SCHEMA if has_system else MEMORY_UPDATE_SCHEMA

    def _build_argv(
        self,
        system_prompt: str,
        schema: dict | None,
    ) -> list[str]:
        argv: list[str] = [
            self.config.cli_path,
            "-p",
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--setting-sources",
            "",
            "--output-format",
            "json",
            "--model",
            self.config.model,
            "--system-prompt",
            system_prompt,
        ]
        if schema is not None:
            argv += ["--json-schema", json.dumps(schema)]
        return argv

    @staticmethod
    def _resolve_tool(
        tools: list[dict],
        tool_choice: str | dict | None,
    ) -> tuple[str, str, dict]:
        """Pick a single target tool and return (name, description, parameters_schema).

        Handles both OpenAI-style (``{"type":"function","function":{...}}``) and
        Anthropic-style (``{"name","input_schema"}``) tool defs. mem0ai's graph
        layer always passes a single tool with ``tool_choice`` set to that tool's
        name or "required", so the picking rules below are conservative.
        """
        target: str | None = None
        if isinstance(tool_choice, dict):
            target = tool_choice.get("name") or tool_choice.get("function", {}).get(
                "name"
            )
        elif isinstance(tool_choice, str) and tool_choice not in (
            "auto",
            "required",
            "none",
            "any",
        ):
            target = tool_choice

        chosen: dict | None = None
        if target is not None:
            for t in tools:
                fn = t.get("function") if "function" in t else t
                if fn.get("name") == target:
                    chosen = t
                    break
        if chosen is None:
            chosen = tools[0]

        fn = chosen.get("function") if "function" in chosen else chosen
        name = fn.get("name", "tool")
        description = fn.get("description", "")
        parameters = (
            fn.get("parameters")
            or fn.get("input_schema")
            or {
                "type": "object",
                "properties": {},
            }
        )
        return name, description, parameters

    def _call_cli(
        self,
        system_prompt: str,
        user_input: str,
        schema: dict | None,
    ) -> dict:
        """Run the CLI subprocess and return the parsed JSON envelope."""
        argv = self._build_argv(system_prompt, schema)
        # Pin cwd so the CLI writes its transcript JSONL into a single
        # dedicated ~/.claude/projects/<encoded-cwd>/ folder. ccusage and
        # similar trackers tail those files; without this every extraction
        # call would land under whichever cwd the MCP server inherited
        # (typically a real project), polluting per-project token totals.
        cwd = self.config.cwd
        os.makedirs(cwd, exist_ok=True)
        try:
            proc = subprocess.run(
                argv,
                input=user_input,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"claude_cli timed out after {self.config.timeout_seconds}s"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude_cli exited {proc.returncode}: "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"claude_cli returned non-JSON stdout: {proc.stdout[:500]!r}"
            ) from exc

        if payload.get("is_error"):
            raise RuntimeError(
                f"claude_cli reported error: "
                f"{payload.get('result') or payload.get('api_error_status')}"
            )
        return payload

    def generate_response(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ):
        # Split system / non-system messages.
        system_parts: list[str] = []
        user_parts: list[str] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
        system_prompt = "\n\n".join(system_parts) if system_parts else ""
        user_input = "\n\n".join(user_parts)

        # --- Tool-calling path: simulate via JSON-schema-enforced output ---
        # The Claude CLI has no tool-call surface. mem0ai's graph layer expects
        # a {"content","tool_calls"[{name,arguments}]} dict. We coerce the call
        # to a structured-output call against the tool's `parameters` schema and
        # synthesize the tool_calls envelope from the validated JSON.
        if tools and tool_choice != "none":
            name, description, parameters = self._resolve_tool(tools, tool_choice)
            tool_directive = (
                f"\n\nReturn ONLY a JSON object matching the schema for the "
                f"`{name}` operation. Do not call any tool, do not explain, "
                f"do not wrap output in code fences. The runtime validates the "
                f"JSON against the schema."
            )
            if description:
                tool_directive += f" Operation description: {description}"
            tool_system = (
                (system_prompt + tool_directive)
                if system_prompt
                else tool_directive.lstrip()
            )

            payload = self._call_cli(tool_system, user_input, parameters)
            structured = payload.get("structured_output")
            if structured is None:
                logger.warning(
                    "[mem0] claude_cli tool-shim: structured_output missing; "
                    "returning empty tool_calls"
                )
                return {"content": payload.get("result", "") or "", "tool_calls": []}
            return {
                "content": "",
                "tool_calls": [{"name": name, "arguments": structured}],
            }

        # --- Structured-output path (no tools) ---
        if response_format:
            schema = self._select_schema(messages)
            payload = self._call_cli(system_prompt, user_input, schema)
            structured = payload.get("structured_output")
            if structured is None:
                logger.warning(
                    "[mem0] claude_cli structured_output missing; "
                    "falling back to result text"
                )
                return payload.get("result", "") or ""
            return json.dumps(structured)

        # --- Plain text path ---
        payload = self._call_cli(system_prompt, user_input, None)
        return payload.get("result", "") or ""
