"""Tests for llm_ollama.py — OllamaToolLLM with defense-in-depth layers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mem0.llms.ollama import OllamaLLM

from mem0_mcp_selfhosted.llm_ollama import (
    OllamaToolLLM,
    _strip_think_tags,
    extract_json,
)


@pytest.fixture
def mock_ollama_config():
    """Create a mock OllamaConfig for OllamaToolLLM."""
    config = MagicMock()
    config.model = "qwen3:14b"
    config.temperature = 0.7
    config.max_tokens = 8192
    config.top_p = 0.9
    config.ollama_base_url = "http://localhost:11434"
    return config


@pytest.fixture
def ollama_llm(mock_ollama_config):
    """Create an OllamaToolLLM with mocked client."""
    with patch.object(OllamaToolLLM, "__init__", lambda self, config=None: None):
        llm = OllamaToolLLM.__new__(OllamaToolLLM)
        llm.config = mock_ollama_config
        llm.client = MagicMock()
        return llm


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "Extract entities from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {"type": "array"},
                },
            },
        },
    }
]


class TestSubclass:
    def test_is_subclass_of_ollama_llm(self):
        """OllamaToolLLM inherits from upstream OllamaLLM."""
        assert issubclass(OllamaToolLLM, OllamaLLM)

    def test_isinstance_check(self, ollama_llm):
        """Instance passes isinstance check for OllamaLLM."""
        assert isinstance(ollama_llm, OllamaLLM)


class TestExtractJson:
    def test_code_fenced_json(self):
        """Code-fenced JSON is extracted."""
        text = '```json\n{"facts": ["a"]}\n```'
        assert extract_json(text) == '{"facts": ["a"]}'

    def test_code_fenced_without_lang(self):
        """Code fence without 'json' label is extracted."""
        text = '```\n{"facts": ["a"]}\n```'
        assert extract_json(text) == '{"facts": ["a"]}'

    def test_text_prefixed_json(self):
        """Text-prefixed JSON is extracted."""
        text = 'Here is the JSON: {"facts": ["a"]}'
        assert extract_json(text) == '{"facts": ["a"]}'

    def test_clean_json_passthrough(self):
        """Clean JSON passes through unchanged."""
        text = '{"facts": ["a"]}'
        assert extract_json(text) == '{"facts": ["a"]}'

    def test_empty_string_passthrough(self):
        """Empty string passes through unchanged."""
        assert extract_json("") == ""

    def test_whitespace_only_passthrough(self):
        """Whitespace-only string passes through."""
        assert extract_json("   ") == ""

    def test_array_extraction(self):
        """Text-prefixed array JSON is extracted."""
        text = 'Result: [{"name": "Alice"}]'
        assert extract_json(text) == '[{"name": "Alice"}]'

    def test_unclosed_code_fence(self):
        """Unclosed code fence content is extracted."""
        text = '```json\n{"facts": ["a"]}'
        assert extract_json(text) == '{"facts": ["a"]}'


class TestStripThinkTags:
    def test_closed_think_tags_removed(self):
        """Closed <think>...</think> blocks are removed."""
        text = '<think>reasoning here</think>{"facts": ["a"]}'
        assert _strip_think_tags(text) == '{"facts": ["a"]}'

    def test_unclosed_think_tag_removed(self):
        """Unclosed <think> removes everything from tag to end."""
        text = '<think>partial reasoning'
        assert _strip_think_tags(text) == ""

    def test_content_without_think_tags_unchanged(self):
        """Content without think tags passes through unchanged."""
        text = '{"facts": ["a"]}'
        assert _strip_think_tags(text) == '{"facts": ["a"]}'

    def test_think_tags_with_surrounding_content(self):
        """Think tags embedded in content are removed, surrounding preserved."""
        text = 'prefix <think>reasoning</think> suffix'
        assert _strip_think_tags(text) == 'prefix  suffix'

    def test_multiple_think_blocks(self):
        """Multiple think blocks are all removed."""
        text = '<think>first</think>middle<think>second</think>end'
        assert _strip_think_tags(text) == 'middleend'

    def test_empty_string(self):
        """Empty string passes through."""
        assert _strip_think_tags("") == ""


class TestGenerateResponse:
    def test_tools_passed_to_client_chat(self, ollama_llm):
        """When tools are provided, they are passed to client.chat()."""
        mock_response = MagicMock()
        mock_response.message.content = "test"
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "hello"}],
            tools=SAMPLE_TOOLS,
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["tools"] == SAMPLE_TOOLS

    def test_no_tools_not_in_params(self, ollama_llm):
        """When tools is None, 'tools' key is not passed to client.chat()."""
        mock_response = MagicMock()
        mock_response.message.content = "plain response"
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "hello"}],
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert "tools" not in params

    def test_json_format_adds_format_param(self, ollama_llm):
        """JSON response format adds format='json' to params."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": []}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "extract facts"}],
            response_format={"type": "json_object"},
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["format"] == "json"

    def test_non_json_object_format_ignored(self, ollama_llm):
        """Non-json_object response_format types are ignored (no format param)."""
        mock_response = MagicMock()
        mock_response.message.content = "plain text"
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "text"},
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert "format" not in params

    def test_json_format_does_not_mutate_caller_messages(self, ollama_llm):
        """JSON response format must not mutate the caller's messages list."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": []}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        original_content = "extract facts"
        messages = [{"role": "user", "content": original_content}]

        ollama_llm.generate_response(
            messages=messages,
            response_format={"type": "json_object"},
        )

        # Caller's list and dict must be unmodified
        assert len(messages) == 1
        assert messages[0]["content"] == original_content

    def test_options_include_config_values(self, ollama_llm):
        """Plain text requests use configured temperature from config."""
        mock_response = MagicMock()
        mock_response.message.content = "test"
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "hello"}],
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["options"]["temperature"] == 0.7
        assert params["options"]["num_predict"] == 8192
        assert params["options"]["top_p"] == 0.9


class TestParseResponse:
    def test_tool_calls_parsed_from_toolcall_objects(self, ollama_llm):
        """Modern SDK format: ToolCall objects with .function.name/.arguments."""
        tc = MagicMock()
        tc.function.name = "extract_entities"
        tc.function.arguments = {"entities": [{"name": "Alice"}]}

        response = MagicMock()
        response.message.content = ""
        response.message.tool_calls = [tc]

        result = ollama_llm._parse_response(response, tools=SAMPLE_TOOLS)

        assert result["tool_calls"] == [
            {"name": "extract_entities", "arguments": {"entities": [{"name": "Alice"}]}}
        ]

    def test_tool_calls_parsed_from_dict_format(self, ollama_llm):
        """Legacy dict format: {"function": {"name": ..., "arguments": ...}}."""
        response = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "extract_entities",
                            "arguments": {"entities": [{"name": "Bob"}]},
                        }
                    }
                ],
            }
        }

        result = ollama_llm._parse_response(response, tools=SAMPLE_TOOLS)

        assert result["tool_calls"] == [
            {"name": "extract_entities", "arguments": {"entities": [{"name": "Bob"}]}}
        ]

    def test_empty_tool_calls_when_none(self, ollama_llm):
        """When response has no tool_calls, returns empty list."""
        response = MagicMock()
        response.message.content = "I found no entities"
        response.message.tool_calls = None

        result = ollama_llm._parse_response(response, tools=SAMPLE_TOOLS)

        assert result["tool_calls"] == []
        assert result["content"] == "I found no entities"

    def test_empty_tool_calls_when_empty_list(self, ollama_llm):
        """When response has empty tool_calls list, returns empty list."""
        response = MagicMock()
        response.message.content = "No entities"
        response.message.tool_calls = []

        result = ollama_llm._parse_response(response, tools=SAMPLE_TOOLS)

        assert result["tool_calls"] == []

    def test_non_tool_response_returns_string(self, ollama_llm):
        """When tools is None/empty, returns plain string content."""
        response = MagicMock()
        response.message.content = "Hello, how can I help?"

        result = ollama_llm._parse_response(response, tools=None)

        assert result == "Hello, how can I help?"

    def test_non_tool_dict_response_returns_string(self, ollama_llm):
        """Dict response without tools returns plain string."""
        response = {"message": {"content": "Plain text response"}}

        result = ollama_llm._parse_response(response, tools=None)

        assert result == "Plain text response"

    def test_think_tags_stripped_from_tool_response(self, ollama_llm):
        """Think tags are stripped from content in tool responses."""
        response = MagicMock()
        response.message.content = "<think>reasoning</think>cleaned content"
        response.message.tool_calls = None

        result = ollama_llm._parse_response(response, tools=SAMPLE_TOOLS)

        assert result["content"] == "cleaned content"

    def test_think_tags_stripped_from_non_tool_response(self, ollama_llm):
        """Think tags are stripped from content in non-tool responses."""
        response = MagicMock()
        response.message.content = "<think>reasoning</think>clean text"

        result = ollama_llm._parse_response(response, tools=None)

        assert result == "clean text"


class TestNoThinkInjection:
    def test_no_think_injected_for_json_request(self, ollama_llm):
        """/no_think is appended to last user message for JSON requests."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": ["a"]}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        with patch.dict("os.environ", {}, clear=False):
            # Ensure MEM0_OLLAMA_THINK is not set
            import os
            os.environ.pop("MEM0_OLLAMA_THINK", None)

            ollama_llm.generate_response(
                messages=[{"role": "user", "content": "extract facts"}],
                response_format={"type": "json_object"},
            )

        params = ollama_llm.client.chat.call_args.kwargs
        last_user = [m for m in params["messages"] if m["role"] == "user"][-1]
        assert "/no_think" in last_user["content"]

    def test_no_think_injected_for_tool_request(self, ollama_llm):
        """/no_think is appended for tool-calling requests."""
        mock_response = MagicMock()
        mock_response.message.content = ""
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MEM0_OLLAMA_THINK", None)

            ollama_llm.generate_response(
                messages=[{"role": "user", "content": "extract entities"}],
                tools=SAMPLE_TOOLS,
            )

        params = ollama_llm.client.chat.call_args.kwargs
        last_user = [m for m in params["messages"] if m["role"] == "user"][-1]
        assert "/no_think" in last_user["content"]

    def test_no_think_not_duplicated(self, ollama_llm):
        """/no_think is not added if already present."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": []}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MEM0_OLLAMA_THINK", None)

            ollama_llm.generate_response(
                messages=[{"role": "user", "content": "extract facts /no_think"}],
                response_format={"type": "json_object"},
            )

        params = ollama_llm.client.chat.call_args.kwargs
        last_user = [m for m in params["messages"] if m["role"] == "user"][-1]
        # Should contain exactly one /no_think, not two
        assert last_user["content"].count("/no_think") == 1

    def test_no_think_not_injected_when_think_present(self, ollama_llm):
        """/no_think is NOT injected when /think is already present."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": []}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MEM0_OLLAMA_THINK", None)

            ollama_llm.generate_response(
                messages=[{"role": "user", "content": "extract facts /think"}],
                response_format={"type": "json_object"},
            )

        params = ollama_llm.client.chat.call_args.kwargs
        last_user = [m for m in params["messages"] if m["role"] == "user"][-1]
        assert "/no_think" not in last_user["content"]

    def test_no_think_disabled_by_env_var(self, ollama_llm):
        """/no_think is NOT injected when MEM0_OLLAMA_THINK=true."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": []}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        with patch.dict("os.environ", {"MEM0_OLLAMA_THINK": "true"}, clear=False):
            ollama_llm.generate_response(
                messages=[{"role": "user", "content": "extract facts"}],
                response_format={"type": "json_object"},
            )

        params = ollama_llm.client.chat.call_args.kwargs
        last_user = [m for m in params["messages"] if m["role"] == "user"][-1]
        assert "/no_think" not in last_user["content"]

    def test_caller_messages_not_mutated(self, ollama_llm):
        """Original caller messages are NOT mutated by /no_think injection."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": []}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        original_content = "extract facts"
        messages = [{"role": "user", "content": original_content}]

        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MEM0_OLLAMA_THINK", None)

            ollama_llm.generate_response(
                messages=messages,
                response_format={"type": "json_object"},
            )

        # Original message must be unchanged
        assert messages[0]["content"] == original_content


class TestDeterministicOptions:
    def test_temperature_zero_for_json(self, ollama_llm):
        """JSON requests override temperature to 0."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": []}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "extract"}],
            response_format={"type": "json_object"},
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["options"]["temperature"] == 0

    def test_temperature_zero_for_tools(self, ollama_llm):
        """Tool requests override temperature to 0."""
        mock_response = MagicMock()
        mock_response.message.content = ""
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "extract"}],
            tools=SAMPLE_TOOLS,
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["options"]["temperature"] == 0

    def test_repeat_penalty_for_json(self, ollama_llm):
        """JSON requests set repeat_penalty to 1.0."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": []}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "extract"}],
            response_format={"type": "json_object"},
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["options"]["repeat_penalty"] == 1.0

    def test_configured_temperature_for_plain_text(self, ollama_llm):
        """Plain text requests use configured temperature."""
        mock_response = MagicMock()
        mock_response.message.content = "hello"
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "hi"}],
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["options"]["temperature"] == 0.7
        assert "repeat_penalty" not in params["options"]


class TestKeepAlive:
    def test_default_keep_alive(self, ollama_llm):
        """Default keep_alive is '30m' when env var not set."""
        mock_response = MagicMock()
        mock_response.message.content = "test"
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MEM0_OLLAMA_KEEP_ALIVE", None)

            ollama_llm.generate_response(
                messages=[{"role": "user", "content": "hello"}],
            )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["keep_alive"] == "30m"

    def test_custom_keep_alive(self, ollama_llm):
        """Custom keep_alive from MEM0_OLLAMA_KEEP_ALIVE env var."""
        mock_response = MagicMock()
        mock_response.message.content = "test"
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        with patch.dict("os.environ", {"MEM0_OLLAMA_KEEP_ALIVE": "1h"}, clear=False):
            ollama_llm.generate_response(
                messages=[{"role": "user", "content": "hello"}],
            )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["keep_alive"] == "1h"


class TestRetryLogic:
    def test_retry_on_empty_string(self, ollama_llm):
        """Empty JSON response triggers a retry."""
        first_response = MagicMock()
        first_response.message.content = ""
        first_response.message.tool_calls = None

        second_response = MagicMock()
        second_response.message.content = '{"facts": ["a"]}'
        second_response.message.tool_calls = None

        ollama_llm.client.chat.side_effect = [first_response, second_response]

        result = ollama_llm.generate_response(
            messages=[{"role": "user", "content": "extract"}],
            response_format={"type": "json_object"},
        )

        assert result == '{"facts": ["a"]}'
        assert ollama_llm.client.chat.call_count == 2

    def test_retry_on_invalid_json(self, ollama_llm):
        """Invalid JSON response triggers a retry."""
        first_response = MagicMock()
        first_response.message.content = "not json at all"
        first_response.message.tool_calls = None

        second_response = MagicMock()
        second_response.message.content = '{"facts": ["b"]}'
        second_response.message.tool_calls = None

        ollama_llm.client.chat.side_effect = [first_response, second_response]

        result = ollama_llm.generate_response(
            messages=[{"role": "user", "content": "extract"}],
            response_format={"type": "json_object"},
        )

        assert result == '{"facts": ["b"]}'
        assert ollama_llm.client.chat.call_count == 2

    def test_retry_on_empty_object(self, ollama_llm):
        """Empty object {} triggers a retry."""
        first_response = MagicMock()
        first_response.message.content = "{}"
        first_response.message.tool_calls = None

        second_response = MagicMock()
        second_response.message.content = '{"facts": ["c"]}'
        second_response.message.tool_calls = None

        ollama_llm.client.chat.side_effect = [first_response, second_response]

        result = ollama_llm.generate_response(
            messages=[{"role": "user", "content": "extract"}],
            response_format={"type": "json_object"},
        )

        assert result == '{"facts": ["c"]}'
        assert ollama_llm.client.chat.call_count == 2

    def test_no_retry_on_valid_json(self, ollama_llm):
        """Valid JSON does not trigger a retry."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": ["valid"]}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        result = ollama_llm.generate_response(
            messages=[{"role": "user", "content": "extract"}],
            response_format={"type": "json_object"},
        )

        assert result == '{"facts": ["valid"]}'
        assert ollama_llm.client.chat.call_count == 1

    def test_retry_limit_is_one(self, ollama_llm):
        """Both attempts fail: returns second result, logs error."""
        first_response = MagicMock()
        first_response.message.content = ""
        first_response.message.tool_calls = None

        second_response = MagicMock()
        second_response.message.content = ""
        second_response.message.tool_calls = None

        ollama_llm.client.chat.side_effect = [first_response, second_response]

        result = ollama_llm.generate_response(
            messages=[{"role": "user", "content": "extract"}],
            response_format={"type": "json_object"},
        )

        # Returns the (empty) retry result — no infinite loop
        assert result == ""
        assert ollama_llm.client.chat.call_count == 2

    def test_no_retry_for_tool_calling(self, ollama_llm):
        """Tool-calling requests never trigger retry, even with empty content."""
        mock_response = MagicMock()
        mock_response.message.content = ""
        mock_response.message.tool_calls = []
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "extract"}],
            tools=SAMPLE_TOOLS,
        )

        assert ollama_llm.client.chat.call_count == 1
