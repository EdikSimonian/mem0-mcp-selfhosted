"""Tests for config.py — build_config() with various env var combinations."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestBuildConfig:
    def _build_with_env(self, env: dict):
        """Build config with the given env vars, mocking resolve_token."""
        # Clear env vars that could leak from integration tests or CLI
        leak_keys = [k for k in os.environ if k.startswith("MEM0_")]
        if "GOOGLE_API_KEY" in os.environ:
            leak_keys.append("GOOGLE_API_KEY")
        with patch.dict("os.environ", env, clear=False) as patched_env:
            for k in leak_keys:
                if k not in env:
                    patched_env.pop(k, None)
            with patch("mem0_mcp_selfhosted.config.resolve_token", return_value="sk-test-token"):
                from mem0_mcp_selfhosted.config import build_config
                config_dict, providers_info, extra_providers = build_config()
                return config_dict, providers_info, extra_providers

    def test_defaults(self):
        """All defaults applied when no env vars set."""
        config_dict, provider_info, *_ = self._build_with_env({})

        assert config_dict["llm"]["provider"] == "anthropic"
        assert config_dict["llm"]["config"]["model"] == "claude-opus-4-6"
        assert config_dict["embedder"]["provider"] == "ollama"
        assert config_dict["embedder"]["config"]["model"] == "bge-m3"
        assert config_dict["vector_store"]["provider"] == "qdrant"
        assert config_dict["vector_store"]["config"]["collection_name"] == "mem0_mcp_selfhosted"
        assert "graph_store" not in config_dict
        assert config_dict["version"] == "v1.1"

    def test_env_overrides(self):
        """Environment variables override defaults."""
        env = {
            "MEM0_LLM_MODEL": "claude-sonnet-4-5-20250929",
            "MEM0_EMBED_MODEL": "nomic-embed-text",
            "MEM0_COLLECTION": "custom_collection",
        }
        config_dict, *_ = self._build_with_env(env)

        assert config_dict["llm"]["config"]["model"] == "claude-sonnet-4-5-20250929"
        assert config_dict["embedder"]["config"]["model"] == "nomic-embed-text"
        assert config_dict["vector_store"]["config"]["collection_name"] == "custom_collection"

    def test_graph_enabled(self):
        """Graph store included when MEM0_ENABLE_GRAPH=true."""
        env = {"MEM0_ENABLE_GRAPH": "true"}
        config_dict, *_ = self._build_with_env(env)

        assert "graph_store" in config_dict
        assert config_dict["graph_store"]["provider"] == "neo4j"
        # graph_store.llm MUST be explicit (never rely on mem0ai's openai default)
        assert "llm" in config_dict["graph_store"]
        assert config_dict["graph_store"]["llm"]["provider"] == "anthropic"

    def test_graph_disabled(self):
        """Graph store omitted when MEM0_ENABLE_GRAPH=false."""
        env = {"MEM0_ENABLE_GRAPH": "false"}
        config_dict, *_ = self._build_with_env(env)
        assert "graph_store" not in config_dict

    def test_explicit_embedder_provider(self):
        """Embedder provider is always explicit (never default to openai)."""
        config_dict, *_ = self._build_with_env({})
        assert config_dict["embedder"]["provider"] == "ollama"

    def test_provider_info_structure(self):
        """Provider info list includes Anthropic and Ollama entries."""
        _, providers_info, *_ = self._build_with_env({})

        provider_names = [pi["name"] for pi in providers_info]
        assert "ollama" in provider_names  # Always registered
        assert "anthropic" in provider_names

        anthropic_pi = next(pi for pi in providers_info if pi["name"] == "anthropic")
        assert "AnthropicOATLLM" in anthropic_pi["class_path"]

        ollama_pi = next(pi for pi in providers_info if pi["name"] == "ollama")
        assert "OllamaToolLLM" in ollama_pi["class_path"]

    def test_qdrant_optional_fields(self):
        """Optional Qdrant fields only included when env vars set."""
        config_dict, *_ = self._build_with_env({})
        assert "api_key" not in config_dict["vector_store"]["config"]

        env = {"MEM0_QDRANT_API_KEY": "test-key"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["vector_store"]["config"]["api_key"] == "test-key"

    def test_graph_llm_ollama(self):
        """Graph LLM can be set to ollama for quota savings."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "ollama",
            "MEM0_GRAPH_LLM_MODEL": "qwen3:14b",
        }
        config_dict, *_ = self._build_with_env(env)

        graph_llm = config_dict["graph_store"]["llm"]
        assert graph_llm["provider"] == "ollama"
        assert graph_llm["config"]["model"] == "qwen3:14b"

    def test_graph_llm_gemini(self):
        """Graph LLM can be set to gemini with API key."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini",
            "GOOGLE_API_KEY": "test-gemini-key",
        }
        config_dict, *_ = self._build_with_env(env)

        graph_llm = config_dict["graph_store"]["llm"]
        assert graph_llm["provider"] == "gemini"
        assert graph_llm["config"]["model"] == "gemini-2.5-flash-lite"
        assert graph_llm["config"]["api_key"] == "test-gemini-key"

    def test_graph_llm_gemini_model_override(self):
        """Gemini graph LLM model can be overridden via env var."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini",
            "MEM0_GRAPH_LLM_MODEL": "gemini-2.0-flash",
            "GOOGLE_API_KEY": "test-gemini-key",
        }
        config_dict, *_ = self._build_with_env(env)

        graph_llm = config_dict["graph_store"]["llm"]
        assert graph_llm["config"]["model"] == "gemini-2.0-flash"

    def test_graph_llm_gemini_no_api_key(self):
        """Gemini graph LLM config produced even without GOOGLE_API_KEY."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini",
        }
        config_dict, *_ = self._build_with_env(env)

        graph_llm = config_dict["graph_store"]["llm"]
        assert graph_llm["provider"] == "gemini"
        assert graph_llm["config"]["model"] == "gemini-2.5-flash-lite"
        assert "api_key" not in graph_llm["config"]

    def test_graph_llm_gemini_split_defaults(self):
        """Split-model config: config uses 'gemini' for validation, split_config returned separately."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini_split",
            "GOOGLE_API_KEY": "test-gemini-key",
        }
        config_dict, _, split_config = self._build_with_env(env)

        # Config dict uses "gemini" for pydantic validation
        graph_llm = config_dict["graph_store"]["llm"]
        assert graph_llm["provider"] == "gemini"
        assert graph_llm["config"]["model"] == "gemini-2.5-flash-lite"
        assert graph_llm["config"]["api_key"] == "test-gemini-key"

        # Split config returned separately for post-creation LLM swap
        assert split_config is not None
        assert split_config["extraction_provider"] == "gemini"
        assert split_config["extraction_model"] == "gemini-2.5-flash-lite"
        assert split_config["extraction_api_key"] == "test-gemini-key"
        assert split_config["contradiction_provider"] == "anthropic"
        assert split_config["contradiction_model"] == "claude-opus-4-6"
        assert split_config["contradiction_api_key"] == "sk-test-token"

    def test_graph_llm_gemini_split_custom_contradiction(self):
        """Split-model with custom contradiction provider/model."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini_split",
            "GOOGLE_API_KEY": "test-gemini-key",
            "MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER": "ollama",
            "MEM0_GRAPH_CONTRADICTION_LLM_MODEL": "qwen3:14b",
        }
        config_dict, _, split_config = self._build_with_env(env)

        assert split_config is not None
        assert split_config["contradiction_provider"] == "ollama"
        assert split_config["contradiction_model"] == "qwen3:14b"
        assert "contradiction_api_key" not in split_config
        assert "contradiction_ollama_base_url" in split_config

    def test_no_split_config_for_regular_providers(self):
        """split_config is None when not using gemini_split."""
        env = {"MEM0_ENABLE_GRAPH": "true"}
        _, _, split_config = self._build_with_env(env)
        assert split_config is None

        env = {"MEM0_ENABLE_GRAPH": "true", "MEM0_GRAPH_LLM_PROVIDER": "gemini"}
        _, _, split_config = self._build_with_env(env)
        assert split_config is None

    # --- Provider selection and config branching (7.x) ---

    def test_default_llm_provider_is_anthropic(self):
        """Default provider is anthropic when MEM0_LLM_PROVIDER not set."""
        config_dict, *_ = self._build_with_env({})
        assert config_dict["llm"]["provider"] == "anthropic"

    def test_ollama_llm_provider(self):
        """MEM0_LLM_PROVIDER=ollama sets the LLM provider to ollama."""
        env = {"MEM0_LLM_PROVIDER": "ollama"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["llm"]["provider"] == "ollama"

    def test_unsupported_llm_provider_raises(self):
        """Unsupported MEM0_LLM_PROVIDER raises ValueError."""
        leak_keys = [k for k in os.environ if k.startswith("MEM0_")]
        env = {"MEM0_LLM_PROVIDER": "gemini"}
        with patch.dict("os.environ", env, clear=False) as patched_env:
            for k in leak_keys:
                if k not in env:
                    patched_env.pop(k, None)
            with patch("mem0_mcp_selfhosted.config.resolve_token", return_value="sk-test"):
                from mem0_mcp_selfhosted.config import build_config
                with pytest.raises(ValueError, match="Unsupported MEM0_LLM_PROVIDER='gemini'"):
                    build_config()

    def test_anthropic_config_has_api_key_and_max_tokens(self):
        """Anthropic LLM config includes api_key and max_tokens."""
        config_dict, *_ = self._build_with_env({})
        llm_cfg = config_dict["llm"]["config"]
        assert llm_cfg["api_key"] == "sk-test-token"
        assert llm_cfg["max_tokens"] == 16384

    def test_ollama_config_has_base_url_no_api_key(self):
        """Ollama LLM config includes ollama_base_url, no api_key or max_tokens."""
        env = {"MEM0_LLM_PROVIDER": "ollama"}
        config_dict, *_ = self._build_with_env(env)
        llm_cfg = config_dict["llm"]["config"]
        assert "ollama_base_url" in llm_cfg
        assert "api_key" not in llm_cfg
        assert "max_tokens" not in llm_cfg

    def test_ollama_default_model(self):
        """Ollama provider defaults to qwen3:14b when MEM0_LLM_MODEL not set."""
        env = {"MEM0_LLM_PROVIDER": "ollama"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["llm"]["config"]["model"] == "qwen3:14b"

    def test_ollama_llm_url_custom(self):
        """MEM0_LLM_URL sets ollama_base_url when provider is ollama."""
        env = {"MEM0_LLM_PROVIDER": "ollama", "MEM0_LLM_URL": "http://gpu:11434"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["llm"]["config"]["ollama_base_url"] == "http://gpu:11434"

    def test_ollama_llm_url_default(self):
        """MEM0_LLM_URL defaults to localhost:11434 when not set."""
        env = {"MEM0_LLM_PROVIDER": "ollama"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["llm"]["config"]["ollama_base_url"] == "http://localhost:11434"

    def test_llm_url_not_read_for_anthropic(self):
        """MEM0_LLM_URL is not included in anthropic config."""
        env = {"MEM0_LLM_URL": "http://gpu:11434"}
        config_dict, *_ = self._build_with_env(env)
        assert "ollama_base_url" not in config_dict["llm"]["config"]

    # --- Conditional provider registration (8.x) ---

    def test_providers_info_includes_anthropic(self):
        """providers_info includes Anthropic when LLM provider is anthropic."""
        _, providers_info, _ = self._build_with_env({})
        provider_names = [pi["name"] for pi in providers_info]
        assert "anthropic" in provider_names
        assert "ollama" in provider_names  # Always included

    def test_providers_info_ollama_only(self):
        """providers_info includes only Ollama when LLM provider is ollama (no graph)."""
        env = {"MEM0_LLM_PROVIDER": "ollama"}
        _, providers_info, _ = self._build_with_env(env)
        provider_names = [pi["name"] for pi in providers_info]
        assert "ollama" in provider_names
        assert "anthropic" not in provider_names

    def test_providers_info_ollama_with_anthropic_graph(self):
        """providers_info includes Anthropic when graph LLM uses anthropic."""
        env = {
            "MEM0_LLM_PROVIDER": "ollama",
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "anthropic",
        }
        _, providers_info, _ = self._build_with_env(env)
        provider_names = [pi["name"] for pi in providers_info]
        assert "ollama" in provider_names
        assert "anthropic" in provider_names

    # --- URL decoupling: graph LLM (9.x) ---

    def test_graph_llm_url_from_dedicated_env(self):
        """Graph LLM ollama_base_url reads from MEM0_GRAPH_LLM_URL when set."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "ollama",
            "MEM0_GRAPH_LLM_URL": "http://gpu-box:11434",
            "MEM0_LLM_URL": "http://main-box:11434",
        }
        config_dict, *_ = self._build_with_env(env)
        graph_url = config_dict["graph_store"]["llm"]["config"]["ollama_base_url"]
        assert graph_url == "http://gpu-box:11434"

    def test_graph_llm_url_falls_back_to_llm_url(self):
        """Graph LLM ollama_base_url falls back to MEM0_LLM_URL."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "ollama",
            "MEM0_LLM_URL": "http://main-box:11434",
        }
        config_dict, *_ = self._build_with_env(env)
        graph_url = config_dict["graph_store"]["llm"]["config"]["ollama_base_url"]
        assert graph_url == "http://main-box:11434"

    def test_graph_llm_url_falls_back_to_default(self):
        """Graph LLM ollama_base_url falls back to localhost when no URLs set."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "ollama",
        }
        config_dict, *_ = self._build_with_env(env)
        graph_url = config_dict["graph_store"]["llm"]["config"]["ollama_base_url"]
        assert graph_url == "http://localhost:11434"

    def test_graph_llm_url_not_affected_by_embed_url(self):
        """Changing MEM0_EMBED_URL does NOT affect graph LLM ollama_base_url."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "ollama",
            "MEM0_EMBED_URL": "http://embed-box:11434",
        }
        config_dict, *_ = self._build_with_env(env)
        graph_url = config_dict["graph_store"]["llm"]["config"]["ollama_base_url"]
        # Should be default, NOT the embed URL
        assert graph_url == "http://localhost:11434"
        # Embedder should still use the embed URL
        assert config_dict["embedder"]["config"]["ollama_base_url"] == "http://embed-box:11434"

    # --- URL decoupling: contradiction LLM (10.x) ---

    def test_contradiction_llm_url_cascade(self):
        """Contradiction LLM uses cascade: MEM0_GRAPH_LLM_URL -> MEM0_LLM_URL -> default."""
        # With MEM0_GRAPH_LLM_URL set
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini_split",
            "GOOGLE_API_KEY": "test-key",
            "MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER": "ollama",
            "MEM0_GRAPH_LLM_URL": "http://gpu-box:11434",
            "MEM0_LLM_URL": "http://main-box:11434",
        }
        _, _, split_config = self._build_with_env(env)
        assert split_config["contradiction_ollama_base_url"] == "http://gpu-box:11434"

        # Falls back to MEM0_LLM_URL
        env2 = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini_split",
            "GOOGLE_API_KEY": "test-key",
            "MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER": "ollama",
            "MEM0_LLM_URL": "http://main-box:11434",
        }
        _, _, split_config2 = self._build_with_env(env2)
        assert split_config2["contradiction_ollama_base_url"] == "http://main-box:11434"

        # Falls back to default
        env3 = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini_split",
            "GOOGLE_API_KEY": "test-key",
            "MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER": "ollama",
        }
        _, _, split_config3 = self._build_with_env(env3)
        assert split_config3["contradiction_ollama_base_url"] == "http://localhost:11434"

    # --- Qdrant timeout (11.x) ---

    def test_qdrant_timeout_from_env(self):
        """MEM0_QDRANT_TIMEOUT sets integer timeout in vector_config."""
        env = {"MEM0_QDRANT_TIMEOUT": "30"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["vector_store"]["config"]["timeout"] == 30

    def test_qdrant_timeout_absent_when_not_set(self):
        """No timeout key in vector_config when MEM0_QDRANT_TIMEOUT is not set."""
        config_dict, *_ = self._build_with_env({})
        assert "timeout" not in config_dict["vector_store"]["config"]

    def test_contradiction_llm_url_not_affected_by_embed_url(self):
        """Changing MEM0_EMBED_URL does NOT affect contradiction LLM URL."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini_split",
            "GOOGLE_API_KEY": "test-key",
            "MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER": "ollama",
            "MEM0_EMBED_URL": "http://embed-box:11434",
        }
        _, _, split_config = self._build_with_env(env)
        # Should be default, NOT the embed URL
        assert split_config["contradiction_ollama_base_url"] == "http://localhost:11434"
