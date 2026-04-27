"""Tests for graph_tools.py — lazy driver, error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import mem0_mcp_selfhosted.graph_tools as gt


class TestLazyDriverInit:
    def setup_method(self):
        """Reset the global driver before each test."""
        gt._driver = None

    def test_driver_created_on_first_use(self):
        """Driver is lazily created on first _get_driver() call."""
        mock_gdb = MagicMock()
        mock_driver = MagicMock()
        mock_gdb.GraphDatabase.driver.return_value = mock_driver

        # neo4j is imported lazily inside _get_driver, so patch sys.modules
        with patch.dict("sys.modules", {"neo4j": mock_gdb}):
            gt._driver = None
            driver = gt._get_driver()
            assert driver is mock_driver
            mock_gdb.GraphDatabase.driver.assert_called_once()

    def test_driver_reused_on_subsequent_calls(self):
        """Once created, driver is reused."""
        mock_driver = MagicMock()
        gt._driver = mock_driver
        assert gt._get_driver() is mock_driver


class TestFindEntity:
    def test_neo4j_unavailable_returns_error(self):
        """Returns structured error when Neo4j is unreachable."""
        gt._driver = None
        with patch("mem0_mcp_selfhosted.graph_tools._get_driver", return_value=None):
            result = json.loads(gt.find_entity("Alice"))
            assert result["error"] == "graph_unavailable"

    def test_successful_search(self):
        """Successful Cypher query returns formatted entities."""
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.data.return_value = {
            "entity": "Alice",
            "labels": ["Person"],
            "relationship": "PREFERS",
            "related_entity": "TypeScript",
        }
        mock_session.run.return_value = [mock_record]
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        gt._driver = mock_driver
        with patch.dict("os.environ", {}, clear=False):
            result = json.loads(gt.find_entity("Alice"))

        assert "entities" in result
        assert len(result["entities"]) == 1
        assert "Alice --[PREFERS]--> TypeScript" in result["entities"][0]["relationship"]


    def _make_mock_driver(self, records):
        """Create a mock Neo4j driver returning the given records."""
        mock_session = MagicMock()
        mock_records = []
        for rec in records:
            mock_record = MagicMock()
            mock_record.data.return_value = rec
            mock_records.append(mock_record)
        mock_session.run.return_value = mock_records
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        return mock_driver, mock_session

    def test_wildcard_asterisk_lists_all(self):
        """Asterisk query skips WHERE clause and lists all entities."""
        mock_driver, mock_session = self._make_mock_driver([
            {"entity": "Alice", "labels": ["Person"], "relationship": None, "related_entity": None},
            {"entity": "Bob", "labels": ["Person"], "relationship": None, "related_entity": None},
        ])
        gt._driver = mock_driver
        with patch.dict("os.environ", {}, clear=False):
            result = json.loads(gt.find_entity("*"))

        assert "entities" in result
        assert len(result["entities"]) == 2
        # Verify the Cypher query does NOT contain WHERE/CONTAINS
        cypher_arg = mock_session.run.call_args[0][0]
        assert "CONTAINS" not in cypher_arg
        assert "WHERE" not in cypher_arg

    def test_empty_string_lists_all(self):
        """Empty query skips WHERE clause and lists all entities."""
        mock_driver, mock_session = self._make_mock_driver([
            {"entity": "Alice", "labels": ["Person"], "relationship": None, "related_entity": None},
        ])
        gt._driver = mock_driver
        with patch.dict("os.environ", {}, clear=False):
            result = json.loads(gt.find_entity(""))

        assert "entities" in result
        cypher_arg = mock_session.run.call_args[0][0]
        assert "CONTAINS" not in cypher_arg

    def test_whitespace_only_lists_all(self):
        """Whitespace-only query treated as list-all."""
        mock_driver, mock_session = self._make_mock_driver([])
        gt._driver = mock_driver
        with patch.dict("os.environ", {}, clear=False):
            result = json.loads(gt.find_entity("   "))

        assert "entities" in result
        cypher_arg = mock_session.run.call_args[0][0]
        assert "CONTAINS" not in cypher_arg

    def test_regular_query_uses_contains(self):
        """Non-wildcard queries still use CONTAINS substring matching."""
        mock_driver, mock_session = self._make_mock_driver([])
        gt._driver = mock_driver
        with patch.dict("os.environ", {}, clear=False):
            gt.find_entity("Alice")

        cypher_arg = mock_session.run.call_args[0][0]
        assert "CONTAINS" in cypher_arg
        # Verify the search term parameter was passed
        params_arg = mock_session.run.call_args[0][1]
        assert params_arg == {"search_term": "Alice"}


class TestGetEntity:
    def test_entity_not_found_returns_empty(self):
        """Returns empty result set (not error) when entity not found."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        gt._driver = mock_driver
        with patch.dict("os.environ", {}, clear=False):
            result = json.loads(gt.get_entity("NonExistent"))

        assert result["entity"] == "NonExistent"
        assert result["relationships"] == []

    def test_neo4j_unavailable_returns_error(self):
        """Returns structured error when Neo4j is unreachable."""
        gt._driver = None
        with patch("mem0_mcp_selfhosted.graph_tools._get_driver", return_value=None):
            result = json.loads(gt.get_entity("Alice"))
            assert result["error"] == "graph_unavailable"
