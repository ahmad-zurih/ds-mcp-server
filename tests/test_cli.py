"""Tests for the CLI entry points."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

from ds_mcp_server import cli


def test_enable_system_tools_flag_sets_env(monkeypatch):
    """`ds-mcp-server --enable-system-tools` must set the env var before importing server."""
    monkeypatch.delenv("DS_MCP_ENABLE_SYSTEM_TOOLS", raising=False)
    monkeypatch.setattr(sys, "argv", ["ds-mcp-server", "--enable-system-tools"])

    fake_mcp = MagicMock()
    fake_module = MagicMock(mcp=fake_mcp)
    with patch.dict(sys.modules, {"ds_mcp_server.server": fake_module}):
        cli.serve()

    assert os.environ.get("DS_MCP_ENABLE_SYSTEM_TOOLS") == "1"
    fake_mcp.run.assert_called_once_with(transport="stdio")


def test_serve_without_flag_leaves_env_untouched(monkeypatch):
    monkeypatch.delenv("DS_MCP_ENABLE_SYSTEM_TOOLS", raising=False)
    monkeypatch.setattr(sys, "argv", ["ds-mcp-server"])

    fake_mcp = MagicMock()
    fake_module = MagicMock(mcp=fake_mcp)
    with patch.dict(sys.modules, {"ds_mcp_server.server": fake_module}):
        cli.serve()

    assert "DS_MCP_ENABLE_SYSTEM_TOOLS" not in os.environ
    fake_mcp.run.assert_called_once_with(transport="stdio")


def test_detect_provider_default_is_openai(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli._detect_provider() == "openai"


def test_detect_provider_anthropic_from_key_prefix(monkeypatch):
    monkeypatch.setenv("API_KEY", "sk-ant-abc")
    assert cli._detect_provider() == "anthropic"


def test_detect_provider_anthropic_from_dedicated_env(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xyz")
    assert cli._detect_provider() == "anthropic"
