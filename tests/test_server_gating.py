"""Verify the system-tools opt-in gate in server.py works both ways."""
from __future__ import annotations

import asyncio
import importlib
import sys


def _reload_server_module(monkeypatch, enabled: bool):
    """Force-reload ds_mcp_server.server with the given DS_MCP_ENABLE_SYSTEM_TOOLS value."""
    if enabled:
        monkeypatch.setenv("DS_MCP_ENABLE_SYSTEM_TOOLS", "1")
    else:
        monkeypatch.delenv("DS_MCP_ENABLE_SYSTEM_TOOLS", raising=False)
    sys.modules.pop("ds_mcp_server.server", None)
    return importlib.import_module("ds_mcp_server.server")


GATED_TOOLS = {
    "run_shell_command",
    "read_file",
    "write_file",
    "patch_file",
    "list_directory",
    "find_in_files",
    "run_background_process",
    "stop_background_process",
    "list_background_processes",
    "http_request",
}


def _list_tool_names(mcp) -> set[str]:
    tools = asyncio.run(mcp.list_tools())
    return {t.name for t in tools}


def test_system_tools_disabled_by_default(monkeypatch, capsys):
    mod = _reload_server_module(monkeypatch, enabled=False)
    names = _list_tool_names(mod.mcp)
    assert names.isdisjoint(GATED_TOOLS), (
        f"System tools should NOT be registered by default, found: {names & GATED_TOOLS}"
    )
    assert "plot_interactive_histogram" in names
    assert "run_correlation" in names
    captured = capsys.readouterr()
    assert "System/coder tools disabled" in captured.err


def test_system_tools_enabled_registers_all_gated(monkeypatch, capsys):
    mod = _reload_server_module(monkeypatch, enabled=True)
    names = _list_tool_names(mod.mcp)
    missing = GATED_TOOLS - names
    assert not missing, f"Missing gated tools when enabled: {missing}"
    captured = capsys.readouterr()
    assert "OPTIONAL SYSTEM TOOLS ARE ENABLED" in captured.err
    for name in GATED_TOOLS:
        assert name in captured.err, f"Warning should list {name}"


def test_enable_flag_accepts_truthy_variants(monkeypatch):
    for val in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("DS_MCP_ENABLE_SYSTEM_TOOLS", val)
        sys.modules.pop("ds_mcp_server.server", None)
        mod = importlib.import_module("ds_mcp_server.server")
        names = _list_tool_names(mod.mcp)
        assert "run_shell_command" in names, f"Value {val!r} should enable tools"


def test_enable_flag_rejects_falsy_variants(monkeypatch):
    for val in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("DS_MCP_ENABLE_SYSTEM_TOOLS", val)
        sys.modules.pop("ds_mcp_server.server", None)
        mod = importlib.import_module("ds_mcp_server.server")
        names = _list_tool_names(mod.mcp)
        assert "run_shell_command" not in names, f"Value {val!r} should NOT enable tools"
