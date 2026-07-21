"""Tests for the optional web UI (ds_mcp_server.web)."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from ds_mcp_server.web import chat as chat_mod


# ---------------------------------------------------------------------------
# parse_tool_output
# ---------------------------------------------------------------------------


class TestParseToolOutput:
    def test_plain_text_passes_through(self):
        visible, plot = chat_mod.parse_tool_output("hello world")
        assert visible == "hello world"
        assert plot is None

    def test_png_return_is_parsed_as_image(self):
        raw = "/tmp/runs/x/plots/foo.png|||plt.plot([1,2,3])"
        visible, plot = chat_mod.parse_tool_output(raw)
        assert plot is not None
        assert plot["kind"] == "image"
        assert plot["path"] == "/tmp/runs/x/plots/foo.png"
        assert "plt.plot" in plot["code"]
        assert "foo.png" in visible

    def test_html_return_is_parsed_as_html(self):
        raw = "/tmp/runs/x/plots/foo.html|||fig = px.scatter(df, x='a', y='b')"
        visible, plot = chat_mod.parse_tool_output(raw)
        assert plot is not None
        assert plot["kind"] == "html"

    def test_json_return_is_html_kind(self):
        raw = "/tmp/runs/x/plots/foo.json|||fig = px.scatter(df, x='a', y='b')"
        _, plot = chat_mod.parse_tool_output(raw)
        assert plot is not None
        assert plot["kind"] == "html"

    def test_multiline_code_is_captured(self):
        raw = "/tmp/runs/x/plots/foo.png|||import pandas\nfig = plt.figure()\nfig.plot()"
        _, plot = chat_mod.parse_tool_output(raw)
        assert plot is not None
        assert "fig.plot()" in plot["code"]


# ---------------------------------------------------------------------------
# FastAPI app — only test routes that don't require a real MCP subprocess.
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_no_bridge(monkeypatch):
    """Create the FastAPI app with the MCP bridge start/stop stubbed out."""
    from ds_mcp_server.web import app as app_mod

    async def _noop(self):
        self.tools = [{"name": "load_data", "description": "Load a CSV"}]

    async def _noop_stop(self):
        return None

    async def _noop_restart(self):
        # Simulate a restart: replace the tool list to reflect current settings.
        if self.settings.get("system_tools"):
            self.tools = [
                {"name": "load_data", "description": "Load a CSV"},
                {"name": "run_shell_command", "description": "Danger"},
            ]
        else:
            self.tools = [{"name": "load_data", "description": "Load a CSV"}]

    monkeypatch.setattr(app_mod._McpBridge, "start", _noop)
    monkeypatch.setattr(app_mod._McpBridge, "stop", _noop_stop)
    monkeypatch.setattr(app_mod._McpBridge, "restart", _noop_restart)
    return app_mod.create_app()


def test_health_endpoint(app_no_bridge):
    with TestClient(app_no_bridge) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_index_serves_html(app_no_bridge):
    with TestClient(app_no_bridge) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "ds-mcp-server" in r.text
        assert "text/html" in r.headers["content-type"]


def test_config_endpoint(app_no_bridge, monkeypatch):
    monkeypatch.delenv("PROVIDER", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with TestClient(app_no_bridge) as client:
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "provider" in data
        assert "model" in data
        assert isinstance(data["tools"], list)
        assert data["tools"][0]["name"] == "load_data"


def test_static_asset_served(app_no_bridge):
    with TestClient(app_no_bridge) as client:
        r = client.get("/static/styles.css")
        assert r.status_code == 200
        assert "css" in r.headers["content-type"].lower()


def test_plot_endpoint_rejects_nonexistent(app_no_bridge):
    with TestClient(app_no_bridge) as client:
        r = client.get("/api/plot", params={"path": "/tmp/does-not-exist-xyz.png"})
        assert r.status_code == 404


def test_plot_endpoint_rejects_bad_extension(app_no_bridge, tmp_path):
    bad = tmp_path / "danger.py"
    bad.write_text("print('nope')")
    with TestClient(app_no_bridge) as client:
        r = client.get("/api/plot", params={"path": str(bad)})
        assert r.status_code == 400


def test_plot_endpoint_rejects_paths_outside_plot_dirs(app_no_bridge, tmp_path):
    # Even a valid .png extension must live under a plots/runs/tmp dir.
    outside = tmp_path / "random.png"
    outside.write_bytes(b"\x89PNG\r\n")
    with TestClient(app_no_bridge) as client:
        r = client.get("/api/plot", params={"path": str(outside)})
        assert r.status_code == 403


def test_plot_endpoint_serves_file_inside_plots_dir(app_no_bridge, tmp_path):
    plots = tmp_path / "plots"
    plots.mkdir()
    f = plots / "chart.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n")
    with TestClient(app_no_bridge) as client:
        r = client.get("/api/plot", params={"path": str(f)})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/")
        assert r.content.startswith(b"\x89PNG")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_webui_cli_help_runs(capsys):
    """`ds-mcp-webui --help` should print usage without touching the network."""
    import sys
    from ds_mcp_server import cli

    old_argv = sys.argv[:]
    sys.argv = ["ds-mcp-webui", "--help"]
    try:
        with pytest.raises(SystemExit) as exc_info:
            cli.webui()
        assert exc_info.value.code == 0
    finally:
        sys.argv = old_argv
    captured = capsys.readouterr()
    assert "ds-mcp-webui" in captured.out
    assert "--host" in captured.out
    assert "--port" in captured.out


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------


class TestSettingsEndpoint:
    def test_get_settings_defaults(self, app_no_bridge, monkeypatch):
        monkeypatch.delenv("DS_MCP_ENABLE_SYSTEM_TOOLS", raising=False)
        monkeypatch.delenv("DS_MCP_ALLOW_UNRESTRICTED_EXEC", raising=False)
        # rebuild app so bridge sees clean env
        from ds_mcp_server.web import app as app_mod
        app = app_mod.create_app()
        with TestClient(app) as client:
            r = client.get("/api/settings")
            assert r.status_code == 200
            assert r.json() == {
                "settings": {"system_tools": False, "unrestricted_exec": False}
            }

    def test_get_settings_reflects_env(self, monkeypatch):
        monkeypatch.setenv("DS_MCP_ENABLE_SYSTEM_TOOLS", "1")
        monkeypatch.setenv("DS_MCP_ALLOW_UNRESTRICTED_EXEC", "true")
        from ds_mcp_server.web import app as app_mod

        async def _noop(self): self.tools = []
        async def _noop_stop(self): return None
        monkeypatch.setattr(app_mod._McpBridge, "start", _noop)
        monkeypatch.setattr(app_mod._McpBridge, "stop", _noop_stop)

        app = app_mod.create_app()
        with TestClient(app) as client:
            r = client.get("/api/settings")
            assert r.status_code == 200
            assert r.json()["settings"] == {
                "system_tools": True, "unrestricted_exec": True
            }

    def test_post_settings_no_change_returns_current(self, app_no_bridge):
        with TestClient(app_no_bridge) as client:
            r = client.post("/api/settings", json={"settings": {"system_tools": False}})
            assert r.status_code == 200
            data = r.json()
            assert data["restarted"] is False
            assert data["settings"]["system_tools"] is False

    def test_post_settings_toggles_and_restarts(self, app_no_bridge):
        with TestClient(app_no_bridge) as client:
            # Before toggle: only load_data
            r0 = client.get("/api/config")
            assert [t["name"] for t in r0.json()["tools"]] == ["load_data"]

            r = client.post(
                "/api/settings",
                json={"settings": {"system_tools": True}},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["restarted"] is True
            assert data["settings"]["system_tools"] is True
            # The stub restart adds run_shell_command
            names = [t["name"] for t in data["tools"]]
            assert "run_shell_command" in names

    def test_post_settings_ignores_unknown_keys(self, app_no_bridge):
        with TestClient(app_no_bridge) as client:
            r = client.post(
                "/api/settings",
                json={"settings": {"unknown_key": True}},
            )
            assert r.status_code == 200
            data = r.json()
            assert "unknown_key" not in data["settings"]
            assert data["restarted"] is False

    def test_post_settings_can_toggle_off(self, app_no_bridge):
        with TestClient(app_no_bridge) as client:
            client.post("/api/settings", json={"settings": {"system_tools": True}})
            r = client.post("/api/settings", json={"settings": {"system_tools": False}})
            assert r.status_code == 200
            data = r.json()
            assert data["restarted"] is True
            assert data["settings"]["system_tools"] is False

    def test_post_settings_reports_restart_failure(self, monkeypatch):
        from ds_mcp_server.web import app as app_mod

        async def _noop(self): self.tools = []
        async def _noop_stop(self): return None
        async def _fail_restart(self): raise RuntimeError("boom")

        monkeypatch.setattr(app_mod._McpBridge, "start", _noop)
        monkeypatch.setattr(app_mod._McpBridge, "stop", _noop_stop)
        monkeypatch.setattr(app_mod._McpBridge, "restart", _fail_restart)

        app = app_mod.create_app()
        with TestClient(app) as client:
            r = client.post(
                "/api/settings",
                json={"settings": {"system_tools": True}},
            )
            assert r.status_code == 500
            assert "boom" in r.json()["error"]


class TestBridgeEnvInjection:
    """The bridge must inject toggled env vars into the MCP subprocess."""

    def test_env_var_added_when_setting_on(self, monkeypatch):
        monkeypatch.delenv("DS_MCP_ENABLE_SYSTEM_TOOLS", raising=False)
        monkeypatch.delenv("DS_MCP_ALLOW_UNRESTRICTED_EXEC", raising=False)
        from ds_mcp_server.web import app as app_mod

        b = app_mod._McpBridge()
        b.settings["system_tools"] = True
        params = b._build_server_params()
        assert params.env is not None
        assert params.env.get("DS_MCP_ENABLE_SYSTEM_TOOLS") == "1"

    def test_env_var_removed_when_setting_off(self, monkeypatch):
        # Even if the outer env has it set, a toggled-off setting must scrub it.
        monkeypatch.setenv("DS_MCP_ENABLE_SYSTEM_TOOLS", "1")
        from ds_mcp_server.web import app as app_mod

        b = app_mod._McpBridge()
        b.settings["system_tools"] = False
        params = b._build_server_params()
        assert params.env is not None
        assert "DS_MCP_ENABLE_SYSTEM_TOOLS" not in params.env
