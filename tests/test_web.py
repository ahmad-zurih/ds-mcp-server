"""Tests for the optional web UI (ds_mcp_server.web)."""
from __future__ import annotations

import asyncio
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
        monkeypatch.delenv("DS_MCP_MULTI_AGENT", raising=False)
        # rebuild app so bridge sees clean env
        from ds_mcp_server.web import app as app_mod
        app = app_mod.create_app()
        with TestClient(app) as client:
            r = client.get("/api/settings")
            assert r.status_code == 200
            assert r.json()["settings"] == {
                "system_tools": False,
                "unrestricted_exec": False,
                "multi_agent": False,
            }

    def test_get_settings_reflects_env(self, monkeypatch):
        monkeypatch.setenv("DS_MCP_ENABLE_SYSTEM_TOOLS", "1")
        monkeypatch.setenv("DS_MCP_ALLOW_UNRESTRICTED_EXEC", "true")
        monkeypatch.delenv("DS_MCP_MULTI_AGENT", raising=False)
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
                "system_tools": True,
                "unrestricted_exec": True,
                "multi_agent": False,
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


# ---------------------------------------------------------------------------
# Multi-agent: settings toggle (runtime-only, no MCP restart) + /api/config
# ---------------------------------------------------------------------------


class TestMultiAgentSettings:
    def test_multi_agent_defaults_off(self, app_no_bridge, monkeypatch):
        monkeypatch.delenv("DS_MCP_MULTI_AGENT", raising=False)
        with TestClient(app_no_bridge) as client:
            data = client.get("/api/settings").json()
            assert data["settings"]["multi_agent"] is False

    def test_multi_agent_env_default_on(self, monkeypatch):
        monkeypatch.setenv("DS_MCP_MULTI_AGENT", "1")
        from ds_mcp_server.web import app as app_mod

        async def _noop(self): self.tools = []
        async def _noop_stop(self): return None
        monkeypatch.setattr(app_mod._McpBridge, "start", _noop)
        monkeypatch.setattr(app_mod._McpBridge, "stop", _noop_stop)

        app = app_mod.create_app()
        with TestClient(app) as client:
            assert client.get("/api/settings").json()["settings"]["multi_agent"] is True

    def test_toggle_multi_agent_does_not_restart(self, app_no_bridge, monkeypatch):
        """Flipping multi_agent must NOT restart the MCP subprocess."""
        from ds_mcp_server.web import app as app_mod

        calls = {"restart": 0}

        async def _count_restart(self):
            calls["restart"] += 1

        monkeypatch.setattr(app_mod._McpBridge, "restart", _count_restart)

        with TestClient(app_no_bridge) as client:
            r = client.post("/api/settings", json={"settings": {"multi_agent": True}})
            assert r.status_code == 200
            data = r.json()
            assert data["restarted"] is False
            assert data["settings"]["multi_agent"] is True
            assert calls["restart"] == 0

            # And it shows up in /api/config
            cfg = client.get("/api/config").json()
            assert cfg["multi_agent"] is True

    def test_toggle_multi_agent_off_again(self, app_no_bridge):
        with TestClient(app_no_bridge) as client:
            client.post("/api/settings", json={"settings": {"multi_agent": True}})
            r = client.post("/api/settings", json={"settings": {"multi_agent": False}})
            assert r.json()["settings"]["multi_agent"] is False

    def test_config_exposes_agent_models(self, monkeypatch):
        monkeypatch.setenv("PLANNER_MODEL", "planner-x")
        monkeypatch.setenv("WORKER_MODEL", "worker-y")
        from ds_mcp_server.web import app as app_mod

        async def _noop(self): self.tools = []
        async def _noop_stop(self): return None
        monkeypatch.setattr(app_mod._McpBridge, "start", _noop)
        monkeypatch.setattr(app_mod._McpBridge, "stop", _noop_stop)

        app = app_mod.create_app()
        with TestClient(app) as client:
            cfg = client.get("/api/config").json()
            assert cfg["planner_model"] == "planner-x"
            assert cfg["worker_model"] == "worker-y"


class TestAgentConfigEditing:
    """The settings modal can edit planner/worker models + iteration knobs."""

    def test_get_settings_includes_agent_config(self, app_no_bridge):
        with TestClient(app_no_bridge) as client:
            data = client.get("/api/settings").json()
            ac = data["agent_config"]
            assert set(ac) == {
                "planner_model",
                "worker_model",
                "max_rounds",
                "max_worker_retries",
                "max_worker_steps",
            }

    def test_agent_config_seeded_from_env(self, monkeypatch):
        monkeypatch.setenv("PLANNER_MODEL", "seed-planner")
        monkeypatch.setenv("WORKER_MODEL", "seed-worker")
        monkeypatch.setenv("MAX_ROUNDS", "5")
        from ds_mcp_server.web import app as app_mod

        async def _noop(self): self.tools = []
        async def _noop_stop(self): return None
        monkeypatch.setattr(app_mod._McpBridge, "start", _noop)
        monkeypatch.setattr(app_mod._McpBridge, "stop", _noop_stop)

        app = app_mod.create_app()
        with TestClient(app) as client:
            ac = client.get("/api/settings").json()["agent_config"]
            assert ac["planner_model"] == "seed-planner"
            assert ac["worker_model"] == "seed-worker"
            assert ac["max_rounds"] == 5

    def test_post_updates_models_without_restart(self, app_no_bridge, monkeypatch):
        from ds_mcp_server.web import app as app_mod

        calls = {"restart": 0}

        async def _count_restart(self):
            calls["restart"] += 1

        monkeypatch.setattr(app_mod._McpBridge, "restart", _count_restart)

        with TestClient(app_no_bridge) as client:
            r = client.post(
                "/api/settings",
                json={
                    "agent_config": {
                        "planner_model": "gpt-4o",
                        "worker_model": "gpt-4o-mini",
                        "max_rounds": 4,
                        "max_worker_retries": 1,
                        "max_worker_steps": 9,
                    }
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert data["restarted"] is False
            assert calls["restart"] == 0
            ac = data["agent_config"]
            assert ac["planner_model"] == "gpt-4o"
            assert ac["worker_model"] == "gpt-4o-mini"
            assert ac["max_rounds"] == 4
            assert ac["max_worker_retries"] == 1
            assert ac["max_worker_steps"] == 9
            # And it's reflected in /api/config used to build the team.
            cfg = client.get("/api/config").json()
            assert cfg["planner_model"] == "gpt-4o"
            assert cfg["worker_model"] == "gpt-4o-mini"

    def test_numeric_values_are_clamped(self, app_no_bridge):
        with TestClient(app_no_bridge) as client:
            r = client.post(
                "/api/settings",
                json={
                    "agent_config": {
                        "max_rounds": 999,       # -> clamped to 20
                        "max_worker_retries": -5,  # -> clamped to 0
                        "max_worker_steps": 0,     # -> clamped to 1
                    }
                },
            )
            ac = r.json()["agent_config"]
            assert ac["max_rounds"] == 20
            assert ac["max_worker_retries"] == 0
            assert ac["max_worker_steps"] == 1

    def test_blank_model_is_ignored(self, app_no_bridge):
        with TestClient(app_no_bridge) as client:
            before = client.get("/api/settings").json()["agent_config"]["planner_model"]
            r = client.post(
                "/api/settings",
                json={"agent_config": {"planner_model": "   "}},
            )
            after = r.json()["agent_config"]["planner_model"]
            assert after == before

    def test_non_numeric_value_is_ignored(self, app_no_bridge):
        with TestClient(app_no_bridge) as client:
            before = client.get("/api/settings").json()["agent_config"]["max_rounds"]
            r = client.post(
                "/api/settings",
                json={"agent_config": {"max_rounds": "not-a-number"}},
            )
            after = r.json()["agent_config"]["max_rounds"]
            assert after == before

    def test_edited_config_feeds_agent_config_builder(self, app_no_bridge):
        """_agent_config(bridge) must reflect live edits, not just env."""
        from ds_mcp_server.web import app as app_mod

        with TestClient(app_no_bridge) as client:
            client.post(
                "/api/settings",
                json={
                    "agent_config": {
                        "planner_model": "big-model",
                        "worker_model": "small-model",
                        "max_worker_steps": 7,
                    }
                },
            )
            bridge = client.app.state.bridge
            cfg = app_mod._agent_config(bridge)
            assert cfg.planner_model == "big-model"
            assert cfg.worker_model == "small-model"
            assert cfg.max_worker_steps == 7


# ---------------------------------------------------------------------------
# Multi-agent: event translation + streaming generator
# ---------------------------------------------------------------------------


class TestAgentEventToUi:
    def test_plan_event(self):
        ui = chat_mod._agent_event_to_ui(
            {
                "type": "plan",
                "round": 2,
                "reasoning": "why",
                "status": "continue",
                "tasks": [{"category": "data", "task": "load"}],
            }
        )
        assert len(ui) == 1
        assert ui[0]["type"] == "plan"
        assert ui[0]["round"] == 2
        assert ui[0]["tasks"][0]["category"] == "data"

    def test_worker_start_event(self):
        ui = chat_mod._agent_event_to_ui(
            {"type": "worker_start", "category": "stats", "task": "corr"}
        )
        assert ui == [
            {"type": "worker_start", "round": None, "category": "stats", "task": "corr"}
        ]

    def test_worker_result_with_artifact_emits_plot(self):
        ui = chat_mod._agent_event_to_ui(
            {
                "type": "worker_result",
                "category": "plot_static",
                "success": True,
                "attempts": 1,
                "tool_calls": ["make_plot"],
                "artifacts": [{"path": "/tmp/runs/x/plots/a.png", "kind": "image"}],
                "error": "",
            }
        )
        types = [u["type"] for u in ui]
        assert "worker_result" in types
        plot_ev = next(u for u in ui if u["type"] == "tool_result")
        assert plot_ev["plot"]["path"] == "/tmp/runs/x/plots/a.png"
        assert plot_ev["plot"]["kind"] == "image"

    def test_final_event_is_dropped(self):
        assert chat_mod._agent_event_to_ui({"type": "final", "text": "done"}) == []


class _FakeSupervisor:
    """Minimal stand-in for Supervisor: emits events then returns an answer."""

    def __init__(self):
        self.on_event = lambda e: None

    async def run(self, message):
        self.on_event(
            {
                "type": "plan",
                "round": 1,
                "reasoning": "r",
                "status": "continue",
                "tasks": [{"category": "data", "task": "load"}],
            }
        )
        self.on_event(
            {"type": "worker_start", "round": 1, "category": "data", "task": "load"}
        )
        await asyncio.sleep(0)  # hand control back to the streaming loop
        self.on_event(
            {
                "type": "worker_result",
                "round": 1,
                "category": "data",
                "success": True,
                "attempts": 1,
                "tool_calls": ["load_data"],
                "artifacts": [{"path": "/tmp/runs/x/plots/a.png", "kind": "image"}],
                "error": "",
            }
        )
        return "the final answer"


def test_run_multi_agent_turn_streams_events(monkeypatch):
    import ds_mcp_server.agents.runner as runner_mod

    async def fake_list_tools(session):
        return []

    monkeypatch.setattr(chat_mod, "list_tools_async", fake_list_tools)
    monkeypatch.setattr(
        runner_mod, "build_team", lambda *a, **k: _FakeSupervisor()
    )

    async def drive():
        events = []
        async for ev in chat_mod.run_multi_agent_turn(object(), object(), "hi"):
            events.append(ev)
        return events

    events = asyncio.run(drive())
    types = [e["type"] for e in events]
    assert "plan" in types
    assert "worker_start" in types
    assert "worker_result" in types
    assert any(e["type"] == "tool_result" and e.get("plot") for e in events)
    # The last event is the synthesised final answer.
    assert events[-1] == {"type": "text", "text": "the final answer"}


def test_run_multi_agent_turn_propagates_errors(monkeypatch):
    import ds_mcp_server.agents.runner as runner_mod

    async def fake_list_tools(session):
        return []

    class _BoomSupervisor:
        def __init__(self):
            self.on_event = lambda e: None

        async def run(self, message):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(chat_mod, "list_tools_async", fake_list_tools)
    monkeypatch.setattr(runner_mod, "build_team", lambda *a, **k: _BoomSupervisor())

    async def drive():
        async for _ in chat_mod.run_multi_agent_turn(object(), object(), "hi"):
            pass

    with pytest.raises(RuntimeError, match="kaboom"):
        asyncio.run(drive())
