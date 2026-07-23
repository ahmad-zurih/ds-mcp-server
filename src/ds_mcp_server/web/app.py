"""
FastAPI application for the ds-mcp-server web UI.

Routes:
  GET  /                    -> chat page (static HTML)
  GET  /health              -> {"status": "ok"}
  GET  /api/config          -> {provider, model, tools: [...]}
  GET  /api/plot?path=...   -> serve a generated plot (whitelisted to
                               the session's runs/ directory)
  WS   /ws                  -> streaming chat protocol (see chat.py)
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from ds_mcp_server.client._base import list_tools_async
from ds_mcp_server.web.chat import (
    run_anthropic_turn,
    run_multi_agent_turn,
    run_openai_turn,
)

# Env vars the UI can toggle at runtime. Each toggle restarts the MCP
# subprocess so its module-level opt-in check sees the new value.
_TOGGLEABLE_ENV = {
    "system_tools": "DS_MCP_ENABLE_SYSTEM_TOOLS",
    "unrestricted_exec": "DS_MCP_ALLOW_UNRESTRICTED_EXEC",
}

# Runtime settings that do NOT require restarting the MCP subprocess (they
# change how the web layer orchestrates the LLMs, not which tools exist).
_RUNTIME_ONLY = ("multi_agent",)

# Editable multi-agent numeric parameters and their safe bounds.
_AGENT_INT_BOUNDS = {
    "max_rounds": (1, 20),
    "max_worker_retries": (0, 10),
    "max_worker_steps": (1, 30),
}
_AGENT_STR_FIELDS = ("planner_model", "worker_model")


def _apply_agent_config(bridge: "_McpBridge", incoming: dict[str, Any]) -> None:
    """Validate and apply editable multi-agent parameters (runtime-only)."""
    if not isinstance(incoming, dict):
        return
    for field in _AGENT_STR_FIELDS:
        if field in incoming:
            val = str(incoming[field]).strip()
            if val:
                bridge.agent_config[field] = val
    for field, (lo, hi) in _AGENT_INT_BOUNDS.items():
        if field in incoming:
            try:
                val = int(incoming[field])
            except (TypeError, ValueError):
                continue
            bridge.agent_config[field] = max(lo, min(hi, val))

_STATIC_DIR = Path(__file__).parent / "static"
_ALLOWED_PLOT_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".html", ".json"}


def _resolve_provider() -> str:
    """Same detection logic as the CLI chat command."""
    explicit = os.environ.get("PROVIDER", "").lower().strip()
    if explicit:
        return explicit
    api_key = os.environ.get("API_KEY", "")
    if api_key.startswith("sk-ant-") or os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def _resolve_model(provider: str) -> str:
    override = os.environ.get("MODEL")
    if override:
        return override
    defaults = {
        "openai": "gpt-4o",
        "gemini": "gemini-2.0-flash",
        "ollama": "llama3",
        "anthropic": "claude-opus-4-5",
        "openai-compat": os.environ.get("MODEL", "gpt-4o"),
    }
    return defaults.get(provider, "gpt-4o")


def _agent_config(bridge: "_McpBridge | None" = None):
    """Build the multi-agent AgentConfig from the bridge's live parameters.

    Falls back to env-derived defaults when no bridge is supplied.
    """
    from ds_mcp_server.agents.runner import AgentConfig, config_from_env

    provider = _resolve_provider()
    if bridge is None:
        return config_from_env(provider, None, None, None, None, None)
    ac = bridge.agent_config
    return AgentConfig(
        provider=provider,
        planner_model=ac["planner_model"],
        worker_model=ac["worker_model"],
        max_rounds=int(ac["max_rounds"]),
        max_worker_retries=int(ac["max_worker_retries"]),
        max_worker_steps=int(ac["max_worker_steps"]),
    )


class _McpBridge:
    """
    Long-lived stdio MCP session shared by all websockets on this process.

    A single MCP subprocess is spawned once at startup and torn down at
    shutdown. WebSocket handlers serialize access with an asyncio.Lock so
    the underlying stdio pipes see one request at a time.

    ``settings`` holds toggles the UI can flip at runtime; changing any of
    them triggers a restart of the subprocess so its module-level env
    checks re-evaluate.
    """

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self._stack: Any = None
        self._lock = asyncio.Lock()
        self.tools: list[dict[str, Any]] = []
        # Runtime-toggleable settings; keys match _TOGGLEABLE_ENV.
        self.settings: dict[str, bool] = {
            "system_tools": os.environ.get("DS_MCP_ENABLE_SYSTEM_TOOLS", "").strip().lower()
            in ("1", "true", "yes", "on"),
            "unrestricted_exec": os.environ.get("DS_MCP_ALLOW_UNRESTRICTED_EXEC", "").strip().lower()
            in ("1", "true", "yes", "on"),
            "multi_agent": os.environ.get("DS_MCP_MULTI_AGENT", "").strip().lower()
            in ("1", "true", "yes", "on"),
        }
        # Editable multi-agent parameters (runtime-only, no MCP restart).
        # Seeded from env / CLI defaults; the UI can override them live.
        from ds_mcp_server.agents.runner import config_from_env

        _cfg = config_from_env(_resolve_provider(), None, None, None, None, None)
        self.agent_config: dict[str, Any] = {
            "planner_model": _cfg.planner_model,
            "worker_model": _cfg.worker_model,
            "max_rounds": _cfg.max_rounds,
            "max_worker_retries": _cfg.max_worker_retries,
            "max_worker_steps": _cfg.max_worker_steps,
        }

    def _build_server_params(self):
        """Build stdio server params, injecting the toggled env vars.

        We can't rely on ``client._base.get_server_params()`` because that
        one passes ``env=None`` (inheriting the parent env). Since we want
        to flip these live without restarting the web process, we build an
        explicit env dict for each MCP subprocess launch.
        """
        import shutil
        import sys as _sys
        from mcp.client.stdio import StdioServerParameters

        env = dict(os.environ)
        for key, env_name in _TOGGLEABLE_ENV.items():
            if self.settings.get(key):
                env[env_name] = "1"
            else:
                env.pop(env_name, None)

        exe = shutil.which("ds-mcp-server")
        if exe:
            return StdioServerParameters(command=exe, args=[], env=env)
        return StdioServerParameters(
            command=_sys.executable,
            args=["-c", "from ds_mcp_server.cli import serve; serve()"],
            env=env,
        )

    async def start(self) -> None:
        from contextlib import AsyncExitStack

        stack = AsyncExitStack()
        params = self._build_server_params()
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.tools = await list_tools_async(session)
        self.session = session
        self._stack = stack

    async def stop(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                pass
            self._stack = None
            self.session = None

    async def restart(self) -> None:
        """Tear down and re-spawn the MCP subprocess with current settings."""
        await self.stop()
        await self.start()

    def lock(self) -> asyncio.Lock:
        return self._lock


def create_app() -> FastAPI:
    bridge = _McpBridge()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await bridge.start()
        try:
            yield
        finally:
            await bridge.stop()

    app = FastAPI(title="ds-mcp-server web UI", lifespan=lifespan)
    app.state.bridge = bridge

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        provider = _resolve_provider()
        cfg = _agent_config(bridge)
        return {
            "provider": provider,
            "model": _resolve_model(provider),
            "multi_agent": bool(bridge.settings.get("multi_agent")),
            "planner_model": cfg.planner_model,
            "worker_model": cfg.worker_model,
            "agent_config": dict(bridge.agent_config),
            "tools": [{"name": t["name"], "description": t["description"]} for t in bridge.tools],
        }

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        return {
            "settings": dict(bridge.settings),
            "agent_config": dict(bridge.agent_config),
        }

    @app.post("/api/settings")
    async def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
        """Toggle runtime settings and restart the MCP subprocess to apply.

        Body: {"settings": {"system_tools": true, "unrestricted_exec": false}}
        Only known keys (defined in _TOGGLEABLE_ENV) are honoured.
        """
        incoming = payload.get("settings") or {}

        # Runtime-only toggles: apply immediately, no MCP restart needed.
        for key in _RUNTIME_ONLY:
            if key in incoming:
                bridge.settings[key] = bool(incoming[key])

        # Editable multi-agent parameters: also runtime-only.
        _apply_agent_config(bridge, payload.get("agent_config") or {})

        changed = False
        for key in _TOGGLEABLE_ENV:
            if key in incoming:
                new_val = bool(incoming[key])
                if bridge.settings.get(key) != new_val:
                    bridge.settings[key] = new_val
                    changed = True

        if not changed:
            return {
                "settings": dict(bridge.settings),
                "agent_config": dict(bridge.agent_config),
                "restarted": False,
                "tools": [{"name": t["name"], "description": t["description"]} for t in bridge.tools],
            }

        async with bridge.lock():
            try:
                await bridge.restart()
            except Exception as exc:  # noqa: BLE001
                return JSONResponse(
                    {"error": f"MCP server failed to restart: {exc}",
                     "settings": dict(bridge.settings),
                     "agent_config": dict(bridge.agent_config)},
                    status_code=500,
                )

        return {
            "settings": dict(bridge.settings),
            "agent_config": dict(bridge.agent_config),
            "restarted": True,
            "tools": [{"name": t["name"], "description": t["description"]} for t in bridge.tools],
        }

    @app.get("/api/plot")
    async def get_plot(path: str) -> Response:
        """Serve a generated plot from disk (with strict validation)."""
        p = Path(path).resolve()
        if not p.exists() or not p.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        if p.suffix.lower() not in _ALLOWED_PLOT_EXTS:
            return JSONResponse({"error": "extension not allowed"}, status_code=400)
        # Only allow reading files under a "plots" or "runs" directory to avoid
        # turning /api/plot into an arbitrary file reader. get_plot_path() in
        # viz_utils always writes into a `plots/` subdir, so this covers every
        # legitimate case without opening up /tmp.
        parts_lower = [x.lower() for x in p.parts]
        if not any(seg in parts_lower for seg in ("plots", "runs")):
            return JSONResponse({"error": "path not permitted"}, status_code=403)
        media, _ = mimetypes.guess_type(str(p))
        return FileResponse(str(p), media_type=media or "application/octet-stream")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        provider = _resolve_provider()
        model = _resolve_model(provider)

        openai_conv: list[dict[str, Any]] = []
        anth_messages: list[dict[str, Any]] = []
        system_prompt = (
            "You are a helpful data science assistant with access to powerful "
            "visualization and analysis tools. When you generate a plot, tell "
            "the user what you plotted in one or two sentences."
        )

        llm = None
        anth_client = None
        if provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                await websocket.send_json(
                    {"type": "error", "message": "anthropic SDK not installed. pip install ds-mcp-server[anthropic]"}
                )
                await websocket.close()
                return
            api_key = (
                os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("API_KEY", "")
            )
            if not api_key:
                await websocket.send_json({"type": "error", "message": "ANTHROPIC_API_KEY / API_KEY not set"})
                await websocket.close()
                return
            anth_client = Anthropic(api_key=api_key)
        else:
            try:
                from openai import OpenAI
            except ImportError:
                await websocket.send_json({"type": "error", "message": "openai SDK not installed"})
                await websocket.close()
                return
            api_key = os.environ.get("API_KEY", "")
            base_url = os.environ.get("API_BASE_URL")
            if provider == "ollama" and not api_key:
                api_key = "ollama"
            if not api_key:
                await websocket.send_json({"type": "error", "message": "API_KEY not set"})
                await websocket.close()
                return
            kwargs: dict[str, str] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            elif provider == "gemini":
                kwargs["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
            elif provider == "ollama":
                kwargs["base_url"] = "http://localhost:11434/v1"
            llm = OpenAI(**kwargs)

        await websocket.send_json({"type": "ready", "provider": provider, "model": model})

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    payload = json.loads(raw)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "invalid JSON"})
                    continue
                user_msg = (payload.get("message") or "").strip()
                if not user_msg:
                    continue

                async with bridge.lock():
                    if bridge.session is None:
                        await websocket.send_json({"type": "error", "message": "MCP session not ready"})
                        continue
                    try:
                        if bridge.settings.get("multi_agent"):
                            gen = run_multi_agent_turn(
                                bridge.session, _agent_config(bridge), user_msg
                            )
                        elif provider == "anthropic":
                            anth_messages.append({"role": "user", "content": user_msg})
                            gen = run_anthropic_turn(
                                bridge.session, anth_client, model, system_prompt, anth_messages
                            )
                        else:
                            openai_conv.append({"role": "user", "content": user_msg})
                            gen = run_openai_turn(bridge.session, llm, model, openai_conv)
                        async for event in gen:
                            await websocket.send_json(event)
                        await websocket.send_json({"type": "done"})
                    except Exception as exc:  # noqa: BLE001
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        await websocket.send_json({"type": "done"})
        except WebSocketDisconnect:
            return

    return app


def run_server(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    """Entry point for the ds-mcp-webui CLI command."""
    try:
        import uvicorn
    except ImportError:
        print(
            "[Error] uvicorn / fastapi not installed. Install the web extra:\n"
            "  pip install 'ds-mcp-server[web]'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[ds-mcp-webui] Starting on http://{host}:{port}", file=sys.stderr)
    print(f"[ds-mcp-webui] Open that URL in your browser to chat.", file=sys.stderr)
    uvicorn.run(
        "ds_mcp_server.web.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
