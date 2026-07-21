"""Entry points for ds-mcp-server and ds-mcp-client CLI commands."""
from __future__ import annotations

import argparse
import os


def serve() -> None:
    """Start the MCP server (stdio transport). Used by ds-mcp-server command."""
    parser = argparse.ArgumentParser(
        prog="ds-mcp-server",
        description="Start the ds-mcp-server MCP endpoint.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Transport for the MCP server.",
    )
    parser.add_argument(
        "--enable-system-tools",
        action="store_true",
        help=(
            "DANGEROUS: register the optional shell/file/HTTP/background-process tools. "
            "Grants the connected LLM RCE-equivalent access to this machine. Only use "
            "inside a sandbox (Docker, WSL, VM, or a dedicated user). Equivalent to "
            "setting DS_MCP_ENABLE_SYSTEM_TOOLS=1."
        ),
    )
    parser.add_argument(
        "--allow-unrestricted-exec",
        action="store_true",
        help=(
            "DANGEROUS: disable the sandbox around LLM-generated code in the "
            "generate_custom_plotly and generate_custom_static_plot tools. When "
            "off (the default), those tools reject imports, dunder attribute "
            "access, and calls to eval/exec/open/__import__/etc. Equivalent to "
            "setting DS_MCP_ALLOW_UNRESTRICTED_EXEC=1."
        ),
    )
    args = parser.parse_args()

    if args.enable_system_tools:
        os.environ["DS_MCP_ENABLE_SYSTEM_TOOLS"] = "1"
    if args.allow_unrestricted_exec:
        os.environ["DS_MCP_ALLOW_UNRESTRICTED_EXEC"] = "1"

    # Import AFTER the env vars are set so the server's opt-in checks see them.
    from ds_mcp_server.server import mcp

    mcp.run(transport=args.transport)


def chat() -> None:
    """Start an interactive chat client. Used by ds-mcp-client command."""
    parser = argparse.ArgumentParser(
        prog="ds-mcp-client",
        description="Interactive chat client for the ds-mcp-server.",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic", "gemini", "ollama", "openai-compat"],
        default=None,
        help=(
            "LLM provider to use. Auto-detected from PROVIDER env var if not set. "
            "openai-compat covers GPUStack, LM Studio, etc."
        ),
    )
    parser.add_argument("--model", "-m", default=None, help="Model name override.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        for candidate in [".env", os.path.expanduser("~/.env")]:
            if os.path.exists(candidate):
                load_dotenv(candidate)
                break
    except ImportError:
        pass

    provider = (
        args.provider
        or os.environ.get("PROVIDER", "").lower()
        or _detect_provider()
    )

    if provider == "anthropic":
        from ds_mcp_server.client.anthropic_client import run_chat

        run_chat(model_override=args.model)
    else:
        from ds_mcp_server.client.openai_compat import run_chat

        run_chat(provider=provider, model_override=args.model)


def _detect_provider() -> str:
    """Guess provider from env var names present."""
    api_key = os.environ.get("API_KEY", "")
    if api_key.startswith("sk-ant-"):
        return "anthropic"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def webui() -> None:
    """Launch the browser-based chat UI. Used by ds-mcp-webui command."""
    parser = argparse.ArgumentParser(
        prog="ds-mcp-webui",
        description="Start the ds-mcp-server browser chat UI.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1). Use 0.0.0.0 to expose on your LAN.",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port to listen on (default: 8765).",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable uvicorn auto-reload (for development only).",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic", "gemini", "ollama", "openai-compat"],
        default=None,
        help="LLM provider (overrides the PROVIDER env var).",
    )
    parser.add_argument("--model", "-m", default=None, help="Model name override.")
    args = parser.parse_args()

    # Best-effort .env loading, same as the CLI chat command.
    try:
        from dotenv import load_dotenv
        for candidate in [".env", os.path.expanduser("~/.env")]:
            if os.path.exists(candidate):
                load_dotenv(candidate)
                break
    except ImportError:
        pass

    if args.provider:
        os.environ["PROVIDER"] = args.provider
    if args.model:
        os.environ["MODEL"] = args.model

    try:
        from ds_mcp_server.web.app import run_server
    except ImportError as exc:
        print(
            "[Error] Web UI dependencies not installed. Install with:\n"
            "  pip install 'ds-mcp-server[web]'\n"
            f"(underlying import error: {exc})",
            file=__import__("sys").stderr,
        )
        __import__("sys").exit(1)

    run_server(host=args.host, port=args.port, reload=args.reload)
