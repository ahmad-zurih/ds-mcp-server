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
    args = parser.parse_args()

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
