"""
Helpers for onboarding pip-installed users:

- write a starter `.env` in the current directory (`--init-env`)
- print cross-platform (CMD / PowerShell / bash) setup instructions
  when required credentials are missing.
"""
from __future__ import annotations

import os
import shutil
import sys
from importlib import resources
from pathlib import Path


_TEMPLATE_RESOURCE = ("ds_mcp_server.resources", "env.template")


def _template_path() -> Path:
    """Locate the bundled env template inside the installed package."""
    pkg, name = _TEMPLATE_RESOURCE
    try:
        ref = resources.files(pkg).joinpath(name)
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise FileNotFoundError(
            f"Bundled resource {pkg}/{name} not found in the installed package."
        ) from exc
    # `resources.files()` returns a Traversable; convert to a real path.
    return Path(str(ref))


def write_starter_env(dest_dir: str | Path = ".", *, force: bool = False) -> Path:
    """
    Copy the bundled `.env` template into `dest_dir`.

    Returns the absolute path to the written file.
    Raises FileExistsError if the destination already has `.env` and force=False.
    """
    dest = Path(dest_dir).resolve() / ".env"
    if dest.exists() and not force:
        raise FileExistsError(str(dest))
    src = _template_path()
    shutil.copy(src, dest)
    return dest


def run_init_env(argv_force: bool = False) -> int:
    """Implementation of `--init-env`. Returns a process exit code."""
    try:
        path = write_starter_env(".", force=argv_force)
    except FileExistsError as exc:
        print(
            f"[ds-mcp-server] {exc} already exists. "
            f"Re-run with --init-env --force to overwrite.",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError as exc:
        print(f"[ds-mcp-server] {exc}", file=sys.stderr)
        return 2

    print(f"[ds-mcp-server] Wrote starter .env to: {path}")
    print("[ds-mcp-server] Next steps:")
    print("  1. Open that file and edit the PROVIDER / API_KEY / MODEL lines.")
    print("  2. From this same folder, run:  ds-mcp-webui   (or ds-mcp-client)")
    return 0


def credentials_look_missing() -> bool:
    """Return True if none of the common credential env vars are set."""
    return not any(
        os.environ.get(k)
        for k in ("API_KEY", "ANTHROPIC_API_KEY", "PROVIDER")
    )


def print_setup_hint(program: str = "ds-mcp-webui") -> None:
    """Print cross-platform setup instructions to stderr."""
    lines = [
        "",
        "=========================================================",
        "  ds-mcp-server: no provider configured",
        "=========================================================",
        "  I could not find any of PROVIDER / API_KEY / ANTHROPIC_API_KEY",
        "  in your environment or in a .env file.",
        "",
        "  Easiest fix — create a starter .env in the current folder:",
        "",
        f"      {program} --init-env",
        "",
        "  Then edit the file, fill in your API key, and re-run",
        f"  `{program}`.",
        "",
        "  Or set the variables directly for this session:",
        "",
        "    Windows Command Prompt:",
        "      set PROVIDER=openai",
        "      set API_KEY=sk-your-key-here",
        "      set MODEL=gpt-4o",
        f"      {program}",
        "",
        "    Windows PowerShell:",
        '      $env:PROVIDER = "openai"',
        '      $env:API_KEY  = "sk-your-key-here"',
        '      $env:MODEL    = "gpt-4o"',
        f"      {program}",
        "",
        "    macOS / Linux (bash / zsh):",
        "      export PROVIDER=openai",
        "      export API_KEY=sk-your-key-here",
        "      export MODEL=gpt-4o",
        f"      {program}",
        "",
        "=========================================================",
        "",
    ]
    for line in lines:
        print(line, file=sys.stderr)
