"""ds-mcp-server — Data Science MCP Server and multi-provider client."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("ds-mcp-server")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout without install)
    try:
        # setuptools-scm writes this file at build time.
        from ._version import version as __version__  # type: ignore[no-redef]
    except ImportError:
        __version__ = "0.0.0+unknown"
