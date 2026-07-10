"""
Sandbox for executing LLM-generated plotting code.

The custom-plot MCP tools accept a Python code string produced by the LLM and
`exec()` it in-process. Because the LLM's input can be influenced by any file,
webpage, or dataset the model reads, that code is untrusted — a hostile CSV
containing "hidden instructions" is enough to trigger destructive behaviour.

This module provides a best-effort, in-process sandbox that blocks the most
common attack surfaces while keeping legitimate plotting code (pandas / plotly /
matplotlib / seaborn on the injected `df`) working.

What the sandbox DOES prevent:

  * `import` / `from ... import` — no new modules can be pulled in.
  * Calls to `eval`, `exec`, `compile`, `open`, `__import__`, `getattr`,
    `setattr`, `delattr`, `globals`, `locals`, `vars`, `input`, `breakpoint`.
  * Attribute access to any name starting with `__` (blocks
    `x.__class__.__bases__[0].__subclasses__()` and similar dunder escapes).
  * Runaway execution — a wall-clock timeout raises after `timeout_s` seconds.

What it does NOT prevent (and cannot in a single process):

  * Filesystem access performed by ALREADY-IMPORTED libraries. If a plot
    injects `pd`, then `pd.read_csv("/etc/passwd")` still works because pandas
    legitimately needs to read files. For strong isolation, run the MCP server
    inside a container / VM / dedicated user account.
  * Native-code CPU/memory exhaustion (numpy loops). Python threads cannot
    interrupt C extensions; the timeout is best-effort.

Users who understand the risk and need unrestricted `exec` can set the
environment variable `DS_MCP_ALLOW_UNRESTRICTED_EXEC=1` (or pass
`--allow-unrestricted-exec` to the CLI). A warning is printed to stderr at
startup in that case.
"""
from __future__ import annotations

import ast
import os
import threading
from typing import Any


class SandboxViolation(Exception):
    """Raised when submitted code violates the sandbox policy."""


class SandboxTimeout(Exception):
    """Raised when submitted code exceeds the wall-clock time limit."""


# Built-in names available to sandboxed code. Deliberately excludes:
#   open, __import__, eval, exec, compile, input, breakpoint, exit, quit,
#   help, globals, locals, vars, getattr, setattr, delattr, memoryview
# (each of which is either a filesystem / import / reflection hazard).
_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset({
    # Types
    "bool", "bytes", "bytearray", "complex", "dict", "float", "frozenset",
    "int", "list", "range", "set", "slice", "str", "tuple", "type",
    # Iteration / functional
    "all", "any", "enumerate", "filter", "iter", "len", "map", "max", "min",
    "next", "reversed", "sorted", "sum", "zip",
    # Numeric
    "abs", "divmod", "hex", "oct", "pow", "round",
    # Misc
    "chr", "ord", "hash", "id", "isinstance", "issubclass", "repr", "print",
    "format", "object", "callable",
    # Exceptions the LLM may legitimately raise / catch
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "ZeroDivisionError", "ArithmeticError",
    "AssertionError", "StopIteration", "NotImplementedError",
    # Constants
    "True", "False", "None", "NotImplemented", "Ellipsis",
})


# Bare-name calls that we refuse even though they may resolve to shadowed
# locals — the intent to use them is itself suspicious in plotting code.
_FORBIDDEN_CALL_NAMES: frozenset[str] = frozenset({
    "eval", "exec", "compile", "open", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "input", "breakpoint", "exit",
    "quit", "help",
})


def build_safe_builtins() -> dict[str, Any]:
    """Return a fresh dict containing only the sandbox-approved builtins."""
    import builtins as _builtins
    return {name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(_builtins, name)}


class _SandboxValidator(ast.NodeVisitor):
    """AST walker that raises SandboxViolation on the first policy breach."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    # --- imports are blanket-denied ---
    def visit_Import(self, node: ast.Import) -> None:
        names = ", ".join(a.name for a in node.names)
        raise SandboxViolation(
            f"import statements are not allowed in sandboxed code (tried to import: {names}). "
            f"The custom-plot tools already inject pd, np, px, go, plt, sns, WordCloud, df."
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise SandboxViolation(
            f"'from ... import' statements are not allowed in sandboxed code "
            f"(tried to import from {node.module!r})."
        )

    # --- dunder attribute access is blocked ---
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise SandboxViolation(
                f"access to dunder attribute '.{node.attr}' is not allowed in sandboxed code."
            )
        self.generic_visit(node)

    # --- refuse calls to dangerous bare-name builtins ---
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALL_NAMES:
            raise SandboxViolation(
                f"call to '{func.id}(...)' is not allowed in sandboxed code."
            )
        self.generic_visit(node)


def validate(code: str) -> None:
    """Parse `code` and raise SandboxViolation on the first policy breach."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SandboxViolation(f"syntax error in sandboxed code: {e}") from e
    _SandboxValidator().visit(tree)


def _sandbox_enabled_by_env() -> bool:
    """Return True unless the user opted out via env var (default: sandbox ON)."""
    val = os.environ.get("DS_MCP_ALLOW_UNRESTRICTED_EXEC", "").strip().lower()
    return val not in ("1", "true", "yes", "on")


def run_sandboxed(
    code: str,
    scope: dict[str, Any],
    *,
    timeout_s: float = 60.0,
) -> None:
    """
    Validate `code`, then exec it inside `scope` with restricted builtins and a
    wall-clock timeout.

    If the environment variable DS_MCP_ALLOW_UNRESTRICTED_EXEC is truthy, the
    sandbox is bypassed entirely and `code` runs with full privileges. Callers
    that want to unconditionally sandbox should not use this helper — call
    `validate()` + a manual exec instead.

    `scope` is used both as globals and locals so that nested scopes (list
    comprehensions, function defs) can resolve injected names like `df`.
    Callers should include only the specific names the model needs (e.g. `pd`,
    `np`, `px`, `df`) and MUST NOT include a `__builtins__` key — this function
    installs a safe one.

    Raises:
        SandboxViolation: static policy check failed.
        SandboxTimeout: exec ran past `timeout_s`.
        Exception: whatever the exec'd code raised.
    """
    if not _sandbox_enabled_by_env():
        # Escape hatch: user has explicitly opted into unrestricted exec.
        exec(code, scope)
        return

    validate(code)

    # Install our safe builtins in a working copy so we don't leak the __builtins__
    # key back into the caller's dict. Assignments performed by the exec'd code
    # need to be visible to the caller, so we copy results back on success.
    work_scope: dict[str, Any] = dict(scope)
    work_scope["__builtins__"] = build_safe_builtins()

    # Run in a thread so we can enforce a wall-clock timeout. Threads cannot
    # be forcibly killed in Python, but joining with a timeout lets us return
    # control to the caller and surface a helpful error even if the sandboxed
    # thread continues to consume CPU.
    result: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            exec(code, work_scope)
        except BaseException as exc:  # noqa: BLE001 — we re-raise below
            result["error"] = exc

    thread = threading.Thread(target=_target, name="ds-mcp-sandbox", daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        raise SandboxTimeout(
            f"sandboxed code exceeded the {timeout_s:.0f}s wall-clock limit. "
            "The offending thread is still running in the background but has "
            "been detached; consider simplifying the plot or reducing data size."
        )

    if "error" in result:
        raise result["error"]

    # Propagate any new / mutated bindings back to the caller's scope, without
    # leaking our restricted __builtins__ dict.
    for key, value in work_scope.items():
        if key == "__builtins__":
            continue
        scope[key] = value
