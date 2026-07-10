"""Tests for the custom-plot sandbox."""
from __future__ import annotations

import importlib
import sys
import time

import pytest

from ds_mcp_server._tools import sandbox
from ds_mcp_server._tools.sandbox import (
    SandboxTimeout,
    SandboxViolation,
    run_sandboxed,
    validate,
)


class TestValidateBlocksBadCode:
    def test_rejects_import_statement(self):
        with pytest.raises(SandboxViolation, match="import"):
            validate("import os")

    def test_rejects_from_import(self):
        with pytest.raises(SandboxViolation, match="from"):
            validate("from os import system")

    def test_rejects_dunder_attribute_access(self):
        with pytest.raises(SandboxViolation, match="dunder"):
            validate("x = [].__class__")

    def test_rejects_class_bases_escape(self):
        with pytest.raises(SandboxViolation, match="dunder"):
            validate("''.__class__.__mro__[-1].__subclasses__()")

    @pytest.mark.parametrize(
        "call",
        [
            "eval('1+1')",
            "exec('x=1')",
            "compile('x','<s>','exec')",
            "open('/etc/passwd')",
            "__import__('os')",
            "getattr(pd, 'read_csv')",
            "setattr(x, 'y', 1)",
            "globals()",
            "locals()",
            "vars()",
        ],
    )
    def test_rejects_dangerous_builtins(self, call: str):
        with pytest.raises(SandboxViolation):
            validate(call)

    def test_rejects_syntax_error(self):
        with pytest.raises(SandboxViolation, match="syntax"):
            validate("def broken(:")


class TestValidateAllowsGoodCode:
    def test_allows_pandas_operations(self):
        # No exception expected.
        validate("result = df.groupby('x')['y'].mean().reset_index()")

    def test_allows_plotly_chart_construction(self):
        validate(
            "fig = px.scatter(df, x='x', y='y', color='category')\n"
            "fig.update_layout(title='hi')"
        )

    def test_allows_matplotlib(self):
        validate(
            "fig, ax = plt.subplots()\n"
            "ax.plot(df['x'], df['y'])\n"
            "ax.set_title('demo')"
        )

    def test_allows_list_comprehensions_and_len(self):
        validate("nums = [len(str(v)) for v in df['x']]")

    def test_allows_normal_attribute_access(self):
        validate("shape = df.shape\nfig = px.line(df, x='x', y='y')")


class TestRunSandboxed:
    def test_executes_simple_arithmetic(self):
        scope: dict = {}
        run_sandboxed("result = 2 + 3", scope)
        assert scope["result"] == 5

    def test_injects_variables_into_scope(self):
        scope = {"a": 10, "b": 20}
        run_sandboxed("total = a + b", scope)
        assert scope["total"] == 30

    def test_open_is_not_available_even_at_runtime(self, tmp_path):
        """Prove the restricted builtins actually apply, not just AST validation.

        `open` as a plain Name reference (not a Call) passes the AST validator,
        but must fail at runtime because it isn't in the sandbox's builtins.
        """
        with pytest.raises(NameError):
            run_sandboxed("f = open", {})

    def test_import_via_dunder_is_blocked(self):
        """__import__('os') must be caught by AST — dunder attribute or bare-name call."""
        with pytest.raises(SandboxViolation):
            run_sandboxed("__import__('os')", {})

    def test_import_is_blocked_at_ast_time(self):
        with pytest.raises(SandboxViolation):
            run_sandboxed("import os\nos.system('echo pwned')", {})

    def test_propagates_runtime_error(self):
        with pytest.raises(ZeroDivisionError):
            run_sandboxed("x = 1 / 0", {})

    def test_timeout_fires_on_infinite_loop(self):
        start = time.time()
        with pytest.raises(SandboxTimeout):
            run_sandboxed("while True: pass", {}, timeout_s=0.5)
        elapsed = time.time() - start
        # Give some slack for CI scheduling variance.
        assert elapsed < 3.0, f"Timeout took too long to fire: {elapsed:.2f}s"


class TestEscapeHatch:
    def test_env_var_disables_sandbox(self, monkeypatch):
        monkeypatch.setenv("DS_MCP_ALLOW_UNRESTRICTED_EXEC", "1")
        # This would normally be rejected by the AST validator, but with the
        # escape hatch it should just run.
        scope: dict = {}
        run_sandboxed("import math\nresult = math.sqrt(16)", scope)
        assert scope["result"] == 4.0

    def test_falsy_env_var_keeps_sandbox_on(self, monkeypatch):
        monkeypatch.setenv("DS_MCP_ALLOW_UNRESTRICTED_EXEC", "0")
        with pytest.raises(SandboxViolation):
            run_sandboxed("import os", {})

    def test_unset_env_var_keeps_sandbox_on(self, monkeypatch):
        monkeypatch.delenv("DS_MCP_ALLOW_UNRESTRICTED_EXEC", raising=False)
        with pytest.raises(SandboxViolation):
            run_sandboxed("import os", {})


class TestSandboxOnRealPlotTools:
    """End-to-end: the custom-plot tools must reject bad code and accept good code."""

    def test_interactive_custom_plot_rejects_import(self, csv_path):
        from ds_mcp_server._tools.plot_interactive import generate_custom_plotly_impl

        out = generate_custom_plotly_impl(
            csv_path,
            "import os\nfig = px.scatter(df, x='x', y='y')",
            "bad",
        )
        assert "sandbox rejected" in out.lower() or "Error" in out
        assert "import" in out.lower()

    def test_interactive_custom_plot_accepts_valid_plotly(self, csv_path, tmp_path):
        from ds_mcp_server._tools.plot_interactive import generate_custom_plotly_impl

        out = generate_custom_plotly_impl(
            csv_path,
            "fig = px.scatter(df, x='x', y='y_linear')",
            "good",
        )
        # Format on success is "path|||code"
        assert "|||" in out, f"Expected success format, got: {out[:200]}"
        path = out.split("|||", 1)[0]
        assert path.endswith(".html")

    def test_static_custom_plot_rejects_dunder(self, csv_path):
        from ds_mcp_server._tools.plot_static import generate_custom_static_plot_impl

        out = generate_custom_static_plot_impl(
            csv_path,
            "x = df.__class__\nfig, ax = plt.subplots()\nax.plot([1,2,3])",
            "bad",
        )
        assert "sandbox rejected" in out.lower() or "Error" in out
        assert "dunder" in out.lower()

    def test_static_custom_plot_accepts_valid_matplotlib(self, csv_path):
        from ds_mcp_server._tools.plot_static import generate_custom_static_plot_impl

        out = generate_custom_static_plot_impl(
            csv_path,
            "fig, ax = plt.subplots()\nax.plot(df['x'], df['y_linear'])",
            "good",
        )
        assert "|||" in out, f"Expected success format, got: {out[:200]}"


class TestServerWarningOnUnrestricted:
    """The startup warning must fire when the sandbox is disabled."""

    def test_warning_banner_on_stderr(self, monkeypatch, capsys):
        monkeypatch.setenv("DS_MCP_ALLOW_UNRESTRICTED_EXEC", "1")
        monkeypatch.delenv("DS_MCP_ENABLE_SYSTEM_TOOLS", raising=False)
        sys.modules.pop("ds_mcp_server.server", None)
        importlib.import_module("ds_mcp_server.server")
        captured = capsys.readouterr()
        assert "CUSTOM-PLOT SANDBOX IS DISABLED" in captured.err

    def test_no_warning_when_sandbox_on(self, monkeypatch, capsys):
        monkeypatch.delenv("DS_MCP_ALLOW_UNRESTRICTED_EXEC", raising=False)
        monkeypatch.delenv("DS_MCP_ENABLE_SYSTEM_TOOLS", raising=False)
        sys.modules.pop("ds_mcp_server.server", None)
        importlib.import_module("ds_mcp_server.server")
        captured = capsys.readouterr()
        assert "CUSTOM-PLOT SANDBOX IS DISABLED" not in captured.err
