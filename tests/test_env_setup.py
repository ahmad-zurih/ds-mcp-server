"""Tests for the pip-user onboarding helpers (env_setup)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ds_mcp_server import env_setup


class TestWriteStarterEnv:
    def test_writes_env_to_directory(self, tmp_path):
        path = env_setup.write_starter_env(tmp_path)
        assert path == (tmp_path / ".env").resolve()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "PROVIDER=openai" in content
        assert "API_KEY" in content

    def test_refuses_to_overwrite(self, tmp_path):
        (tmp_path / ".env").write_text("EXISTING=1", encoding="utf-8")
        with pytest.raises(FileExistsError):
            env_setup.write_starter_env(tmp_path)
        # Original untouched
        assert (tmp_path / ".env").read_text(encoding="utf-8") == "EXISTING=1"

    def test_force_overwrites(self, tmp_path):
        (tmp_path / ".env").write_text("EXISTING=1", encoding="utf-8")
        env_setup.write_starter_env(tmp_path, force=True)
        assert "PROVIDER=openai" in (tmp_path / ".env").read_text(encoding="utf-8")


class TestRunInitEnv:
    def test_success_returns_zero(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        rc = env_setup.run_init_env()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Wrote starter .env" in out
        assert (tmp_path / ".env").exists()

    def test_refuses_existing_returns_one(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("KEEP=1", encoding="utf-8")
        rc = env_setup.run_init_env()
        assert rc == 1
        err = capsys.readouterr().err
        assert "already exists" in err
        assert "--force" in err

    def test_force_overwrites_and_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("KEEP=1", encoding="utf-8")
        rc = env_setup.run_init_env(argv_force=True)
        assert rc == 0
        assert "PROVIDER=openai" in (tmp_path / ".env").read_text(encoding="utf-8")


class TestCredentialsLookMissing:
    def test_all_unset_returns_true(self, monkeypatch):
        for k in ("API_KEY", "ANTHROPIC_API_KEY", "PROVIDER"):
            monkeypatch.delenv(k, raising=False)
        assert env_setup.credentials_look_missing() is True

    def test_api_key_set_returns_false(self, monkeypatch):
        for k in ("ANTHROPIC_API_KEY", "PROVIDER"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("API_KEY", "sk-something")
        assert env_setup.credentials_look_missing() is False

    def test_anthropic_key_set_returns_false(self, monkeypatch):
        for k in ("API_KEY", "PROVIDER"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-something")
        assert env_setup.credentials_look_missing() is False

    def test_provider_set_returns_false(self, monkeypatch):
        for k in ("API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PROVIDER", "ollama")
        assert env_setup.credentials_look_missing() is False


class TestPrintSetupHint:
    def test_mentions_all_three_platforms(self, capsys):
        env_setup.print_setup_hint(program="ds-mcp-webui")
        err = capsys.readouterr().err
        assert "Command Prompt" in err
        assert "PowerShell" in err
        assert "bash" in err or "zsh" in err
        assert "--init-env" in err
        assert "ds-mcp-webui" in err

    def test_uses_given_program_name(self, capsys):
        env_setup.print_setup_hint(program="ds-mcp-client")
        err = capsys.readouterr().err
        assert "ds-mcp-client --init-env" in err


class TestCliInitEnv:
    """Verify --init-env flag is wired through both CLI entry points."""

    def test_webui_init_env_flag(self, tmp_path, monkeypatch, capsys):
        import sys
        from ds_mcp_server import cli

        monkeypatch.chdir(tmp_path)
        old_argv = sys.argv[:]
        sys.argv = ["ds-mcp-webui", "--init-env"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli.webui()
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv
        assert (tmp_path / ".env").exists()

    def test_client_init_env_flag(self, tmp_path, monkeypatch):
        import sys
        from ds_mcp_server import cli

        monkeypatch.chdir(tmp_path)
        old_argv = sys.argv[:]
        sys.argv = ["ds-mcp-client", "--init-env"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli.chat()
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv
        assert (tmp_path / ".env").exists()


class TestCliMissingCredsHint:
    def test_webui_exits_with_hint_when_no_creds(self, tmp_path, monkeypatch, capsys):
        import sys
        from ds_mcp_server import cli

        # Isolate from user's real env + block .env file discovery
        for k in ("API_KEY", "ANTHROPIC_API_KEY", "PROVIDER"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "no-such-home"))

        old_argv = sys.argv[:]
        sys.argv = ["ds-mcp-webui"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli.webui()
            assert exc_info.value.code == 1
        finally:
            sys.argv = old_argv
        err = capsys.readouterr().err
        assert "no provider configured" in err
        assert "--init-env" in err
