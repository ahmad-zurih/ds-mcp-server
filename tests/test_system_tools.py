"""Tests for _tools.system_tools (the opt-in dangerous surface)."""
from __future__ import annotations

from pathlib import Path

from ds_mcp_server._tools.system_tools import (
    find_in_files_impl,
    list_directory_impl,
    patch_file_impl,
    read_file_impl,
    run_shell_command_impl,
    write_file_impl,
)


class TestWriteReadFile:
    def test_write_then_read_roundtrip(self, tmp_path: Path):
        p = tmp_path / "sub" / "file.txt"
        write_res = write_file_impl(str(p), "hello world")
        assert "OK" in write_res
        assert p.exists()
        assert read_file_impl(str(p)) == "hello world"

    def test_read_missing_file_returns_error(self, tmp_path: Path):
        out = read_file_impl(str(tmp_path / "missing.txt"))
        assert "Error" in out and "does not exist" in out

    def test_read_rejects_large_file(self, tmp_path: Path):
        p = tmp_path / "big.txt"
        p.write_bytes(b"x" * 600_000)
        out = read_file_impl(str(p))
        assert "Error" in out and "too large" in out


class TestPatchFile:
    def test_unique_match_succeeds(self, tmp_path: Path):
        p = tmp_path / "code.py"
        p.write_text("def foo():\n    return 1\n")
        out = patch_file_impl(str(p), "return 1", "return 42")
        assert "OK" in out
        assert "return 42" in p.read_text()

    def test_not_found_returns_error_with_preview(self, tmp_path: Path):
        p = tmp_path / "code.py"
        p.write_text("hello\nworld\n")
        out = patch_file_impl(str(p), "does not exist", "x")
        assert "Error" in out
        assert "First 40 lines" in out

    def test_multiple_matches_refused(self, tmp_path: Path):
        p = tmp_path / "code.py"
        p.write_text("dup\ndup\n")
        out = patch_file_impl(str(p), "dup", "unique")
        assert "Error" in out
        assert "2 times" in out
        assert p.read_text() == "dup\ndup\n"


class TestListDirectory:
    def test_lists_files_and_dirs(self, tmp_path: Path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "file.txt").write_text("hi")
        out = list_directory_impl(str(tmp_path))
        assert "sub" in out
        assert "file.txt" in out

    def test_missing_dir_returns_error(self, tmp_path: Path):
        out = list_directory_impl(str(tmp_path / "nope"))
        assert "Error" in out


class TestFindInFiles:
    def test_finds_pattern(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("def hello():\n    pass\n")
        (tmp_path / "b.py").write_text("x = 1\n")
        out = find_in_files_impl("def hello", str(tmp_path), file_glob="*.py")
        assert "a.py" in out
        assert "def hello" in out

    def test_no_match_message(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("no matches here\n")
        out = find_in_files_impl("xyzzy_never", str(tmp_path), file_glob="*.py")
        assert "No matches" in out

    def test_invalid_regex_reports_error(self, tmp_path: Path):
        out = find_in_files_impl("[unclosed", str(tmp_path), file_glob="*")
        assert "Error" in out and "regex" in out


class TestRunShellCommand:
    def test_echo(self):
        out = run_shell_command_impl("echo hello_from_test")
        assert "hello_from_test" in out
