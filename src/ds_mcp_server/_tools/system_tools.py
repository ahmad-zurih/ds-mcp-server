"""
System Tools Module for the AI Code Agent.
Provides shell command execution, file read/write/patch, directory listing,
background process management, HTTP requests, and file search.
"""
import os
import re
import subprocess
import threading
import time
import urllib.request
import urllib.error
from typing import Any

# ---------------------------------------------------------------------------
# Background process registry  (label -> {"proc": Popen, "command": str})
# ---------------------------------------------------------------------------
_bg_processes: dict[str, dict[str, Any]] = {}
_bg_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Existing tools
# ---------------------------------------------------------------------------

def run_shell_command_impl(command: str, working_dir: str | None = None) -> str:
    try:
        cwd = working_dir if (working_dir and os.path.isdir(working_dir)) else os.path.expanduser("~")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=cwd,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if out and err:
            return f"{out}\n[stderr]\n{err}"
        return out or err or "(command produced no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds."
    except Exception as e:
        return f"Error running command: {str(e)}"


def read_file_impl(file_path: str) -> str:
    try:
        path = os.path.expanduser(file_path)
        if not os.path.exists(path):
            return f"Error: file '{path}' does not exist."
        size = os.path.getsize(path)
        if size > 500_000:
            return f"Error: file is too large ({size:,} bytes > 500 KB). Use run_shell_command with head/tail instead."
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file_impl(file_path: str, content: str) -> str:
    try:
        path = os.path.expanduser(file_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"OK: wrote {len(content):,} chars to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def list_directory_impl(dir_path: str) -> str:
    try:
        path = os.path.expanduser(dir_path)
        if not os.path.isdir(path):
            return f"Error: '{path}' is not a directory or does not exist."
        entries = []
        for entry in sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name)):
            if entry.is_dir():
                entries.append(f"  [DIR ] {entry.name}/")
            else:
                entries.append(f"  [FILE] {entry.name}  ({entry.stat().st_size:,} B)")
        body = "\n".join(entries) if entries else "  (empty)"
        return f"{path}/\n{body}"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


# ---------------------------------------------------------------------------
# New tools
# ---------------------------------------------------------------------------

def patch_file_impl(file_path: str, old_str: str, new_str: str) -> str:
    """
    Find the FIRST occurrence of old_str in the file and replace it with new_str.
    Ideal for surgical edits without rewriting the whole file.
    Returns a success message with line number, or an error if old_str is not found.
    """
    try:
        path = os.path.expanduser(file_path)
        if not os.path.exists(path):
            return f"Error: file '{path}' does not exist."
        original = open(path, "r", encoding="utf-8", errors="replace").read()
        if old_str not in original:
            # Help the model debug: show nearby content
            lines = original.splitlines()
            preview = "\n".join(f"  {i+1}: {l}" for i, l in enumerate(lines[:40]))
            return (
                f"Error: exact string not found in '{path}'.\n"
                f"First 40 lines of file for reference:\n{preview}"
            )
        count = original.count(old_str)
        if count > 1:
            return (
                f"Error: old_str appears {count} times in '{path}'. "
                "Make it more specific so exactly one match exists."
            )
        patched = original.replace(old_str, new_str, 1)
        # Find the line number of the edit
        line_no = original[: original.index(old_str)].count("\n") + 1
        open(path, "w", encoding="utf-8").write(patched)
        return f"OK: patched '{path}' at line {line_no} ({len(new_str) - len(old_str):+d} chars)"
    except Exception as e:
        return f"Error patching file: {str(e)}"


def run_background_process_impl(command: str, label: str, working_dir: str | None = None) -> str:
    """
    Start a long-running background process (e.g. a web server) and track it by label.
    Returns the PID and label. Use http_request to probe it, stop_background_process to kill it.
    """
    try:
        with _bg_lock:
            if label in _bg_processes:
                old = _bg_processes[label]
                if old["proc"].poll() is None:
                    return (
                        f"Error: a process with label '{label}' is already running "
                        f"(PID {old['proc'].pid}). Stop it first with stop_background_process."
                    )
        cwd = working_dir if (working_dir and os.path.isdir(working_dir)) else os.path.expanduser("~")
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
        )
        # Give it a moment to start or fail immediately
        time.sleep(1.5)
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            return f"Error: process '{label}' exited immediately (rc={proc.returncode}).\n{out.strip()}"
        with _bg_lock:
            _bg_processes[label] = {"proc": proc, "command": command}
        return f"OK: started '{label}' (PID {proc.pid}). Use http_request to test it."
    except Exception as e:
        return f"Error starting background process: {str(e)}"


def stop_background_process_impl(label: str) -> str:
    """
    Stop a background process started with run_background_process.
    """
    try:
        with _bg_lock:
            if label not in _bg_processes:
                labels = list(_bg_processes.keys())
                return f"Error: no process with label '{label}'. Running: {labels}"
            entry = _bg_processes.pop(label)
        proc = entry["proc"]
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        return f"OK: stopped '{label}' (PID {proc.pid}, rc={proc.returncode})"
    except Exception as e:
        return f"Error stopping process: {str(e)}"


def list_background_processes_impl() -> str:
    """
    List all background processes currently tracked (running or recently exited).
    """
    with _bg_lock:
        if not _bg_processes:
            return "No background processes running."
        lines = []
        for label, entry in _bg_processes.items():
            proc = entry["proc"]
            status = "running" if proc.poll() is None else f"exited (rc={proc.returncode})"
            lines.append(f"  [{status}] '{label}' PID={proc.pid}  cmd={entry['command']}")
        return "\n".join(lines)


def http_request_impl(url: str, method: str = "GET", body: str = "") -> str:
    """
    Make an HTTP request and return the status code + response body (truncated to 4 KB).
    Use this to test web servers started with run_background_process.
    method: GET, POST, PUT, DELETE, PATCH
    body: optional request body for POST/PUT
    """
    try:
        data = body.encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, method=method.upper())
        if data:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(4096).decode("utf-8", errors="replace")
            return f"HTTP {resp.status}\n{raw}"
    except urllib.error.HTTPError as e:
        raw = e.read(4096).decode("utf-8", errors="replace")
        return f"HTTP {e.code} {e.reason}\n{raw}"
    except urllib.error.URLError as e:
        return f"Error: could not reach {url}: {e.reason}"
    except Exception as e:
        return f"Error making request: {str(e)}"


def find_in_files_impl(pattern: str, directory: str, file_glob: str = "*.py") -> str:
    """
    Search file contents for a regex pattern (like grep -rn).
    Returns matching lines with file paths and line numbers (max 60 results).
    pattern: a regular expression
    directory: root directory to search in
    file_glob: shell glob to filter filenames, e.g. '*.py', '*.ts', '*' for all
    """
    try:
        import fnmatch
        dir_path = os.path.expanduser(directory)
        if not os.path.isdir(dir_path):
            return f"Error: '{dir_path}' is not a directory."
        compiled = re.compile(pattern)
        results = []
        for root, dirs, files in os.walk(dir_path):
            # Skip hidden dirs and common noise
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".git", "venv", ".venv")]
            for fname in files:
                if not fnmatch.fnmatch(fname, file_glob):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        for lineno, line in enumerate(f, 1):
                            if compiled.search(line):
                                rel = os.path.relpath(fpath, dir_path)
                                results.append(f"  {rel}:{lineno}: {line.rstrip()}")
                                if len(results) >= 60:
                                    results.append("  ... (truncated at 60 results)")
                                    return "\n".join(results)
                except Exception:
                    continue
        return "\n".join(results) if results else f"No matches for '{pattern}' in {dir_path}"
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"
    except Exception as e:
        return f"Error searching files: {str(e)}"