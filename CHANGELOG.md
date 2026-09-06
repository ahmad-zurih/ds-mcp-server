# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.4] - 2026-09-06

### Fixed
- **Web UI now streams tokens progressively.** The chat UI previously waited
  for the full LLM response before displaying anything (all-at-once output).
  The backend now runs the synchronous OpenAI/Anthropic streaming SDK calls in
  a daemon thread and pipes each token into an `asyncio.Queue`, so the event
  loop is never blocked and the browser receives `text_delta` events in real
  time as the model generates them.
- The frontend `app.js` handler now processes `text_delta` events to update
  the message bubble incrementally, matching the streaming backend.

### Changed
- Bumped minimum dependency versions to current stable releases:
  `openai>=3.8`, `fastapi>=0.141`, `uvicorn>=0.52`, `websockets>=17.1`,
  `anthropic>=1.4`, `playwright>=1.62`, `plotly>=7.0`, `seaborn>=0.13.2`,
  `pingouin>=0.6`, `statsmodels>=0.15`, `ddgs>=9.16`, `python-dotenv>=1.2`,
  `python-multipart>=0.0.32`, `wordcloud>=1.9.6`, `beautifulsoup4>=4.15`,
  `pypdf>=6.17`, `pdfplumber>=0.11.10`, `python-docx>=1.2`, `openpyxl>=3.1.5`,
  `pytesseract>=0.3.13`, `Pillow>=12.3`, `ydata-profiling>=4.18`,
  `youtube-transcript-api>=1.2`, `httpx>=0.28`, `reportlab>=5.0`.
  Upper-bound constraints added for `pandas<3.0`, `matplotlib<=3.10`, and
  `numpy<2.4` to remain compatible with `ydata-profiling`'s strict pins.

### Tests
- Updated `_FakeResp` / `_FakeCompletions` in `test_chat_safeguards.py` to
  implement the streaming context-manager protocol (`__enter__`/`__exit__`/
  `__iter__`) required by the new `run_openai_turn` implementation. Assertions
  updated to match `text_delta` events instead of the old single `text` event.

## [0.3.3] - 2026-07-30

### Fixed
- **Chat turns can no longer hang forever.** The single-agent web and terminal
  loops previously ran `while True` and only exited when the model returned a
  message with no tool calls — a weak or looping model (or malformed tool-call
  arguments) could keep calling tools indefinitely, leaving the web UI spinning
  with no end. All single-agent loops are now capped at `DS_MCP_MAX_STEPS`
  rounds (default 16) and emit a clear "stopped after N steps" message instead
  of hanging.
- **Stalled provider requests fail fast.** The OpenAI/Anthropic clients are now
  built with an explicit request timeout (`DS_MCP_LLM_TIMEOUT`, default 300s)
  instead of relying on the SDK's ~10-minute default.
- **Hanging tools time out.** `call_tool_async` is now bounded by
  `DS_MCP_TOOL_TIMEOUT` (default 180s), so a stuck tool (slow URL fetch, a
  shell command that never returns, ...) returns a readable timeout error
  rather than blocking the whole turn.
- Blocking LLM SDK calls in the web path now run via `asyncio.to_thread`, so a
  slow provider no longer freezes the server's event loop / other traffic.

### Changed
- **Consolidated and enriched the system prompts.** Introduced a single
  capability-aware prompt builder (`ds_mcp_server.prompts.build_system_prompt`)
  used by every single-agent entry point (terminal Anthropic + OpenAI clients,
  web Anthropic + OpenAI paths). The prompt now reflects the full toolset —
  data inspection, interactive/static plotting, statistics, documents/OCR,
  web and research — and only includes guidance for the tool groups actually
  loaded. When the optional system/shell tools are enabled it adds a short,
  practical "system & file tools" guardrail (read before editing, prefer
  `patch_file`, avoid destructive commands).
- Multi-agent workers now receive a focused per-category playbook, and the
  supervisor gains a guardrail note when the `system` worker is available.

### Fixed
- The web UI's OpenAI-compatible path (Ollama / LM Studio / OpenAI) previously
  ran with **no system prompt** — the configured prompt was silently dropped.
  It is now threaded through correctly.
- The terminal OpenAI-compatible client had no system prompt at all; it now
  uses the shared capability-aware prompt.

### Removed
- Deleted the large block of dead multi-agent prompts and tool-scoping tables
  from `_tools/viz_config.py` (obsolete `delegate_task`/`coder` design). Only
  the still-used `MAX_ROWS` constant remains.

## [0.3.0] - 2026-07-29

### Added
- **Multi-agent (supervisor/worker) architecture.** A planner/supervisor agent
  delegates tasks to specialized worker agents, each of which only sees the
  tools for its category (plotting, stats, documents, web, research, ...). This
  keeps any single LLM from having to reason over all tools at once. Supports
  separate planner and worker models, a configurable number of supervisor
  rounds, per-worker retries and step limits. Toggleable from the CLI, the
  web UI, and `DS_MCP_MULTI_AGENT`.
- **Document / file-intelligence tools** (optional `documents`/`ocr` extras):
  - `read_pdf` — extract text (and optionally tables) from a PDF, with page
    ranges like `"1,3,5-8"`.
  - `extract_tables_from_pdf` — pull structured tables out of a PDF as markdown.
  - `read_docx` — extract paragraphs and tables from Word documents.
  - `read_excel_sheets` — list sheets and preview the first rows of each.
  - `ocr_image` — OCR text from screenshots/scans/photos (needs the system
    `tesseract` binary).
  - `summarize_document` — chunk a long PDF/DOCX/TXT for LLM summarization.
- **`profile_dataset`** (optional `profiling` extra) — one-shot interactive
  ydata-profiling HTML report (types, distributions, missing values,
  correlations, warnings) rendered inline in the web UI.
- **Web / research tools** — `fetch_webpage`, `search_web`, `screenshot_webpage`,
  `screenshot_webpages`, `arxiv_search`, `github_search`, `github_read_file`,
  `wikipedia`, and `youtube_transcript` (the last via the optional `research`
  extra).
- **Chat history** in both the web UI and the CLI client, including in the
  multi-agent flow.
- **File upload in the web UI** — attach a file (📎) directly in the composer;
  the saved path is passed to the model so it can call any document/data tool
  on it.

### Changed
- Pinned `mcp>=1.0.0,<2`. The MCP Python SDK v2 (released 2026-07-28) reworked
  the package and removed `mcp.server.fastmcp`; the upper bound keeps the
  server importable until a v2 migration is done.
- Markdown output in the web UI is now rendered reliably (bold, tables, etc.).
- Documented that `ds-mcp-webui` / `ds-mcp-client` are intended for **local,
  single-user** use and should not be exposed to untrusted networks or users.

### Fixed
- Unrendered markdown in the web chat view.

## [0.2.2] - 2026-07-22

### Added
- Windows installation instructions.

## [0.2.1] - 2026-07-21

### Changed
- The web UI is now bundled into the base install, so `ds-mcp-webui` works out
  of the box (the `[web]` extra is now a no-op kept for compatibility).

## [0.2.0] - 2026-07-21

### Added
- Browser-based web GUI (`ds-mcp-webui`).

## [0.1.3] - 2026-07-10

### Added
- Sandboxing for LLM-generated custom-plot code, with an
  `--allow-unrestricted-exec` escape hatch and a loud stderr warning when the
  sandbox is disabled.
- Release workflow and test suite.
- Opt-in gating for the dangerous "system tools" group
  (`DS_MCP_ENABLE_SYSTEM_TOOLS` / `--enable-system-tools`), disabled by default.

[Unreleased]: https://github.com/ahmad-zurih/ds-mcp-server/compare/v0.3.4...HEAD
[0.3.4]: https://github.com/ahmad-zurih/ds-mcp-server/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/ahmad-zurih/ds-mcp-server/compare/v0.3.0...v0.3.3
[0.3.0]: https://github.com/ahmad-zurih/ds-mcp-server/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/ahmad-zurih/ds-mcp-server/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ahmad-zurih/ds-mcp-server/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ahmad-zurih/ds-mcp-server/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/ahmad-zurih/ds-mcp-server/releases/tag/v0.1.3
