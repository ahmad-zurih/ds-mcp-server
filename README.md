# ds-mcp-server

`ds-mcp-server` packages a FastMCP server with data science, plotting, statistics, system, and web tools, plus interactive CLI clients for OpenAI-compatible providers and Anthropic Claude.

## What's in the box

After you `pip install ds-mcp-server`, three commands are available:

| Command | What it is | When to use it |
|---|---|---|
| **`ds-mcp-webui`** | Browser chat UI | You want to chat and see plots in your browser. **Start here.** |
| **`ds-mcp-client`** | Interactive terminal chat | You prefer the CLI. Same features as the web UI, minus inline plot rendering. |
| **`ds-mcp-server`** | The MCP server itself | You are configuring an **external** MCP client (Claude Desktop, LM Studio, Cursor, etc.) to launch it. **Do not run this by hand** — it will look "frozen" because it's silently waiting for MCP protocol messages on stdin. |

> **In short:** for humans → `ds-mcp-webui` or `ds-mcp-client`.
> For MCP clients configured with a `command` field → `ds-mcp-server`.

## Installation

Install from PyPI:

```bash
pip install ds-mcp-server
```

Local development install:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[anthropic]"
pip install -e ".[playwright]"
pip install -e ".[all]"
```

## First-time setup (all platforms)

After `pip install`, you need to tell the client which LLM to talk to. The
fastest way is to let the tool generate a template for you:

```bash
ds-mcp-webui --init-env
```

This writes a `.env` file to the current folder with every provider commented
out. Open it in any text editor, uncomment the block for your provider, paste
your API key, save, then run `ds-mcp-webui` again from the same folder.

### Alternative: set the variables directly for one session

**Windows Command Prompt**
```cmd
set PROVIDER=openai
set API_KEY=sk-your-key-here
set MODEL=gpt-4o
ds-mcp-webui
```

**Windows PowerShell**
```powershell
$env:PROVIDER = "openai"
$env:API_KEY  = "sk-your-key-here"
$env:MODEL    = "gpt-4o"
ds-mcp-webui
```

**macOS / Linux (bash / zsh)**
```bash
export PROVIDER=openai
export API_KEY=sk-your-key-here
export MODEL=gpt-4o
ds-mcp-webui
```

If you run `ds-mcp-webui` or `ds-mcp-client` without any credentials
configured, you'll get a helpful setup message pointing you at these same
options — you can't get stuck.

## Quick start

1. Copy `.env.example` to `.env`.
2. Fill in your provider settings.
3. Install the package.
4. Run **`ds-mcp-webui`** (browser) or **`ds-mcp-client`** (terminal) to chat.

> The examples below use `export …` (bash/zsh syntax). On **Windows**, use
> `set …` in Command Prompt or `$env:… = "…"` in PowerShell — see
> [First-time setup](#first-time-setup-all-platforms) above, or just run
> `ds-mcp-webui --init-env` and edit the generated `.env` file.

### OpenAI

```bash
export PROVIDER=openai
export API_KEY=sk-...
export MODEL=gpt-4o
ds-mcp-webui         # browser chat  →  http://127.0.0.1:8765
# or
ds-mcp-client        # terminal chat
```

### Claude / Anthropic

```bash
export PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export MODEL=claude-opus-4-5
ds-mcp-webui
```

### Gemini (OpenAI-compatible endpoint)

```bash
export PROVIDER=gemini
export API_KEY=AIza...
export MODEL=gemini-2.0-flash
ds-mcp-webui
```

### Ollama

```bash
export PROVIDER=ollama
export API_BASE_URL=http://localhost:11434/v1
export MODEL=llama3
ds-mcp-webui
```

### GPUStack / LM Studio / other OpenAI-compatible servers

```bash
export PROVIDER=openai-compat
export API_BASE_URL=https://your-endpoint.example/v1
export API_KEY=your-key
export MODEL=your-model
ds-mcp-webui
```

## Running the MCP server (for external MCP clients only)

If you are wiring up an **external** MCP client — Claude Desktop, LM Studio,
Cursor, or anything else that spawns MCP servers as subprocesses — point it at
the `ds-mcp-server` command. **You don't run this yourself in a terminal**;
the MCP client does it for you and talks to it over stdin/stdout.

```bash
ds-mcp-server                     # what an MCP client will invoke for you
ds-mcp-server --enable-system-tools    # add shell/file/HTTP tools (dangerous)
```

If you ran `ds-mcp-server` in your terminal and it appears to hang after
printing a startup line — that's expected. It's waiting for MCP protocol
messages that only an MCP client can send. Press **Ctrl+C** to exit and use
`ds-mcp-webui` or `ds-mcp-client` instead.

See the [Optional system tools](#optional-system-tools) section below before
enabling the `--enable-system-tools` flag.

## ⚠️ Optional system tools

By default `ds-mcp-server` only exposes safe read-only data-science tools
(plots, statistics, dataset summaries, web fetch/search). A second group of
**system / coder tools** is bundled in the package but is **disabled by default**
because it grants the connected LLM effectively remote-code-execution power.

The gated tools are:

- `run_shell_command` — runs any shell command with your user's privileges
- `read_file`, `write_file`, `patch_file`, `list_directory` — arbitrary file I/O
- `find_in_files` — regex-search anywhere on disk
- `run_background_process`, `stop_background_process`, `list_background_processes`
- `http_request` — arbitrary outbound HTTP (SSRF risk: can reach localhost,
  cloud metadata endpoints, internal services, etc.)

### Enabling

Only enable inside a sandbox you trust (Docker container, WSL, dedicated VM,
or a throwaway user account). The LLM decides when to call these — a single
prompt-injection or misinterpretation is enough to trigger destructive actions.

Two equivalent ways to enable:

```bash
# Preferred: env var, works with any MCP client (Claude Desktop, LM Studio, …)
export DS_MCP_ENABLE_SYSTEM_TOOLS=1

# Or as a CLI flag when launching the server directly
ds-mcp-server --enable-system-tools
```

When enabled, the server prints a warning banner to stderr at startup listing
every dangerous tool that was registered. When disabled, it prints a one-line
hint telling you how to opt in.

### Claude Desktop config with system tools enabled

```json
{
  "mcpServers": {
    "ds-mcp-server": {
      "command": "ds-mcp-server",
      "args": ["--enable-system-tools"]
    }
  }
}
```

## 🔒 Sandbox for LLM-generated plotting code

Two tools — `generate_custom_plotly` and `generate_custom_static_plot` — accept
a Python code string produced by the LLM and `exec()` it in-process to render a
plot. Because that code can be influenced by any dataset, webpage, or file the
model reads, `ds-mcp-server` sandboxes it **by default**.

### What the sandbox blocks

- `import` and `from ... import` statements (all needed libraries — `pd`, `np`,
  `px`, `go`, `plt`, `sns`, `WordCloud`, `df` — are pre-injected).
- Calls to `eval`, `exec`, `compile`, `open`, `__import__`, `getattr`,
  `setattr`, `delattr`, `globals`, `locals`, `vars`, `input`, `breakpoint`.
- Access to any dunder attribute (`.__class__`, `.__subclasses__`, etc.) — this
  closes the common `().__class__.__mro__[-1].__subclasses__()` escape.
- Runaway execution — a 60s wall-clock timeout aborts the tool call.

### What the sandbox does NOT block (honest limits)

- **Filesystem access via pre-imported libraries.** `pd.read_csv("/etc/passwd")`
  still works because pandas legitimately needs to read files. For strong
  isolation run the server inside a container, VM, or dedicated user account.
- **Native-code CPU/memory exhaustion.** Python threads cannot interrupt C
  extensions, so the timeout is best-effort against numpy/pandas hot loops.

### Disabling the sandbox

If you trust the LLM and want unrestricted `exec` (e.g. for advanced plotting
that legitimately needs `import`), you can opt out:

```bash
# Env var (works with any MCP client)
export DS_MCP_ALLOW_UNRESTRICTED_EXEC=1

# Or CLI flag
ds-mcp-server --allow-unrestricted-exec
```

When disabled, the server prints a warning banner to stderr at startup.

## Claude Desktop MCP config

Add the server to your Claude Desktop MCP configuration:

## 🖥️ Browser chat UI (optional)

Prefer clicking over typing? `ds-mcp-server` ships with an optional
browser-based chat UI that talks to the same MCP server and renders plots
inline (interactive Plotly HTML in an iframe, PNG/SVG as images).

It's included in the base install — no extras needed:

```bash
pip install ds-mcp-server
```

Launch it (with your `.env` in the current directory or in `~/.env`):

```bash
ds-mcp-webui                # http://127.0.0.1:8765
ds-mcp-webui --port 9000    # custom port
ds-mcp-webui -p openai -m gpt-4o
```

Then open the printed URL in your browser. The UI:

- Streams tool calls as they happen (little pill chips per tool).
- Renders generated plots inline — interactive Plotly plots are fully
  scrollable/zoomable directly in the chat.
- Shows all available MCP tools in a searchable sidebar.
- Works with any provider the CLI client supports (OpenAI, Anthropic,
  Gemini, Ollama, LM Studio / GPUStack / any OpenAI-compat endpoint).
- **Settings panel** (⚙ in the sidebar) lets you toggle the dangerous
  opt-ins — **System / coder tools** and **Unrestricted `exec()`** — with
  clear warnings. Toggling either one restarts the underlying MCP process
  so the change takes effect without leaving the browser.

By default it binds to `127.0.0.1` (localhost only). Use `--host 0.0.0.0`
to expose it on your LAN — but be aware that anyone reaching the port can
chat through your API key.

```json
{
  "mcpServers": {
    "ds-mcp-server": {
      "command": "ds-mcp-server",
      "args": []
    }
  }
}
```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `PROVIDER` | No | One of `openai`, `anthropic`, `gemini`, `ollama`, `openai-compat`. |
| `API_KEY` | Usually | Generic API key used by OpenAI-compatible providers and as a fallback for Anthropic. |
| `ANTHROPIC_API_KEY` | Anthropic only | Preferred Anthropic key. |
| `API_BASE_URL` | Sometimes | Required for `openai-compat`; optional override for Ollama, Gemini, or self-hosted endpoints. |
| `MODEL` | No | Model override. Defaults are provider-specific. |

## Available tools

### Interactive plots

- `plot_interactive_histogram`
- `plot_interactive_scatterplot`
- `plot_interactive_boxplot`
- `plot_interactive_lineplot`
- `plot_interactive_barchart`
- `plot_interactive_scatter_matrix`
- `plot_interactive_correlation_heatmap`
- `generate_custom_plotly`
- `get_all_columns_summary`
- `get_column_summary`

### Static plots

- `plot_static_histogram`
- `plot_static_scatterplot`
- `plot_static_boxplot`
- `plot_static_lineplot`
- `plot_static_barchart`
- `plot_static_pairplot`
- `plot_static_correlation_heatmap`
- `plot_static_wordcloud`
- `generate_custom_static_plot`

### Statistical analysis

- `run_correlation`
- `run_group_comparison`
- `run_linear_regression`
- `rank_target_correlations`

### System tools (opt-in — see [Optional system tools](#optional-system-tools))

Only registered when `DS_MCP_ENABLE_SYSTEM_TOOLS=1` (or `--enable-system-tools`).

- `run_shell_command`
- `read_file`
- `write_file`
- `patch_file`
- `list_directory`
- `find_in_files`
- `run_background_process`
- `stop_background_process`
- `list_background_processes`
- `http_request`

### Web tools

- `search_web`
- `fetch_webpage`
- `screenshot_webpage`

## Requirements

- Python 3.11+
- `mcp`
- `pandas`, `numpy`
- `plotly`, `matplotlib`, `seaborn`, `wordcloud`
- `pingouin`, `statsmodels`
- `beautifulsoup4`, `ddgs`
- `openai`
- `anthropic` (optional)
- `playwright` (optional, for screenshots)
