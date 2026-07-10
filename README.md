# ds-mcp-server

`ds-mcp-server` packages a FastMCP server with data science, plotting, statistics, system, and web tools, plus interactive CLI clients for OpenAI-compatible providers and Anthropic Claude.

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

## Quick start

1. Copy `.env.example` to `.env`.
2. Fill in your provider settings.
3. Install the package.
4. Start either the MCP server or the interactive client.

### OpenAI

```bash
export PROVIDER=openai
export API_KEY=sk-...
export MODEL=gpt-4o
ds-mcp-client
```

### Claude / Anthropic

```bash
export PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export MODEL=claude-opus-4-5
ds-mcp-client
```

### Gemini (OpenAI-compatible endpoint)

```bash
export PROVIDER=gemini
export API_KEY=AIza...
export MODEL=gemini-2.0-flash
ds-mcp-client
```

### Ollama

```bash
export PROVIDER=ollama
export API_BASE_URL=http://localhost:11434/v1
export MODEL=llama3
ds-mcp-client
```

### GPUStack / LM Studio / other OpenAI-compatible servers

```bash
export PROVIDER=openai-compat
export API_BASE_URL=https://your-endpoint.example/v1
export API_KEY=your-key
export MODEL=your-model
ds-mcp-client
```

## Running the MCP server

```bash
ds-mcp-server
```

To also expose the optional (dangerous) system tools — shell execution, file
read/write/patch, background processes, and HTTP requests:

```bash
ds-mcp-server --enable-system-tools
# or, equivalently:
DS_MCP_ENABLE_SYSTEM_TOOLS=1 ds-mcp-server
```

See the [Optional system tools](#optional-system-tools) section below before enabling.

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

## Claude Desktop MCP config

Add the server to your Claude Desktop MCP configuration:

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
