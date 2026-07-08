# ds-mcp-server

`ds-mcp-server` packages a FastMCP server with data science, plotting, statistics, system, and web tools, plus interactive CLI clients for OpenAI-compatible providers and Anthropic Claude.

## Installation

Future PyPI install:

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

### System tools

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
