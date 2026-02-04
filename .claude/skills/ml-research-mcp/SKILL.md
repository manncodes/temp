# ml-research-mcp

## Description
This project is configured with MCP (Model Context Protocol) servers tailored for ML research workflows. This skill describes how to install, configure, and use them.

## MCP Servers Available

### 1. arxiv (`arxiv-mcp-server`)
Search, download, and read arXiv papers as markdown.

**Tools:**
- `search_papers` -- Search arxiv with advanced query syntax, category filtering, date ranges
- `download_paper` -- Download a paper by arxiv ID (e.g., `1706.03762`)
- `read_paper` -- Read a downloaded paper's full content in markdown
- `list_papers` -- List all previously downloaded papers

**Query tips:**
- Use quoted phrases: `"attention is all you need"`
- Field-specific: `ti:"transformer"`, `au:"Vaswani"`, `abs:"self-attention"`
- Category filter: `cs.LG`, `cs.CL`, `cs.CV`, `cs.AI`
- Combine with OR/ANDNOT: `"diffusion models" ANDNOT "survey"`
- Date filtering: `date_from: "2024-01-01"` for recent work

### 2. fetch (`mcp-server-fetch`)
Fetch any URL and convert it to markdown for reading.

**Tools:**
- `fetch` -- Fetch a URL, returns content as markdown. Params: `url`, `max_length`, `start_index`, `raw`

**Use cases:**
- Read web pages, blog posts, documentation
- Fetch raw content from GitHub
- Read HTML paper pages when TeX source isn't available
- Access API documentation

### 3. huggingface (HF Hub MCP)
Search and explore models, datasets, and spaces on Hugging Face Hub.

**Tools:**
- Model search and details
- Dataset search and details
- Space search

**Note:** For full access, set `HF_TOKEN` environment variable. Without it, only public resources are accessible.

## Installation (for a fresh environment)

Run these commands to install the MCP server dependencies:

```bash
# arxiv MCP server (Python, via uv)
uv tool install arxiv-mcp-server

# fetch MCP server (Python, via pip)
pip install mcp-server-fetch

# huggingface uses npx mcp-remote (no pre-install needed, npx handles it)
```

Then add the servers to Claude Code:

```bash
# Option A: CLI commands
claude mcp add --transport stdio --scope user arxiv -- arxiv-mcp-server --storage-path ~/.arxiv-papers
claude mcp add --transport stdio --scope user fetch -- python3 -m mcp_server_fetch
claude mcp add --transport http --scope user huggingface https://huggingface.co/mcp

# Option B: Project-level config is already in .mcp.json at the repo root
```

## Configuration Files

### `.mcp.json` (project root, committed to repo)
Defines the MCP servers. Claude Code reads this automatically when starting a session in this project. No manual setup needed if this file exists.

### `~/.claude.json` (user level)
For user-wide MCP config that persists across all projects. Add an `"mcpServers"` key with the same format as `.mcp.json`.

## Workflow: Reading an arXiv Paper

1. **Search:** Use `search_papers` tool with a query like `ti:"attention" AND au:"Vaswani"` with categories `["cs.CL"]`
2. **Download:** Use `download_paper` with the arxiv ID (e.g., `1706.03762`)
3. **Read:** Use `read_paper` with the same ID to get full markdown content
4. **Summarize:** Write a summary to `./knowledge/summary_{tag}.md` following the format in the `read-arxiv-paper` skill

Alternatively, use the `read-arxiv-paper` skill to download TeX source directly and parse LaTeX.

## Workflow: Exploring Hugging Face

1. Search for models/datasets using the huggingface MCP tools
2. Use `fetch` to read model cards or dataset documentation if the MCP tools don't return enough detail

## Workflow: Reading Any Web Resource

1. Use `fetch` with the URL to get markdown content
2. Use `start_index` to paginate through long pages
3. Set `raw: true` to get original HTML if markdown conversion loses important formatting

## Adding More MCP Servers

Other useful servers for ML research (install as needed):

```bash
# Semantic Scholar -- academic paper search with citation graphs
# (requires uv and git)
claude mcp add --transport stdio --scope user semantic-scholar \
  -- uv run --with git+https://github.com/FujishigeTemma/semantic-scholar-mcp semantic-scholar-mcp serve

# Papers with Code -- SOTA benchmarks and leaderboards
# (requires npx)
claude mcp add --transport stdio --scope user paperswithcode \
  -- npx -y @smithery/cli install @hbg/mcp-paperswithcode --client claude

# Kaggle -- datasets and competitions (needs KAGGLE_USERNAME + KAGGLE_KEY)
pip install kaggle-mcp
claude mcp add --transport stdio --scope user kaggle -- kaggle-mcp

# Weights & Biases -- experiment tracking (needs WANDB_API_KEY)
claude mcp add --transport http --scope user wandb https://mcp.withwandb.com/mcp

# Replicate -- run ML models on demand (needs REPLICATE_API_TOKEN)
claude mcp add --transport stdio --scope user replicate -- npx -y replicate-mcp@latest
```

## Directory Structure

```
.claude/
  skills/
    read-arxiv-paper/SKILL.md   # Skill for reading papers from TeX source
    ml-research-mcp/SKILL.md    # This file -- MCP setup and usage guide
.mcp.json                       # MCP server configuration (auto-loaded)
.arxiv-papers/                  # Downloaded arxiv papers (gitignored)
knowledge/                      # Paper summaries and research notes
```
