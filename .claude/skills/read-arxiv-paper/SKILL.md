# read-arxiv-paper

## Description
Retrieve and analyze arXiv papers by processing their TeX source code rather than PDFs.

## Steps
1. Given an arXiv URL like `https://www.arxiv.org/abs/2601.07372`, transform it to a source URL by replacing `/abs/` with `/src/` (e.g., `https://www.arxiv.org/src/2601.07372`)
2. Download the `.tar.gz` source archive to `~/.cache/arxiv-papers/{arxiv_id}.tar.gz` (skip if already cached)
3. Extract the contents into `~/.cache/arxiv-papers/{arxiv_id}/`
4. Find the main LaTeX entrypoint file (usually `main.tex` or similar)
5. Read the entrypoint and recursively read all `\input{}` and `\include{}` referenced source files
6. Generate a markdown summary at `./knowledge/summary_{tag}.md` where `{tag}` is a short descriptive name (e.g., `conditional_memory`, `sparse_attention`)

## Summary Format
The summary should include:
- Paper title, authors, and arxiv ID
- Core contribution / key idea (1-2 paragraphs)
- Method overview with key equations or algorithms
- Main results and comparisons
- Potential applications and implementation insights
- Connections to relevant parts of the current codebase (if applicable)

## Notes
- Use local `./knowledge/` directory for summaries (not `~/.cache/`)
- Check that the tag name is unique before creating the summary file
- If the source is not available (some papers don't provide it), fall back to the arxiv MCP server's `download_paper` and `read_paper` tools
- Focus on practical, implementable insights rather than exhaustive detail
