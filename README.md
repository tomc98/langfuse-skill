# Langfuse MCP Server

[![PyPI](https://badge.fury.io/py/langfuse-mcp.svg)](https://badge.fury.io/py/langfuse-mcp)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10–3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Model Context Protocol](https://modelcontextprotocol.io) server for [Langfuse](https://langfuse.com) observability. Query traces, debug errors, analyze sessions, manage prompts.

## Quick Start

Get credentials from [Langfuse Cloud](https://cloud.langfuse.com) → Settings → API Keys.

```bash
# Claude Code
claude mcp add langfuse -s project \
  -e LANGFUSE_PUBLIC_KEY=pk-... \
  -e LANGFUSE_SECRET_KEY=sk-... \
  -e LANGFUSE_HOST=https://cloud.langfuse.com \
  -- uvx --python 3.11 langfuse-mcp

# Codex CLI
codex mcp add langfuse \
  --env LANGFUSE_PUBLIC_KEY=pk-... \
  --env LANGFUSE_SECRET_KEY=sk-... \
  --env LANGFUSE_HOST=https://cloud.langfuse.com \
  -- uvx --python 3.11 langfuse-mcp
```

Restart your CLI, then verify with `/mcp` (Claude Code) or `codex mcp list` (Codex).

## Tools

| Category | Tools |
|----------|-------|
| Traces | `fetch_traces`, `fetch_trace` |
| Observations | `fetch_observations`, `fetch_observation` |
| Sessions | `fetch_sessions`, `get_session_details`, `get_user_sessions` |
| Exceptions | `find_exceptions`, `find_exceptions_in_file`, `get_exception_details`, `get_error_count` |
| Prompts | `list_prompts`, `get_prompt`, `create_text_prompt`, `create_chat_prompt`, `update_prompt_labels` |

## Skill

This project includes a skill with debugging playbooks. Install globally:

```bash
cp -r skills/langfuse ~/.claude/skills/   # Claude Code
cp -r skills/langfuse ~/.codex/skills/    # Codex CLI
```

Then say "help me debug langfuse traces" to activate.

See [`skills/langfuse/SKILL.md`](skills/langfuse/SKILL.md) for full documentation.

## Other Clients

<details>
<summary>Cursor, Docker</summary>

**Cursor** — Create `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "langfuse": {
      "command": "uvx",
      "args": ["--python", "3.11", "langfuse-mcp"],
      "env": {
        "LANGFUSE_PUBLIC_KEY": "pk-...",
        "LANGFUSE_SECRET_KEY": "sk-...",
        "LANGFUSE_HOST": "https://cloud.langfuse.com"
      }
    }
  }
}
```

**Docker**:
```bash
docker run --rm -i \
  -e LANGFUSE_PUBLIC_KEY=pk-... \
  -e LANGFUSE_SECRET_KEY=sk-... \
  -e LANGFUSE_HOST=https://cloud.langfuse.com \
  ghcr.io/avivsinai/langfuse-mcp:latest
```

</details>

## Development

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

## License

MIT
