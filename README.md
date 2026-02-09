# Langfuse Skill

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Stateless Python scripts for [Langfuse](https://langfuse.com) observability. Query traces, debug errors, analyze sessions, manage prompts and datasets — no external dependencies.

## Why this skill?

- **Zero dependencies** — stdlib only (urllib, json, argparse). No SDK version conflicts, no Python version restrictions.
- **Stateless** — each invocation is independent. No persistent processes, no caches to invalidate.
- **Full observability** — traces, observations, sessions, exceptions, prompts, datasets, and schema.

## Quick Start

**1. Get credentials** from [Langfuse Cloud](https://cloud.langfuse.com) → Settings → API Keys.

**2. Add to `~/.claude/settings.json`:**

```json
{
  "env": {
    "LANGFUSE_PUBLIC_KEY": "pk-...",
    "LANGFUSE_SECRET_KEY": "sk-...",
    "LANGFUSE_HOST": "https://cloud.langfuse.com"
  }
}
```

**3. Test:**

```bash
python3 skills/langfuse/scripts/traces.py fetch --age 60
```

## Scripts

| Script | Commands | Description |
|--------|----------|-------------|
| `traces.py` | fetch, get | Search and retrieve traces |
| `observations.py` | fetch, get | Search and retrieve observations |
| `sessions.py` | fetch, details, user | Session listing and details |
| `exceptions.py` | find, file, details, count | Exception analysis |
| `prompts.py` | list, get, get-unresolved, create-text, create-chat, update-labels | Prompt management |
| `datasets.py` | list, get, list-items, get-item, create, create-item, delete-item | Dataset management |
| `schema.py` | *(none)* | Data schema reference |

All scripts are in `skills/langfuse/scripts/`. Run with:

```bash
python3 skills/langfuse/scripts/<script>.py <command> [args]
```

## Examples

```bash
# Find recent traces
python3 skills/langfuse/scripts/traces.py fetch --age 60

# Find exceptions by file
python3 skills/langfuse/scripts/exceptions.py find --age 1440 --group-by file

# Get a prompt
python3 skills/langfuse/scripts/prompts.py get my-prompt --label production

# Create a dataset item
python3 skills/langfuse/scripts/datasets.py create-item my-dataset \
  --input '{"question": "What is 2+2?"}' \
  --expected-output '{"answer": "4"}'
```

## Skill Installation

**Via [skills](https://github.com/vercel-labs/add-skill)** (recommended):
```bash
npx skills add avivsinai/langfuse-mcp -g -y
```

**Manual install:**
```bash
cp -r skills/langfuse ~/.claude/skills/   # Claude Code
cp -r skills/langfuse ~/.codex/skills/    # Codex CLI
```

Try asking: "help me debug langfuse traces"

See [`skills/langfuse/SKILL.md`](skills/langfuse/SKILL.md) for full documentation.

## Development

```bash
# Tests
python3 -m pytest tests/

# Lint
ruff check skills/ tests/
ruff format skills/ tests/
```

## License

MIT
