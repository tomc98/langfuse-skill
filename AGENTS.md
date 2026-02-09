# Claude Code Instructions

## Project summary
Langfuse skill — stateless Python scripts that query the Langfuse REST API for traces, observations, sessions, exceptions, prompts, and datasets. No external dependencies (stdlib only).

## Repo layout
- `skills/langfuse/scripts/` — Script files (traces.py, observations.py, sessions.py, exceptions.py, prompts.py, datasets.py, schema.py)
- `skills/langfuse/scripts/_client.py` — Shared auth, HTTP, truncation utilities
- `skills/langfuse/SKILL.md` — Skill definition with playbooks
- `skills/langfuse/references/` — Setup and tool reference docs
- `tests/` — Pytest suite

## Common commands
```bash
# Run a script
python3 skills/langfuse/scripts/traces.py fetch --age 60

# Tests
python3 -m pytest tests/

# Lint/format
ruff format skills/ tests/
ruff check --fix skills/ tests/
```

## Environment variables
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- Stored in `~/.claude/settings.json` under `env`

## Code style
- Stdlib only — no external dependencies
- Argparse subcommands per script
- Ruff enforces formatting and lint rules
- Line length: 140; target Python: 3.10+

## When adding commands
- Add tests in `tests/test_scripts.py`
- Update `references/tool-reference.md`
- Update SKILL.md quick reference table
