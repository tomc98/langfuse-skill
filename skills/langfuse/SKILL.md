---
name: langfuse
version: 2.0.0
description: Debug AI traces, find exceptions, analyze sessions, and manage prompts/datasets via Langfuse. Use when debugging AI, finding exceptions, analyzing traces/sessions, or managing prompts and datasets.
allowed-tools: Bash, Read
---

# Langfuse Skill

Debug your AI systems through Langfuse observability.

**Triggers:** langfuse, traces, debug AI, find exceptions, set up langfuse, what went wrong, why is it slow, datasets, evaluation sets

## Setup

**Step 1:** Get credentials from https://cloud.langfuse.com → Settings → API Keys

If self-hosted, use your instance URL for `LANGFUSE_HOST` and create keys there.

**Step 2:** Add credentials to `~/.claude/settings.json`:

```json
{
  "env": {
    "LANGFUSE_PUBLIC_KEY": "pk-...",
    "LANGFUSE_SECRET_KEY": "sk-...",
    "LANGFUSE_HOST": "https://cloud.langfuse.com"
  }
}
```

**Step 3:** Test: `python3 skills/langfuse/scripts/traces.py fetch --age 60`

For manual setup details, see `references/setup.md`.

---

## Playbooks

### "Where are the errors?"

```bash
python3 skills/langfuse/scripts/exceptions.py find --age 1440 --group-by file
```
→ Shows error counts by file. Pick the worst offender.

```bash
python3 skills/langfuse/scripts/exceptions.py file src/ai/chat.py --age 1440
```
→ Lists specific exceptions. Grab a trace_id.

```bash
python3 skills/langfuse/scripts/exceptions.py details <trace_id>
```
→ Full stacktrace and context.

---

### "What happened in this interaction?"

```bash
python3 skills/langfuse/scripts/traces.py fetch --age 60 --user-id "user_123"
```
→ Find the trace. Note the trace_id.

```bash
python3 skills/langfuse/scripts/traces.py get <trace_id>
```
→ See all details for the trace.

```bash
python3 skills/langfuse/scripts/observations.py fetch --age 60 --trace-id <trace_id>
```
→ See all observations in the trace.

```bash
python3 skills/langfuse/scripts/observations.py get <observation_id>
```
→ Inspect a specific generation's input/output.

---

### "Why is it slow?"

```bash
python3 skills/langfuse/scripts/observations.py fetch --age 60 --type GENERATION
```
→ Find recent LLM calls. Look for high latency.

```bash
python3 skills/langfuse/scripts/observations.py get <observation_id>
```
→ Check token counts, model, timing.

---

### "What's this user experiencing?"

```bash
python3 skills/langfuse/scripts/sessions.py user <user_id> --age 1440
```
→ List their sessions.

```bash
python3 skills/langfuse/scripts/sessions.py details <session_id>
```
→ See all traces in the session.

---

### "Manage datasets"

```bash
python3 skills/langfuse/scripts/datasets.py list
```
→ See all datasets.

```bash
python3 skills/langfuse/scripts/datasets.py get evaluation-set-v1
```
→ Get dataset details.

```bash
python3 skills/langfuse/scripts/datasets.py list-items evaluation-set-v1 --limit 10
```
→ Browse items.

```bash
python3 skills/langfuse/scripts/datasets.py create qa-test-cases --description "QA evaluation set"
```
→ Create a new dataset.

```bash
python3 skills/langfuse/scripts/datasets.py create-item qa-test-cases \
  --input '{"question": "What is 2+2?"}' \
  --expected-output '{"answer": "4"}'
```
→ Add test cases.

```bash
python3 skills/langfuse/scripts/datasets.py create-item qa-test-cases \
  --item-id item_123 \
  --input '{"question": "What is 3+3?"}' \
  --expected-output '{"answer": "6"}'
```
→ Upsert: updates existing item by id or creates if missing.

---

### "Manage prompts"

```bash
python3 skills/langfuse/scripts/prompts.py list
```
→ See all prompts with labels.

```bash
python3 skills/langfuse/scripts/prompts.py get my-prompt --label production
```
→ Fetch current production version.

```bash
python3 skills/langfuse/scripts/prompts.py create-text my-prompt --prompt "Hello {{name}}" --labels staging
```
→ Create new version in staging.

```bash
python3 skills/langfuse/scripts/prompts.py update-labels my-prompt --version 3 --labels production
```
→ Promote to production. (Rollback = re-apply label to older version)

---

## Quick Reference

| Task | Command |
|------|---------|
| List traces | `traces.py fetch --age 60` |
| Get trace | `traces.py get <id>` |
| List LLM calls | `observations.py fetch --age 60 --type GENERATION` |
| Get observation | `observations.py get <id>` |
| Error count | `exceptions.py count --age 60` |
| Find exceptions | `exceptions.py find --age 1440 --group-by file` |
| List sessions | `sessions.py fetch --age 1440` |
| User sessions | `sessions.py user <user_id> --age 1440` |
| List prompts | `prompts.py list` |
| Get prompt | `prompts.py get <name> --label production` |
| List datasets | `datasets.py list` |
| Get dataset | `datasets.py get <name>` |
| Dataset items | `datasets.py list-items <name> --limit 10` |
| Create dataset item | `datasets.py create-item <name> --input '...'` |
| Schema reference | `schema.py` |

All scripts are at `skills/langfuse/scripts/`. Prefix with `python3 skills/langfuse/scripts/`.

`--age` = minutes to look back (max 10080 = 7 days). `--no-truncate` = full output.

---

## References

- `references/tool-reference.md` — Full parameter docs, response schemas
- `references/setup.md` — Credentials setup, troubleshooting
