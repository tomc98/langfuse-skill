# Langfuse Script Command Reference

All scripts are in `skills/langfuse/scripts/`. Run with `python3 skills/langfuse/scripts/<script>.py <command> [args]`.

Common flags: `--no-truncate` outputs full data without truncation. `--age` is minutes to look back (max 10080 = 7 days).

## Scripts by Category

| Category | Script | Commands |
|----------|--------|----------|
| Traces | `traces.py` | fetch, get |
| Observations | `observations.py` | fetch, get |
| Sessions | `sessions.py` | fetch, details, user |
| Exceptions | `exceptions.py` | find, file, details, count |
| Prompts | `prompts.py` | list, get, get-unresolved, create-text, create-chat, update-labels |
| Datasets | `datasets.py` | list, get, list-items, get-item, create, create-item, delete-item |
| Schema | `schema.py` | *(no subcommands)* |

---

## Traces

### `traces.py fetch`

Search and filter traces with pagination.

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `--age` | Yes | - | Minutes to look back (max 10080) |
| `--name` | No | - | Filter by trace name |
| `--user-id` | No | - | Filter by user ID |
| `--session-id` | No | - | Filter by session ID |
| `--metadata` | No | - | JSON metadata filter |
| `--tags` | No | - | Comma-separated tags |
| `--page` | No | 1 | Page number |
| `--limit` | No | 50 | Items per page |
| `--no-truncate` | No | - | Full output |

```bash
python3 skills/langfuse/scripts/traces.py fetch --age 1440 --user-id "user_123"
```

### `traces.py get`

Fetch a specific trace by ID.

| Arg | Required | Description |
|-----|----------|-------------|
| `trace_id` | Yes | Trace ID (positional) |
| `--no-truncate` | No | Full output |

```bash
python3 skills/langfuse/scripts/traces.py get abc-123
```

---

## Observations

### `observations.py fetch`

Search and filter observations (spans, generations, events).

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `--age` | Yes | - | Minutes to look back |
| `--type` | No | - | SPAN, GENERATION, or EVENT |
| `--name` | No | - | Filter by name |
| `--user-id` | No | - | Filter by user ID |
| `--trace-id` | No | - | Filter by trace ID |
| `--parent-observation-id` | No | - | Filter by parent observation |
| `--page` | No | 1 | Page number |
| `--limit` | No | 50 | Items per page |

```bash
python3 skills/langfuse/scripts/observations.py fetch --age 60 --type GENERATION
```

### `observations.py get`

Fetch a specific observation by ID.

| Arg | Required | Description |
|-----|----------|-------------|
| `observation_id` | Yes | Observation ID (positional) |

```bash
python3 skills/langfuse/scripts/observations.py get obs-abc-123
```

---

## Sessions

### `sessions.py fetch`

List recent sessions with pagination.

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `--age` | Yes | - | Minutes to look back |
| `--page` | No | 1 | Page number |
| `--limit` | No | 50 | Items per page |

### `sessions.py details`

Get detailed session info including all traces.

| Arg | Required | Description |
|-----|----------|-------------|
| `session_id` | Yes | Session ID (positional) |

```bash
python3 skills/langfuse/scripts/sessions.py details session-abc
```

### `sessions.py user`

Get all sessions for a user.

| Arg | Required | Description |
|-----|----------|-------------|
| `user_id` | Yes | User ID (positional) |
| `--age` | Yes | Minutes to look back |

```bash
python3 skills/langfuse/scripts/sessions.py user user_123 --age 1440
```

---

## Exceptions

### `exceptions.py find`

Find exceptions grouped by file, function, or type.

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `--age` | Yes | - | Minutes to look back |
| `--group-by` | No | file | file, function, or type |

Returns: `[{group, count}, ...]` sorted by count (top 50).

```bash
python3 skills/langfuse/scripts/exceptions.py find --age 1440 --group-by type
```

### `exceptions.py file`

Find exceptions in a specific file.

| Arg | Required | Description |
|-----|----------|-------------|
| `filepath` | Yes | File path as in Langfuse metadata (positional) |
| `--age` | Yes | Minutes to look back |

Returns: Top 10 exceptions (newest first) with trace_id, stacktrace, etc.

```bash
python3 skills/langfuse/scripts/exceptions.py file src/ai/chat.py --age 1440
```

### `exceptions.py details`

Get detailed exception info for a trace/span.

| Arg | Required | Description |
|-----|----------|-------------|
| `trace_id` | Yes | Trace ID (positional) |
| `--span-id` | No | Filter to specific span |

### `exceptions.py count`

Get error counts.

| Arg | Required | Description |
|-----|----------|-------------|
| `--age` | Yes | Minutes to look back |

Returns: `{trace_count, observation_count, exception_count}`

---

## Prompts

### `prompts.py list`

List and filter prompts.

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `--name` | No | - | Filter by exact name |
| `--label` | No | - | Filter by label |
| `--tag` | No | - | Filter by tag |
| `--page` | No | 1 | Page number |
| `--limit` | No | 50 | Items per page (max 100) |

### `prompts.py get`

Fetch a specific prompt (resolved).

| Arg | Required | Description |
|-----|----------|-------------|
| `name` | Yes | Prompt name (positional) |
| `--label` | No | Label to fetch (e.g. 'production') |
| `--version` | No | Specific version number |

```bash
python3 skills/langfuse/scripts/prompts.py get chat-system --label production
```

### `prompts.py get-unresolved`

Fetch a prompt without resolving dependencies. Same args as `get`.

### `prompts.py create-text`

Create a new text prompt version.

| Arg | Required | Description |
|-----|----------|-------------|
| `name` | Yes | Prompt name (positional) |
| `--prompt` | Yes | Prompt text content |
| `--labels` | No | Comma-separated labels |
| `--config` | No | JSON config |
| `--tags` | No | Comma-separated tags |
| `--commit-message` | No | Commit message |

```bash
python3 skills/langfuse/scripts/prompts.py create-text greeting \
  --prompt "Hello {{name}}" --labels staging
```

### `prompts.py create-chat`

Create a new chat prompt version.

| Arg | Required | Description |
|-----|----------|-------------|
| `name` | Yes | Prompt name (positional) |
| `--prompt` | Yes | JSON array of {role, content} messages |
| `--labels` | No | Comma-separated labels |
| `--config` | No | JSON config |
| `--tags` | No | Comma-separated tags |
| `--commit-message` | No | Commit message |

```bash
python3 skills/langfuse/scripts/prompts.py create-chat assistant \
  --prompt '[{"role":"system","content":"You are helpful."}]' --labels staging
```

### `prompts.py update-labels`

Update labels for a specific prompt version.

| Arg | Required | Description |
|-----|----------|-------------|
| `name` | Yes | Prompt name (positional) |
| `--version` | Yes | Version number |
| `--labels` | Yes | Comma-separated labels to add |

Labels are merged with existing. Labels are unique across versions.

```bash
python3 skills/langfuse/scripts/prompts.py update-labels greeting --version 3 --labels production
```

---

## Datasets

### `datasets.py list`

List all datasets.

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `--page` | No | 1 | Page number |
| `--limit` | No | 50 | Items per page |

### `datasets.py get`

Get a dataset by name.

| Arg | Required | Description |
|-----|----------|-------------|
| `name` | Yes | Dataset name (positional) |

### `datasets.py list-items`

List items in a dataset.

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `dataset_name` | Yes | - | Dataset name (positional) |
| `--source-trace-id` | No | - | Filter by source trace |
| `--source-observation-id` | No | - | Filter by source observation |
| `--page` | No | 1 | Page number |
| `--limit` | No | 50 | Items per page |

### `datasets.py get-item`

Get a specific dataset item by ID.

| Arg | Required | Description |
|-----|----------|-------------|
| `item_id` | Yes | Item ID (positional) |

### `datasets.py create`

Create a new dataset.

| Arg | Required | Description |
|-----|----------|-------------|
| `name` | Yes | Dataset name (positional) |
| `--description` | No | Description |
| `--metadata` | No | JSON metadata |

### `datasets.py create-item`

Create or upsert a dataset item.

| Arg | Required | Description |
|-----|----------|-------------|
| `dataset_name` | Yes | Dataset name (positional) |
| `--input` | No | JSON input data |
| `--expected-output` | No | JSON expected output |
| `--metadata` | No | JSON metadata |
| `--source-trace-id` | No | Link to source trace |
| `--source-observation-id` | No | Link to source observation |
| `--item-id` | No | Item ID (enables upsert) |
| `--status` | No | ACTIVE or ARCHIVED |

### `datasets.py delete-item`

Delete a dataset item (permanent).

| Arg | Required | Description |
|-----|----------|-------------|
| `item_id` | Yes | Item ID (positional) |

---

## Schema

### `schema.py`

Output Langfuse data schema documentation. No arguments.

```bash
python3 skills/langfuse/scripts/schema.py
```

---

## Response Format

All commands output JSON to stdout:

```json
{
  "data": [...] | {...},
  "metadata": {
    "item_count": 10,
    "page": 1,
    "total": 100
  }
}
```

Errors are printed to stderr and exit with code 1.
