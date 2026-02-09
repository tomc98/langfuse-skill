# Langfuse Skill Setup

## Credentials

Get API keys from https://cloud.langfuse.com → Settings → API Keys.

Add to `~/.claude/settings.json`:

```json
{
  "env": {
    "LANGFUSE_PUBLIC_KEY": "pk-...",
    "LANGFUSE_SECRET_KEY": "sk-...",
    "LANGFUSE_HOST": "https://cloud.langfuse.com"
  }
}
```

Self-hosted: use your instance URL for `LANGFUSE_HOST`.

Scripts check environment variables first (auto-populated from settings.json by Claude Code), then read settings.json directly as fallback.

## Test

```bash
python3 skills/langfuse/scripts/traces.py fetch --age 60
```

---

## Troubleshooting

### Authentication Errors

- Public key must start with `pk-`
- Secret key must start with `sk-`
- Host must match your Langfuse instance (cloud vs self-hosted)

### Empty Results

- Check `--age` parameter (minutes, not hours/days)
- Verify filters match your data
- Try `traces.py fetch --age 1440` with no filters to confirm data exists
- Data older than 7 days (10080 minutes) cannot be retrieved

### Connection Errors

- Verify `LANGFUSE_HOST` is correct and reachable
- Check firewall/VPN if self-hosted

---

## Date-to-Minutes Conversion

| Time Range | Minutes |
|------------|---------|
| 1 hour | 60 |
| 6 hours | 360 |
| 12 hours | 720 |
| 1 day | 1440 |
| 2 days | 2880 |
| 3 days | 4320 |
| 7 days | 10080 (max) |

---

## Glossary

| Term | Definition |
|------|------------|
| **Trace** | Top-level container for a user interaction/request |
| **Observation** | Span, generation, or event inside a trace |
| **Generation** | LLM API call with input/output/tokens/latency |
| **Span** | Timed operation (function call, API request) |
| **Event** | Discrete log entry or exception |
| **Session** | Group of traces from the same user session |

---

## Security Notes

- Never commit credentials to version control
- Rotate keys if leaked
- Use environment variable injection in CI/CD rather than hardcoded values
