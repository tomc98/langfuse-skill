"""Shared client for Langfuse REST API scripts.

Provides auth, HTTP requests, truncation, and output formatting.
No external dependencies — stdlib only.
"""

import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

# Constants
HOUR = 60
DAY = 24 * HOUR
MAX_AGE_MINUTES = 7 * DAY
MAX_FIELD_LENGTH = 500
MAX_RESPONSE_SIZE = 20000
TRUNCATE_SUFFIX = "..."

LARGE_FIELDS = {
    "input",
    "output",
    "content",
    "prompt",
    "completion",
    "system_prompt",
    "user_prompt",
    "message",
    "exception.stacktrace",
    "exception.message",
    "stacktrace",
    "llm.prompts",
    "llm.prompt",
    "llm.prompts.system",
    "llm.prompts.user",
    "llm.prompt.system",
    "llm.prompt.user",
    "langfuseprompt",
    "prompt.content",
    "prompt.messages",
    "prompt.system",
    "metadata.langfuseprompt",
    "metadata.system_prompt",
    "metadata.prompt",
    "attributes.llm.prompts",
    "attributes.llm.prompt",
    "attributes.system_prompt",
    "attributes.prompt",
    "attributes.input",
    "attributes.output",
    "expected_output",
    "expectedoutput",
    "input_schema",
    "expected_output_schema",
}

ESSENTIAL_FIELDS = {
    "id",
    "trace_id",
    "observation_id",
    "parent_observation_id",
    "name",
    "type",
    "timestamp",
    "start_time",
    "end_time",
    "level",
    "status_message",
    "user_id",
    "session_id",
}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def get_credentials():
    """Get (public_key, secret_key, host) from env or ~/.claude/settings.json."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")

    if public_key and secret_key:
        return public_key, secret_key, host or "https://cloud.langfuse.com"

    settings_path = os.path.expanduser("~/.claude/settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            settings = json.load(f)
            env = settings.get("env", {})
            public_key = public_key or env.get("LANGFUSE_PUBLIC_KEY")
            secret_key = secret_key or env.get("LANGFUSE_SECRET_KEY")
            host = host or env.get("LANGFUSE_HOST")

    if not public_key or not secret_key:
        print("Error: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY not set", file=sys.stderr)
        print("Add to ~/.claude/settings.json:", file=sys.stderr)
        print(
            '  {"env": {"LANGFUSE_PUBLIC_KEY": "pk-...", "LANGFUSE_SECRET_KEY": "sk-...", "LANGFUSE_HOST": "https://cloud.langfuse.com"}}',
            file=sys.stderr,
        )
        sys.exit(1)

    return public_key, secret_key, host or "https://cloud.langfuse.com"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def api_request(method, path, params=None, body=None, timeout=30):
    """Make an authenticated request to the Langfuse REST API.

    Returns parsed JSON response dict, or None for 204/empty responses.
    Exits with error message on failure.
    """
    public_key, secret_key, host = get_credentials()
    url = f"{host.rstrip('/')}{path}"

    if params:
        filtered = {k: v for k, v in params.items() if v is not None}
        if filtered:
            url += "?" + urllib.parse.urlencode(filtered, doseq=True)

    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "Accept": "application/json",
    }

    data = None
    if body is not None:
        data = json.dumps(body, default=str).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read().decode("utf-8")
            if content:
                return json.loads(content)
            return None
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_json = json.loads(error_body)
            msg = error_json.get("message") or error_json.get("error") or error_body
        except (json.JSONDecodeError, ValueError):
            msg = error_body
        print(f"Error {e.code}: {msg}", file=sys.stderr)
        print(f"  {method} {path}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        print(f"  Host: {host}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def truncate_large_strings(
    obj, max_length=MAX_FIELD_LENGTH, max_response_size=MAX_RESPONSE_SIZE, path="", current_size=0, truncation_level=0
):
    """Recursively truncate large string values while preserving essential fields.

    Returns (processed_obj, size).
    """
    adjusted_max = max_length
    if truncation_level == 1:
        adjusted_max = max(50, max_length // 2)
    elif truncation_level >= 2:
        adjusted_max = max(20, max_length // 5)

    if current_size > max_response_size * 1.5:
        return "[TRUNCATED]", len("[TRUNCATED]")

    if isinstance(obj, dict):
        result = {}
        result_size = 2

        # Essential fields first
        for key in list(obj.keys()):
            if key in ESSENTIAL_FIELDS:
                val, vs = truncate_large_strings(
                    obj[key],
                    adjusted_max,
                    max_response_size,
                    f"{path}.{key}" if path else key,
                    current_size + result_size,
                    truncation_level,
                )
                result[key] = val
                result_size += len(str(key)) + 2 + vs

        # Large fields
        if truncation_level < 2:
            for key in list(obj.keys()):
                if key in result:
                    continue
                lower_key = key.lower()
                if lower_key in LARGE_FIELDS or any(f in lower_key for f in LARGE_FIELDS):
                    value = obj[key]
                    if isinstance(value, str) and len(value) > adjusted_max:
                        if "stack" in lower_key and "\n" in value:
                            lines = value.split("\n")
                            if len(lines) > 6:
                                truncated = "\n".join(lines[:3] + ["..."] + lines[-3:])
                                result[key] = truncated
                                result_size += len(str(key)) + 2 + len(truncated)
                            else:
                                result[key] = value
                                result_size += len(str(key)) + 2 + len(value)
                        else:
                            result[key] = value[:adjusted_max] + TRUNCATE_SUFFIX
                            result_size += len(str(key)) + 2 + adjusted_max + len(TRUNCATE_SUFFIX)
                    else:
                        val, vs = truncate_large_strings(
                            value,
                            adjusted_max,
                            max_response_size,
                            f"{path}.{key}" if path else key,
                            current_size + result_size,
                            truncation_level,
                        )
                        result[key] = val
                        result_size += len(str(key)) + 2 + vs

        # Remaining fields
        remaining = [k for k in obj if k not in result]
        if truncation_level >= 2 and remaining:
            result["_note"] = f"{len(remaining)} non-essential fields omitted"
            result_size += len("_note") + 2 + len(result["_note"])
        else:
            for key in remaining:
                if current_size + result_size > max_response_size * 0.9:
                    next_level = min(2, truncation_level + 1)
                    if next_level > truncation_level:
                        result["_truncation_note"] = "Response truncated due to size constraints"
                        result_size += len("_truncation_note") + 2 + len(result["_truncation_note"])

                level = min(2, truncation_level + (1 if current_size + result_size > max_response_size * 0.7 else 0))
                val, vs = truncate_large_strings(
                    obj[key], adjusted_max, max_response_size, f"{path}.{key}" if path else key, current_size + result_size, level
                )
                result[key] = val
                result_size += len(str(key)) + 2 + vs

        return result, result_size

    elif isinstance(obj, list):
        if not obj:
            return [], 2

        result = []
        result_size = 2

        sample_item, sample_size = truncate_large_strings(
            obj[0], adjusted_max, max_response_size, f"{path}[0]", current_size + result_size, truncation_level
        )

        estimated_total = sample_size * len(obj)
        target_level = truncation_level
        if estimated_total > max_response_size * 0.8:
            target_level = min(2, truncation_level + 1)

        if target_level == 2 and estimated_total > max_response_size:
            max_items = max(5, int(max_response_size * 0.8 / (sample_size or 100)))
        else:
            max_items = len(obj)

        for i, item in enumerate(obj):
            if i >= max_items:
                result.append({"_note": f"List truncated, {len(obj) - i} of {len(obj)} items omitted"})
                result_size += 50
                break

            item_level = target_level
            if current_size + result_size > max_response_size * 0.8:
                item_level = 2

            val, vs = truncate_large_strings(item, adjusted_max, max_response_size, f"{path}[{i}]", current_size + result_size, item_level)
            result.append(val)
            result_size += vs + (1 if i < len(obj) - 1 else 0)

        return result, result_size

    elif isinstance(obj, str):
        if len(obj) <= adjusted_max:
            return obj, len(obj)

        if truncation_level == 0 and ("stacktrace" in path.lower() or "stack" in path.lower()) and "\n" in obj:
            lines = obj.split("\n")
            if len(lines) > 6:
                truncated = "\n".join(lines[:3] + ["..."] + lines[-3:])
                return truncated, len(truncated)

        truncated = obj[:adjusted_max] + TRUNCATE_SUFFIX
        return truncated, len(truncated)

    else:
        return obj, len(str(obj))


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_output(data, metadata=None, compact=True):
    """Format and print output as JSON to stdout."""
    if compact:
        data, _ = truncate_large_strings(data, truncation_level=0)

    output = {"data": data}
    if metadata:
        output["metadata"] = metadata

    print(json.dumps(output, indent=2, default=str))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def age_to_timestamps(age_minutes):
    """Convert age in minutes to (from_iso, to_iso) timestamp strings."""
    now = datetime.now(timezone.utc)
    from_dt = now - timedelta(minutes=age_minutes)
    return from_dt.isoformat(), now.isoformat()


def validate_age(value):
    """Validate age parameter (1-10080). Returns int or exits."""
    try:
        age = int(value)
    except (TypeError, ValueError):
        print(f"Error: age must be a positive integer, got '{value}'", file=sys.stderr)
        sys.exit(1)

    if age < 1 or age > MAX_AGE_MINUTES:
        print(f"Error: age must be 1-{MAX_AGE_MINUTES} minutes, got {age}", file=sys.stderr)
        sys.exit(1)

    return age


def parse_json_arg(value):
    """Parse a JSON string argument into a dict/list. Exits on error."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)


def parse_list_arg(value):
    """Parse a comma-separated string into a list of strings."""
    if value is None:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


def parse_datetime_for_sort(value):
    """Parse ISO8601 string into datetime for sorting. Returns None on failure."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


_MIN_SORT_DATETIME = datetime.min.replace(tzinfo=timezone.utc)


def datetime_sort_key(value):
    """Return a datetime suitable for key= sorting with None-safe fallback."""
    return parse_datetime_for_sort(value) or _MIN_SORT_DATETIME


def add_common_args(parser):
    """Add --no-truncate flag to an argparse parser."""
    parser.add_argument("--no-truncate", action="store_true", help="Output full data without truncation")


def get_compact(args):
    """Return True if output should be compact (truncated)."""
    return not getattr(args, "no_truncate", False)
