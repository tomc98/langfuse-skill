#!/usr/bin/env python3
"""Langfuse exceptions: find, analyze, and count exceptions from span observations."""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from _client import (
    add_common_args,
    age_to_timestamps,
    api_request,
    datetime_sort_key,
    format_output,
    get_compact,
    validate_age,
)


def _fetch_span_observations(from_ts, to_ts):
    """Fetch SPAN observations that may contain exceptions."""
    response = api_request(
        "GET",
        "/api/public/observations",
        params={
            "type": "SPAN",
            "fromStartTime": from_ts,
            "toStartTime": to_ts,
            "limit": 100,
        },
    )
    return response.get("data", []) if response else []


def _extract_exceptions_from_observation(observation):
    """Extract exception events from an observation. Returns list of exception info dicts."""
    events = observation.get("events", [])
    if not events:
        return []

    exceptions = []
    metadata = observation.get("metadata") or {}

    for event in events:
        attrs = event.get("attributes", {})
        if not attrs.get("exception.type"):
            continue

        exceptions.append(
            {
                "observation_id": observation.get("id", "unknown"),
                "trace_id": observation.get("traceId") or observation.get("trace_id", "unknown"),
                "observation_name": observation.get("name", "unknown"),
                "observation_type": observation.get("type", "unknown"),
                "timestamp": observation.get("startTime") or observation.get("start_time", "unknown"),
                "exception_type": attrs.get("exception.type", "unknown"),
                "exception_message": attrs.get("exception.message", ""),
                "exception_stacktrace": attrs.get("exception.stacktrace", ""),
                "filepath": metadata.get("code.filepath", "unknown"),
                "function": metadata.get("code.function", "unknown"),
                "line_number": metadata.get("code.lineno", "unknown"),
                "event_id": event.get("id", "unknown"),
                "event_name": event.get("name", "unknown"),
            }
        )

    return exceptions


def cmd_find(args):
    """Get exception counts grouped by file, function, or type."""
    age = validate_age(args.age)
    from_ts, to_ts = age_to_timestamps(age)

    observations = _fetch_span_observations(from_ts, to_ts)
    groups = Counter()

    for obs in observations:
        events = obs.get("events", [])
        if not events:
            continue

        metadata = obs.get("metadata") or {}
        for event in events:
            attrs = event.get("attributes", {})
            if not attrs.get("exception.type"):
                continue

            if args.group_by == "file":
                key = metadata.get("code.filepath", "unknown_file")
            elif args.group_by == "function":
                key = metadata.get("code.function", "unknown_function")
            elif args.group_by == "type":
                key = attrs.get("exception.type", "unknown_exception")
            else:
                key = "unknown"

            groups[key] += 1

    results = [{"group": group, "count": count} for group, count in groups.most_common(50)]
    format_output(results, metadata={"item_count": len(results)}, compact=True)


def cmd_file(args):
    """Find exceptions in a specific file."""
    age = validate_age(args.age)
    from_ts, to_ts = age_to_timestamps(age)

    observations = _fetch_span_observations(from_ts, to_ts)
    exceptions = []

    for obs in observations:
        metadata = obs.get("metadata") or {}
        if metadata.get("code.filepath") != args.filepath:
            continue

        for exc in _extract_exceptions_from_observation(obs):
            exceptions.append(exc)

    exceptions.sort(key=lambda x: datetime_sort_key(x.get("timestamp")), reverse=True)
    top = exceptions[:10]

    format_output(
        top,
        metadata={
            "file_path": args.filepath,
            "item_count": len(top),
            "total_exceptions": len(exceptions),
        },
        compact=get_compact(args),
    )


def cmd_details(args):
    """Get detailed exception info for a trace/span."""
    # Fetch observations for this trace
    response = api_request(
        "GET",
        "/api/public/observations",
        params={
            "traceId": args.trace_id,
            "limit": 100,
        },
    )
    observations = response.get("data", []) if response else []

    if args.span_id:
        observations = [o for o in observations if o.get("id") == args.span_id]

    exceptions = []
    for obs in observations:
        exceptions.extend(_extract_exceptions_from_observation(obs))

    exceptions.sort(key=lambda x: datetime_sort_key(x.get("timestamp")), reverse=True)

    format_output(
        exceptions,
        metadata={
            "trace_id": args.trace_id,
            "span_id": args.span_id,
            "item_count": len(exceptions),
        },
        compact=get_compact(args),
    )


def cmd_count(args):
    """Get error counts for the last N minutes."""
    age = validate_age(args.age)
    from_ts, to_ts = age_to_timestamps(age)

    observations = _fetch_span_observations(from_ts, to_ts)

    trace_ids = set()
    obs_with_exceptions = 0
    total_exceptions = 0

    for obs in observations:
        events = obs.get("events", [])
        if not events:
            continue

        exc_count = sum(1 for event in events if event.get("attributes", {}).get("exception.type"))
        if exc_count == 0:
            continue

        obs_with_exceptions += 1
        total_exceptions += exc_count

        trace_id = obs.get("traceId") or obs.get("trace_id")
        if trace_id:
            trace_ids.add(trace_id)

    format_output(
        {
            "age_minutes": age,
            "from_timestamp": from_ts,
            "to_timestamp": to_ts,
            "trace_count": len(trace_ids),
            "observation_count": obs_with_exceptions,
            "exception_count": total_exceptions,
        },
        compact=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Langfuse exceptions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # find
    find_p = subparsers.add_parser("find", help="Find exceptions grouped by file/function/type")
    find_p.add_argument("--age", required=True, help="Minutes to look back (max 10080)")
    find_p.add_argument("--group-by", choices=["file", "function", "type"], default="file", help="How to group exceptions (default: file)")
    find_p.set_defaults(func=cmd_find)

    # file
    file_p = subparsers.add_parser("file", help="Find exceptions in a specific file")
    file_p.add_argument("filepath", help="File path as recorded in Langfuse metadata")
    file_p.add_argument("--age", required=True, help="Minutes to look back (max 10080)")
    add_common_args(file_p)
    file_p.set_defaults(func=cmd_file)

    # details
    details_p = subparsers.add_parser("details", help="Get exception details for a trace")
    details_p.add_argument("trace_id", help="Trace ID to analyze")
    details_p.add_argument("--span-id", help="Optional span ID to filter by")
    add_common_args(details_p)
    details_p.set_defaults(func=cmd_details)

    # count
    count_p = subparsers.add_parser("count", help="Get error counts")
    count_p.add_argument("--age", required=True, help="Minutes to look back (max 10080)")
    count_p.set_defaults(func=cmd_count)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
