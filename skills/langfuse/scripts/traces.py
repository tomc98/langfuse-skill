#!/usr/bin/env python3
"""Langfuse traces: fetch and get trace data."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _client import (
    add_common_args,
    age_to_timestamps,
    api_request,
    format_output,
    get_compact,
    parse_json_arg,
    parse_list_arg,
    validate_age,
)


def cmd_fetch(args):
    """Search and filter traces with pagination."""
    age = validate_age(args.age)
    from_ts, to_ts = age_to_timestamps(age)

    params = {
        "page": args.page,
        "limit": args.limit,
        "fromTimestamp": from_ts,
    }
    if args.name:
        params["name"] = args.name
    if args.user_id:
        params["userId"] = args.user_id
    if args.session_id:
        params["sessionId"] = args.session_id
    if args.tags:
        tags = parse_list_arg(args.tags)
        if tags:
            params["tags"] = tags

    response = api_request("GET", "/api/public/traces", params=params)
    items = response.get("data", []) if response else []
    meta = response.get("meta", {}) if response else {}

    # Client-side metadata filtering
    if args.metadata:
        metadata_filter = parse_json_arg(args.metadata)
        if metadata_filter:
            items = [t for t in items if _metadata_matches(t, metadata_filter)]

    format_output(
        items,
        metadata={
            "item_count": len(items),
            "page": args.page,
            "total": meta.get("totalItems") or meta.get("total"),
            "next_page": args.page + 1 if len(items) == args.limit else None,
        },
        compact=get_compact(args),
    )


def cmd_get(args):
    """Fetch a specific trace by ID."""
    trace = api_request("GET", f"/api/public/traces/{args.trace_id}")

    if trace is None:
        print(f"Error: Trace '{args.trace_id}' not found", file=sys.stderr)
        sys.exit(1)

    format_output(trace, metadata={"trace_id": args.trace_id}, compact=get_compact(args))


def _metadata_matches(item, metadata_filter):
    """Check if item metadata matches all filter key-value pairs."""
    metadata = item.get("metadata") or {}
    return all(metadata.get(k) == v for k, v in metadata_filter.items())


def main():
    parser = argparse.ArgumentParser(description="Langfuse traces")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch
    fetch_p = subparsers.add_parser("fetch", help="Search and filter traces")
    fetch_p.add_argument("--age", required=True, help="Minutes to look back (max 10080)")
    fetch_p.add_argument("--name", help="Filter by trace name")
    fetch_p.add_argument("--user-id", help="Filter by user ID")
    fetch_p.add_argument("--session-id", help="Filter by session ID")
    fetch_p.add_argument("--metadata", help='JSON metadata filter (e.g. \'{"key": "value"}\')')
    fetch_p.add_argument("--tags", help="Comma-separated tags")
    fetch_p.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    fetch_p.add_argument("--limit", type=int, default=50, help="Items per page (default: 50)")
    add_common_args(fetch_p)
    fetch_p.set_defaults(func=cmd_fetch)

    # get
    get_p = subparsers.add_parser("get", help="Fetch a specific trace by ID")
    get_p.add_argument("trace_id", help="Trace ID to fetch")
    add_common_args(get_p)
    get_p.set_defaults(func=cmd_get)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
