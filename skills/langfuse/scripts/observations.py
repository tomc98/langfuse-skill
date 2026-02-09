#!/usr/bin/env python3
"""Langfuse observations: fetch and get observation data."""

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
    validate_age,
)


def cmd_fetch(args):
    """Search and filter observations with pagination."""
    age = validate_age(args.age)
    from_ts, to_ts = age_to_timestamps(age)

    params = {
        "page": args.page,
        "limit": args.limit,
        "fromStartTime": from_ts,
    }
    if args.type:
        params["type"] = args.type
    if args.name:
        params["name"] = args.name
    if args.user_id:
        params["userId"] = args.user_id
    if args.trace_id:
        params["traceId"] = args.trace_id
    if args.parent_observation_id:
        params["parentObservationId"] = args.parent_observation_id

    response = api_request("GET", "/api/public/observations", params=params)
    items = response.get("data", []) if response else []
    meta = response.get("meta", {}) if response else {}

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
    """Fetch a specific observation by ID."""
    obs = api_request("GET", f"/api/public/observations/{args.observation_id}")

    if obs is None:
        print(f"Error: Observation '{args.observation_id}' not found", file=sys.stderr)
        sys.exit(1)

    format_output(obs, metadata={"observation_id": args.observation_id}, compact=get_compact(args))


def main():
    parser = argparse.ArgumentParser(description="Langfuse observations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch
    fetch_p = subparsers.add_parser("fetch", help="Search and filter observations")
    fetch_p.add_argument("--age", required=True, help="Minutes to look back (max 10080)")
    fetch_p.add_argument("--type", choices=["SPAN", "GENERATION", "EVENT"], help="Filter by type")
    fetch_p.add_argument("--name", help="Filter by name")
    fetch_p.add_argument("--user-id", help="Filter by user ID")
    fetch_p.add_argument("--trace-id", help="Filter by trace ID")
    fetch_p.add_argument("--parent-observation-id", help="Filter by parent observation ID")
    fetch_p.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    fetch_p.add_argument("--limit", type=int, default=50, help="Items per page (default: 50)")
    add_common_args(fetch_p)
    fetch_p.set_defaults(func=cmd_fetch)

    # get
    get_p = subparsers.add_parser("get", help="Fetch a specific observation by ID")
    get_p.add_argument("observation_id", help="Observation ID to fetch")
    add_common_args(get_p)
    get_p.set_defaults(func=cmd_get)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
