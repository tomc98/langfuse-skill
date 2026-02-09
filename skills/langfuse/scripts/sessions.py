#!/usr/bin/env python3
"""Langfuse sessions: fetch, details, and user session data."""

import argparse
import os
import sys

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


def cmd_fetch(args):
    """List recent sessions with pagination."""
    age = validate_age(args.age)
    from_ts, _ = age_to_timestamps(age)

    params = {
        "page": args.page,
        "limit": args.limit,
        "fromTimestamp": from_ts,
    }

    response = api_request("GET", "/api/public/sessions", params=params)
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


def cmd_details(args):
    """Get detailed session info by ID, including traces."""
    # Fetch session metadata
    session = api_request("GET", f"/api/public/sessions/{args.session_id}")

    # Fetch traces for this session
    trace_response = api_request(
        "GET",
        "/api/public/traces",
        params={
            "sessionId": args.session_id,
            "limit": 100,
        },
    )
    traces = trace_response.get("data", []) if trace_response else []

    # Build session object
    timestamps = []
    user_ids = set()
    for t in traces:
        ts = t.get("timestamp")
        if ts:
            timestamps.append(ts)
        uid = t.get("userId") or t.get("user_id")
        if uid:
            user_ids.add(uid)

    timestamps.sort()

    result = {
        "id": args.session_id,
        "trace_count": len(traces),
        "first_timestamp": timestamps[0] if timestamps else None,
        "last_timestamp": timestamps[-1] if timestamps else None,
        "user_id": list(user_ids)[0] if len(user_ids) == 1 else list(user_ids) if user_ids else None,
        "traces": traces,
    }

    if session:
        for key in ("createdAt", "projectId"):
            if key in session:
                result[key] = session[key]

    format_output(
        result,
        metadata={
            "session_id": args.session_id,
            "found": True,
        },
        compact=get_compact(args),
    )


def cmd_user(args):
    """Get all sessions for a user."""
    age = validate_age(args.age)
    from_ts, _ = age_to_timestamps(age)

    # Fetch traces for user
    trace_response = api_request(
        "GET",
        "/api/public/traces",
        params={
            "userId": args.user_id,
            "fromTimestamp": from_ts,
            "limit": 100,
        },
    )
    traces = trace_response.get("data", []) if trace_response else []

    # Group traces by session_id
    sessions_map = {}
    for t in traces:
        sid = t.get("sessionId") or t.get("session_id") or "no_session"
        if sid not in sessions_map:
            sessions_map[sid] = []
        sessions_map[sid].append(t)

    # Build session objects
    sessions = []
    for sid, session_traces in sessions_map.items():
        timestamps = []
        for t in session_traces:
            ts = t.get("timestamp")
            if ts:
                timestamps.append(ts)
        timestamps.sort()

        sessions.append(
            {
                "id": sid,
                "trace_count": len(session_traces),
                "first_timestamp": timestamps[0] if timestamps else None,
                "last_timestamp": timestamps[-1] if timestamps else None,
                "traces": session_traces,
            }
        )

    # Sort by most recent
    sessions.sort(key=lambda s: datetime_sort_key(s.get("last_timestamp")), reverse=True)

    format_output(
        sessions,
        metadata={
            "user_id": args.user_id,
            "session_count": len(sessions),
        },
        compact=get_compact(args),
    )


def main():
    parser = argparse.ArgumentParser(description="Langfuse sessions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch
    fetch_p = subparsers.add_parser("fetch", help="List recent sessions")
    fetch_p.add_argument("--age", required=True, help="Minutes to look back (max 10080)")
    fetch_p.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    fetch_p.add_argument("--limit", type=int, default=50, help="Items per page (default: 50)")
    add_common_args(fetch_p)
    fetch_p.set_defaults(func=cmd_fetch)

    # details
    details_p = subparsers.add_parser("details", help="Get session details with traces")
    details_p.add_argument("session_id", help="Session ID")
    add_common_args(details_p)
    details_p.set_defaults(func=cmd_details)

    # user
    user_p = subparsers.add_parser("user", help="Get all sessions for a user")
    user_p.add_argument("user_id", help="User ID")
    user_p.add_argument("--age", required=True, help="Minutes to look back (max 10080)")
    add_common_args(user_p)
    user_p.set_defaults(func=cmd_user)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
