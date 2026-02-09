#!/usr/bin/env python3
"""Langfuse prompts: list, get, create, and manage prompt versions."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _client import (
    add_common_args,
    api_request,
    format_output,
    get_compact,
    parse_json_arg,
    parse_list_arg,
)


def cmd_list(args):
    """List and filter prompts in the project."""
    params = {
        "page": args.page,
        "limit": args.limit,
    }
    if args.name:
        params["name"] = args.name
    if args.label:
        params["label"] = args.label
    if args.tag:
        params["tag"] = args.tag

    response = api_request("GET", "/api/public/v2/prompts", params=params)
    items = response.get("data", []) if response else []
    meta = response.get("meta", {}) if response else {}

    prompt_list = []
    for p in items:
        prompt_list.append(
            {
                "name": p.get("name"),
                "type": p.get("type"),
                "versions": p.get("versions", []),
                "labels": p.get("labels", []),
                "tags": p.get("tags", []),
                "lastUpdatedAt": p.get("lastUpdatedAt") or p.get("updatedAt"),
                "lastConfig": p.get("lastConfig") or p.get("config"),
            }
        )

    format_output(
        prompt_list,
        metadata={
            "page": args.page,
            "limit": args.limit,
            "item_count": len(prompt_list),
            "total": meta.get("totalItems") or meta.get("total"),
        },
        compact=True,
    )


def cmd_get(args):
    """Fetch a specific prompt with resolved dependencies."""
    if args.label and args.version:
        print("Error: Cannot specify both --label and --version", file=sys.stderr)
        sys.exit(1)

    params = {}
    if args.label:
        params["label"] = args.label
    if args.version:
        params["version"] = args.version

    response = api_request("GET", f"/api/public/v2/prompts/{args.name}", params=params)

    if response is None:
        print(f"Error: Prompt '{args.name}' not found", file=sys.stderr)
        sys.exit(1)

    # Determine type
    prompt_type = response.get("type", "unknown")
    if isinstance(prompt_type, str):
        prompt_type = prompt_type.lower()

    prompt_content = response.get("prompt")
    if prompt_content is None:
        prompt_content = response.get("messages")

    result = {
        "name": response.get("name", args.name),
        "version": response.get("version"),
        "type": prompt_type,
        "prompt": prompt_content,
        "labels": response.get("labels", []),
        "tags": response.get("tags", []),
        "config": response.get("config", {}),
    }
    if response.get("id"):
        result["id"] = response["id"]

    format_output(result, metadata={"found": True}, compact=get_compact(args))


def cmd_get_unresolved(args):
    """Fetch a prompt WITHOUT resolving dependencies."""
    if args.label and args.version:
        print("Error: Cannot specify both --label and --version", file=sys.stderr)
        sys.exit(1)

    params = {"resolve": "false"}
    if args.label:
        params["label"] = args.label
    if args.version:
        params["version"] = args.version

    response = api_request("GET", f"/api/public/v2/prompts/{args.name}", params=params)

    if response is None:
        print(f"Error: Prompt '{args.name}' not found", file=sys.stderr)
        sys.exit(1)

    format_output(response, metadata={"found": True, "resolved": False}, compact=get_compact(args))


def cmd_create_text(args):
    """Create a new text prompt version."""
    labels = parse_list_arg(args.labels)
    tags = parse_list_arg(args.tags)
    config = parse_json_arg(args.config)

    body = {
        "name": args.name,
        "prompt": args.prompt,
        "type": "text",
        "labels": labels or [],
        "config": config or {},
    }
    if tags:
        body["tags"] = tags
    if args.commit_message:
        body["commitMessage"] = args.commit_message

    response = api_request("POST", "/api/public/v2/prompts", body=body)

    format_output(response, metadata={"created": True}, compact=get_compact(args))


def cmd_create_chat(args):
    """Create a new chat prompt version."""
    prompt = parse_json_arg(args.prompt)
    if not isinstance(prompt, list):
        print("Error: Chat prompt must be a JSON array of {role, content} objects", file=sys.stderr)
        sys.exit(1)

    labels = parse_list_arg(args.labels)
    tags = parse_list_arg(args.tags)
    config = parse_json_arg(args.config)

    body = {
        "name": args.name,
        "prompt": prompt,
        "type": "chat",
        "labels": labels or [],
        "config": config or {},
    }
    if tags:
        body["tags"] = tags
    if args.commit_message:
        body["commitMessage"] = args.commit_message

    response = api_request("POST", "/api/public/v2/prompts", body=body)

    format_output(response, metadata={"created": True}, compact=get_compact(args))


def cmd_update_labels(args):
    """Update labels for a specific prompt version."""
    labels = parse_list_arg(args.labels)
    if not labels:
        print("Error: --labels is required and must not be empty", file=sys.stderr)
        sys.exit(1)

    # Fetch current prompt to get existing labels
    current = api_request("GET", f"/api/public/v2/prompts/{args.name}", params={"version": args.version})
    existing_labels = current.get("labels", []) if current else []

    # Merge: new labels first, then existing (dedup preserving order)
    merged = list(dict.fromkeys(labels + existing_labels))

    response = api_request(
        "PATCH",
        f"/api/public/v2/prompts/{args.name}/versions/{args.version}",
        body={"labels": merged},
    )

    format_output(response, metadata={"updated": True}, compact=get_compact(args))


def main():
    parser = argparse.ArgumentParser(description="Langfuse prompts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_p = subparsers.add_parser("list", help="List prompts")
    list_p.add_argument("--name", help="Filter by exact prompt name")
    list_p.add_argument("--label", help="Filter by label")
    list_p.add_argument("--tag", help="Filter by tag")
    list_p.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    list_p.add_argument("--limit", type=int, default=50, help="Items per page (max 100)")
    list_p.set_defaults(func=cmd_list)

    # get
    get_p = subparsers.add_parser("get", help="Get a specific prompt")
    get_p.add_argument("name", help="Prompt name")
    get_p.add_argument("--label", help="Label to fetch (e.g. 'production')")
    get_p.add_argument("--version", type=int, help="Specific version number")
    add_common_args(get_p)
    get_p.set_defaults(func=cmd_get)

    # get-unresolved
    unresolved_p = subparsers.add_parser("get-unresolved", help="Get prompt without resolving deps")
    unresolved_p.add_argument("name", help="Prompt name")
    unresolved_p.add_argument("--label", help="Label to fetch")
    unresolved_p.add_argument("--version", type=int, help="Specific version number")
    add_common_args(unresolved_p)
    unresolved_p.set_defaults(func=cmd_get_unresolved)

    # create-text
    create_text_p = subparsers.add_parser("create-text", help="Create a text prompt version")
    create_text_p.add_argument("name", help="Prompt name")
    create_text_p.add_argument("--prompt", required=True, help="Prompt text content")
    create_text_p.add_argument("--labels", help="Comma-separated labels")
    create_text_p.add_argument("--config", help='JSON config (e.g. \'{"model": "gpt-4"}\')')
    create_text_p.add_argument("--tags", help="Comma-separated tags")
    create_text_p.add_argument("--commit-message", help="Commit message")
    add_common_args(create_text_p)
    create_text_p.set_defaults(func=cmd_create_text)

    # create-chat
    create_chat_p = subparsers.add_parser("create-chat", help="Create a chat prompt version")
    create_chat_p.add_argument("name", help="Prompt name")
    create_chat_p.add_argument("--prompt", required=True, help='JSON array of messages: [{"role":"system","content":"..."}]')
    create_chat_p.add_argument("--labels", help="Comma-separated labels")
    create_chat_p.add_argument("--config", help="JSON config")
    create_chat_p.add_argument("--tags", help="Comma-separated tags")
    create_chat_p.add_argument("--commit-message", help="Commit message")
    add_common_args(create_chat_p)
    create_chat_p.set_defaults(func=cmd_create_chat)

    # update-labels
    update_labels_p = subparsers.add_parser("update-labels", help="Update labels for a prompt version")
    update_labels_p.add_argument("name", help="Prompt name")
    update_labels_p.add_argument("--version", type=int, required=True, help="Version number to update")
    update_labels_p.add_argument("--labels", required=True, help="Comma-separated labels to add")
    add_common_args(update_labels_p)
    update_labels_p.set_defaults(func=cmd_update_labels)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
