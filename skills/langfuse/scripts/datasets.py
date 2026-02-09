#!/usr/bin/env python3
"""Langfuse datasets: list, get, create, and manage dataset items."""

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
)


def cmd_list(args):
    """List all datasets with pagination."""
    params = {
        "page": args.page,
        "limit": args.limit,
    }

    response = api_request("GET", "/api/public/v2/datasets", params=params)
    items = response.get("data", []) if response else []
    meta = response.get("meta", {}) if response else {}

    dataset_list = []
    for d in items:
        dataset_list.append(
            {
                "id": d.get("id"),
                "name": d.get("name"),
                "description": d.get("description"),
                "metadata": d.get("metadata"),
                "projectId": d.get("projectId") or d.get("project_id"),
                "createdAt": d.get("createdAt") or d.get("created_at"),
                "updatedAt": d.get("updatedAt") or d.get("updated_at"),
            }
        )

    format_output(
        dataset_list,
        metadata={
            "page": args.page,
            "limit": args.limit,
            "item_count": len(dataset_list),
            "total": meta.get("totalItems") or meta.get("total"),
        },
        compact=True,
    )


def cmd_get(args):
    """Get a specific dataset by name."""
    response = api_request("GET", f"/api/public/v2/datasets/{args.name}")

    if response is None:
        print(f"Error: Dataset '{args.name}' not found", file=sys.stderr)
        sys.exit(1)

    format_output(response, metadata={"name": args.name}, compact=get_compact(args))


def cmd_list_items(args):
    """List items in a dataset with pagination."""
    params = {
        "datasetName": args.dataset_name,
        "page": args.page,
        "limit": args.limit,
    }
    if args.source_trace_id:
        params["sourceTraceId"] = args.source_trace_id
    if args.source_observation_id:
        params["sourceObservationId"] = args.source_observation_id

    response = api_request("GET", "/api/public/v2/dataset-items", params=params)
    items = response.get("data", []) if response else []
    meta = response.get("meta", {}) if response else {}

    format_output(
        items,
        metadata={
            "dataset_name": args.dataset_name,
            "page": args.page,
            "limit": args.limit,
            "item_count": len(items),
            "total": meta.get("totalItems") or meta.get("total"),
        },
        compact=get_compact(args),
    )


def cmd_get_item(args):
    """Get a specific dataset item by ID."""
    response = api_request("GET", f"/api/public/v2/dataset-items/{args.item_id}")

    if response is None:
        print(f"Error: Dataset item '{args.item_id}' not found", file=sys.stderr)
        sys.exit(1)

    format_output(response, metadata={"item_id": args.item_id}, compact=get_compact(args))


def cmd_create(args):
    """Create a new dataset."""
    metadata = parse_json_arg(args.metadata)

    body = {"name": args.name}
    if args.description:
        body["description"] = args.description
    if metadata:
        body["metadata"] = metadata

    response = api_request("POST", "/api/public/v2/datasets", body=body)

    format_output(response, metadata={"created": True, "name": args.name}, compact=get_compact(args))


def cmd_create_item(args):
    """Create or upsert a dataset item."""
    input_data = parse_json_arg(args.input)
    expected_output = parse_json_arg(args.expected_output)
    metadata = parse_json_arg(args.metadata)

    body = {"datasetName": args.dataset_name}
    if input_data is not None:
        body["input"] = input_data
    if expected_output is not None:
        body["expectedOutput"] = expected_output
    if metadata:
        body["metadata"] = metadata
    if args.source_trace_id:
        body["sourceTraceId"] = args.source_trace_id
    if args.source_observation_id:
        body["sourceObservationId"] = args.source_observation_id
    if args.item_id:
        body["id"] = args.item_id
    if args.status:
        body["status"] = args.status

    response = api_request("POST", "/api/public/v2/dataset-items", body=body)

    format_output(
        response,
        metadata={
            "created": True,
            "dataset_name": args.dataset_name,
            "item_id": response.get("id") if response else None,
        },
        compact=get_compact(args),
    )


def cmd_delete_item(args):
    """Delete a dataset item."""
    response = api_request("DELETE", f"/api/public/v2/dataset-items/{args.item_id}")

    format_output(
        response or {},
        metadata={
            "deleted": True,
            "item_id": args.item_id,
        },
        compact=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Langfuse datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_p = subparsers.add_parser("list", help="List all datasets")
    list_p.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    list_p.add_argument("--limit", type=int, default=50, help="Items per page (default: 50)")
    list_p.set_defaults(func=cmd_list)

    # get
    get_p = subparsers.add_parser("get", help="Get a dataset by name")
    get_p.add_argument("name", help="Dataset name")
    add_common_args(get_p)
    get_p.set_defaults(func=cmd_get)

    # list-items
    list_items_p = subparsers.add_parser("list-items", help="List items in a dataset")
    list_items_p.add_argument("dataset_name", help="Dataset name")
    list_items_p.add_argument("--source-trace-id", help="Filter by source trace ID")
    list_items_p.add_argument("--source-observation-id", help="Filter by source observation ID")
    list_items_p.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    list_items_p.add_argument("--limit", type=int, default=50, help="Items per page (default: 50)")
    add_common_args(list_items_p)
    list_items_p.set_defaults(func=cmd_list_items)

    # get-item
    get_item_p = subparsers.add_parser("get-item", help="Get a dataset item by ID")
    get_item_p.add_argument("item_id", help="Dataset item ID")
    add_common_args(get_item_p)
    get_item_p.set_defaults(func=cmd_get_item)

    # create
    create_p = subparsers.add_parser("create", help="Create a new dataset")
    create_p.add_argument("name", help="Dataset name")
    create_p.add_argument("--description", help="Dataset description")
    create_p.add_argument("--metadata", help="JSON metadata")
    add_common_args(create_p)
    create_p.set_defaults(func=cmd_create)

    # create-item
    create_item_p = subparsers.add_parser("create-item", help="Create or upsert a dataset item")
    create_item_p.add_argument("dataset_name", help="Dataset name")
    create_item_p.add_argument("--input", help="JSON input data")
    create_item_p.add_argument("--expected-output", help="JSON expected output")
    create_item_p.add_argument("--metadata", help="JSON metadata")
    create_item_p.add_argument("--source-trace-id", help="Link to source trace")
    create_item_p.add_argument("--source-observation-id", help="Link to source observation")
    create_item_p.add_argument("--item-id", help="Item ID (for upsert)")
    create_item_p.add_argument("--status", choices=["ACTIVE", "ARCHIVED"], help="Item status")
    add_common_args(create_item_p)
    create_item_p.set_defaults(func=cmd_create_item)

    # delete-item
    delete_item_p = subparsers.add_parser("delete-item", help="Delete a dataset item")
    delete_item_p.add_argument("item_id", help="Dataset item ID to delete")
    delete_item_p.set_defaults(func=cmd_delete_item)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
