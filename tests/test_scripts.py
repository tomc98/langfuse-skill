"""Integration tests for Langfuse scripts (mock api_request)."""

import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "skills", "langfuse", "scripts"))


def mock_api_on(module_name):
    """Patch api_request on a specific script module."""
    return patch(f"{module_name}.api_request")


class TestTraces:
    def test_fetch_traces(self, capsys):
        with mock_api_on("traces") as mock:
            mock.return_value = {
                "data": [
                    {"id": "trace-1", "name": "test", "timestamp": "2024-01-15T10:00:00Z"},
                    {"id": "trace-2", "name": "test2", "timestamp": "2024-01-15T11:00:00Z"},
                ],
                "meta": {"totalItems": 2},
            }
            import argparse

            from traces import cmd_fetch

            args = argparse.Namespace(
                age="60",
                name=None,
                user_id=None,
                session_id=None,
                metadata=None,
                tags=None,
                page=1,
                limit=50,
                no_truncate=False,
            )
            cmd_fetch(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["metadata"]["item_count"] == 2
            assert len(output["data"]) == 2

    def test_fetch_traces_with_metadata_filter(self, capsys):
        with mock_api_on("traces") as mock:
            mock.return_value = {
                "data": [
                    {"id": "t1", "metadata": {"env": "prod"}},
                    {"id": "t2", "metadata": {"env": "staging"}},
                ],
                "meta": {},
            }
            import argparse

            from traces import cmd_fetch

            args = argparse.Namespace(
                age="60",
                name=None,
                user_id=None,
                session_id=None,
                metadata='{"env": "prod"}',
                tags=None,
                page=1,
                limit=50,
                no_truncate=False,
            )
            cmd_fetch(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["metadata"]["item_count"] == 1
            assert output["data"][0]["id"] == "t1"

    def test_get_trace(self, capsys):
        with mock_api_on("traces") as mock:
            mock.return_value = {"id": "trace-1", "name": "test", "observations": []}
            import argparse

            from traces import cmd_get

            args = argparse.Namespace(trace_id="trace-1", no_truncate=True)
            cmd_get(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["data"]["id"] == "trace-1"


class TestObservations:
    def test_fetch_observations(self, capsys):
        with mock_api_on("observations") as mock:
            mock.return_value = {
                "data": [{"id": "obs-1", "type": "GENERATION", "name": "chat"}],
                "meta": {"totalItems": 1},
            }
            import argparse

            from observations import cmd_fetch

            args = argparse.Namespace(
                age="60",
                type="GENERATION",
                name=None,
                user_id=None,
                trace_id=None,
                parent_observation_id=None,
                page=1,
                limit=50,
                no_truncate=False,
            )
            cmd_fetch(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["metadata"]["item_count"] == 1

    def test_get_observation(self, capsys):
        with mock_api_on("observations") as mock:
            mock.return_value = {"id": "obs-1", "type": "SPAN", "name": "process"}
            import argparse

            from observations import cmd_get

            args = argparse.Namespace(observation_id="obs-1", no_truncate=True)
            cmd_get(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["data"]["id"] == "obs-1"


class TestSessions:
    def test_fetch_sessions(self, capsys):
        with mock_api_on("sessions") as mock:
            mock.return_value = {
                "data": [{"id": "session-1"}],
                "meta": {"totalItems": 1},
            }
            import argparse

            from sessions import cmd_fetch

            args = argparse.Namespace(age="60", page=1, limit=50, no_truncate=False)
            cmd_fetch(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["metadata"]["item_count"] == 1

    def test_session_details(self, capsys):
        with mock_api_on("sessions") as mock:
            # First call: session metadata, second call: traces
            mock.side_effect = [
                {"id": "s1", "createdAt": "2024-01-15T10:00:00Z"},
                {
                    "data": [
                        {"id": "t1", "timestamp": "2024-01-15T10:00:00Z", "userId": "user1"},
                        {"id": "t2", "timestamp": "2024-01-15T11:00:00Z", "userId": "user1"},
                    ],
                    "meta": {},
                },
            ]
            import argparse

            from sessions import cmd_details

            args = argparse.Namespace(session_id="s1", no_truncate=True)
            cmd_details(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["data"]["trace_count"] == 2
            assert output["data"]["user_id"] == "user1"

    def test_user_sessions(self, capsys):
        with mock_api_on("sessions") as mock:
            mock.return_value = {
                "data": [
                    {"id": "t1", "sessionId": "s1", "timestamp": "2024-01-15T10:00:00Z"},
                    {"id": "t2", "sessionId": "s1", "timestamp": "2024-01-15T11:00:00Z"},
                    {"id": "t3", "sessionId": "s2", "timestamp": "2024-01-15T12:00:00Z"},
                ],
                "meta": {},
            }
            import argparse

            from sessions import cmd_user

            args = argparse.Namespace(user_id="user1", age="1440", no_truncate=True)
            cmd_user(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["metadata"]["session_count"] == 2


class TestExceptions:
    def test_find_exceptions(self, capsys):
        with mock_api_on("exceptions") as mock:
            mock.return_value = {
                "data": [
                    {
                        "id": "obs-1",
                        "type": "SPAN",
                        "metadata": {"code.filepath": "src/main.py"},
                        "events": [
                            {"attributes": {"exception.type": "ValueError", "exception.message": "bad value"}},
                        ],
                    },
                    {
                        "id": "obs-2",
                        "type": "SPAN",
                        "metadata": {"code.filepath": "src/main.py"},
                        "events": [
                            {"attributes": {"exception.type": "TypeError", "exception.message": "wrong type"}},
                        ],
                    },
                ],
                "meta": {},
            }
            import argparse

            from exceptions import cmd_find

            args = argparse.Namespace(age="60", group_by="file")
            cmd_find(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["data"][0]["group"] == "src/main.py"
            assert output["data"][0]["count"] == 2

    def test_find_exceptions_by_type(self, capsys):
        with mock_api_on("exceptions") as mock:
            mock.return_value = {
                "data": [
                    {
                        "id": "obs-1",
                        "type": "SPAN",
                        "metadata": {},
                        "events": [
                            {"attributes": {"exception.type": "ValueError"}},
                            {"attributes": {"exception.type": "ValueError"}},
                        ],
                    },
                ],
                "meta": {},
            }
            import argparse

            from exceptions import cmd_find

            args = argparse.Namespace(age="60", group_by="type")
            cmd_find(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["data"][0]["group"] == "ValueError"
            assert output["data"][0]["count"] == 2

    def test_exception_count(self, capsys):
        with mock_api_on("exceptions") as mock:
            mock.return_value = {
                "data": [
                    {
                        "id": "obs-1",
                        "traceId": "t1",
                        "events": [{"attributes": {"exception.type": "Error"}}],
                    },
                ],
                "meta": {},
            }
            import argparse

            from exceptions import cmd_count

            args = argparse.Namespace(age="60")
            cmd_count(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["data"]["exception_count"] == 1
            assert output["data"]["trace_count"] == 1


class TestPrompts:
    def test_list_prompts(self, capsys):
        with mock_api_on("prompts") as mock:
            mock.return_value = {
                "data": [
                    {"name": "prompt1", "type": "text", "versions": [1, 2], "labels": ["production"]},
                ],
                "meta": {"totalItems": 1},
            }
            import argparse

            from prompts import cmd_list

            args = argparse.Namespace(name=None, label=None, tag=None, page=1, limit=50)
            cmd_list(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["metadata"]["item_count"] == 1
            assert output["data"][0]["name"] == "prompt1"

    def test_get_prompt(self, capsys):
        with mock_api_on("prompts") as mock:
            mock.return_value = {
                "name": "greeting",
                "version": 3,
                "type": "text",
                "prompt": "Hello {{name}}",
                "labels": ["production"],
                "tags": [],
                "config": {"model": "gpt-4"},
                "id": "p1",
            }
            import argparse

            from prompts import cmd_get

            args = argparse.Namespace(name="greeting", label="production", version=None, no_truncate=True)
            cmd_get(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["data"]["name"] == "greeting"
            assert output["data"]["version"] == 3


class TestDatasets:
    def test_list_datasets(self, capsys):
        with mock_api_on("datasets") as mock:
            mock.return_value = {
                "data": [
                    {"id": "d1", "name": "test-dataset", "description": "Test"},
                ],
                "meta": {"totalItems": 1},
            }
            import argparse

            from datasets import cmd_list

            args = argparse.Namespace(page=1, limit=50)
            cmd_list(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["metadata"]["item_count"] == 1

    def test_create_dataset_item(self, capsys):
        with mock_api_on("datasets") as mock:
            mock.return_value = {
                "id": "item-1",
                "datasetName": "test",
                "input": {"q": "test"},
                "expectedOutput": {"a": "yes"},
            }
            import argparse

            from datasets import cmd_create_item

            args = argparse.Namespace(
                dataset_name="test",
                input='{"q": "test"}',
                expected_output='{"a": "yes"}',
                metadata=None,
                source_trace_id=None,
                source_observation_id=None,
                item_id=None,
                status=None,
                no_truncate=False,
            )
            cmd_create_item(args)
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["metadata"]["created"] is True


class TestScriptHelp:
    """Verify all scripts respond to --help without errors."""

    @pytest.mark.parametrize(
        "script",
        [
            "traces",
            "observations",
            "sessions",
            "exceptions",
            "prompts",
            "datasets",
            "schema",
        ],
    )
    def test_help_flag(self, script):
        import subprocess

        result = subprocess.run(
            [sys.executable, f"skills/langfuse/scripts/{script}.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
