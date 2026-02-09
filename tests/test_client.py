"""Unit tests for _client.py shared utilities."""

import json
import os
import sys

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "skills", "langfuse", "scripts"))

from _client import (
    MAX_AGE_MINUTES,
    age_to_timestamps,
    datetime_sort_key,
    format_output,
    parse_datetime_for_sort,
    parse_json_arg,
    parse_list_arg,
    truncate_large_strings,
    validate_age,
)


class TestTruncation:
    def test_short_string_unchanged(self):
        result, size = truncate_large_strings("hello", max_length=100)
        assert result == "hello"

    def test_long_string_truncated(self):
        long_str = "a" * 1000
        result, size = truncate_large_strings(long_str, max_length=100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_essential_fields_preserved(self):
        data = {
            "id": "test-id",
            "name": "test-name",
            "big_field": "x" * 1000,
        }
        result, size = truncate_large_strings(data, max_length=50)
        assert result["id"] == "test-id"
        assert result["name"] == "test-name"

    def test_large_field_truncated(self):
        data = {"input": "a" * 1000}
        result, size = truncate_large_strings(data, max_length=100)
        assert len(result["input"]) == 103

    def test_stacktrace_keeps_first_last_lines(self):
        lines = [f"line {i}" for i in range(20)]
        stacktrace = "\n".join(lines)
        data = {"exception.stacktrace": stacktrace}
        result, size = truncate_large_strings(data, max_length=50)
        result_lines = result["exception.stacktrace"].split("\n")
        assert result_lines[0] == "line 0"
        assert result_lines[-1] == "line 19"
        assert "..." in result_lines

    def test_empty_list(self):
        result, size = truncate_large_strings([])
        assert result == []

    def test_dict_with_nested_objects(self):
        data = {
            "id": "123",
            "nested": {
                "input": "a" * 1000,
                "id": "nested-id",
            },
        }
        result, size = truncate_large_strings(data, max_length=100)
        assert result["id"] == "123"
        assert result["nested"]["id"] == "nested-id"

    def test_truncation_level_1(self):
        result, size = truncate_large_strings("a" * 1000, max_length=500, truncation_level=1)
        # Level 1: max(50, 500//2) = 250
        assert len(result) == 253  # 250 + "..."

    def test_truncation_level_2(self):
        result, size = truncate_large_strings("a" * 1000, max_length=500, truncation_level=2)
        # Level 2: max(20, 500//5) = 100
        assert len(result) == 103  # 100 + "..."

    def test_oversized_returns_truncated_marker(self):
        data = "a" * 100
        result, size = truncate_large_strings(data, max_length=50, max_response_size=10, current_size=20)
        assert result == "[TRUNCATED]"

    def test_list_truncation(self):
        data = [{"input": "a" * 500} for _ in range(100)]
        result, size = truncate_large_strings(data, max_length=100, max_response_size=5000)
        assert len(result) <= 100  # Should be truncated or at boundary

    def test_none_passthrough(self):
        result, size = truncate_large_strings(None)
        assert result is None

    def test_int_passthrough(self):
        result, size = truncate_large_strings(42)
        assert result == 42

    def test_bool_passthrough(self):
        result, size = truncate_large_strings(True)
        assert result is True


class TestValidateAge:
    def test_valid_age(self):
        assert validate_age(60) == 60
        assert validate_age("1440") == 1440

    def test_max_age(self):
        assert validate_age(MAX_AGE_MINUTES) == MAX_AGE_MINUTES

    def test_invalid_age_exits(self):
        try:
            validate_age(0)
            assert False, "Should have exited"
        except SystemExit:
            pass

    def test_negative_age_exits(self):
        try:
            validate_age(-1)
            assert False, "Should have exited"
        except SystemExit:
            pass

    def test_too_large_age_exits(self):
        try:
            validate_age(MAX_AGE_MINUTES + 1)
            assert False, "Should have exited"
        except SystemExit:
            pass

    def test_non_numeric_exits(self):
        try:
            validate_age("abc")
            assert False, "Should have exited"
        except SystemExit:
            pass


class TestAgeToTimestamps:
    def test_returns_iso_strings(self):
        from_ts, to_ts = age_to_timestamps(60)
        assert isinstance(from_ts, str)
        assert isinstance(to_ts, str)
        assert "T" in from_ts
        assert "T" in to_ts

    def test_from_before_to(self):
        from _client import datetime

        from_ts, to_ts = age_to_timestamps(60)
        from_dt = datetime.fromisoformat(from_ts)
        to_dt = datetime.fromisoformat(to_ts)
        assert from_dt < to_dt


class TestParseJsonArg:
    def test_valid_json(self):
        result = parse_json_arg('{"key": "value"}')
        assert result == {"key": "value"}

    def test_none_returns_none(self):
        assert parse_json_arg(None) is None

    def test_array(self):
        result = parse_json_arg("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_invalid_json_exits(self):
        try:
            parse_json_arg("{invalid")
            assert False, "Should have exited"
        except SystemExit:
            pass


class TestParseListArg:
    def test_comma_separated(self):
        result = parse_list_arg("a,b,c")
        assert result == ["a", "b", "c"]

    def test_with_spaces(self):
        result = parse_list_arg("a, b , c")
        assert result == ["a", "b", "c"]

    def test_none_returns_none(self):
        assert parse_list_arg(None) is None

    def test_single_value(self):
        result = parse_list_arg("abc")
        assert result == ["abc"]

    def test_empty_parts_filtered(self):
        result = parse_list_arg("a,,b,")
        assert result == ["a", "b"]


class TestParseDatetimeForSort:
    def test_iso_string(self):
        result = parse_datetime_for_sort("2024-01-15T10:30:00+00:00")
        assert result is not None
        assert result.year == 2024

    def test_z_suffix(self):
        result = parse_datetime_for_sort("2024-01-15T10:30:00Z")
        assert result is not None

    def test_none(self):
        assert parse_datetime_for_sort(None) is None

    def test_empty_string(self):
        assert parse_datetime_for_sort("") is None

    def test_invalid_string(self):
        assert parse_datetime_for_sort("not-a-date") is None


class TestDatetimeSortKey:
    def test_valid_date(self):
        result = datetime_sort_key("2024-01-15T10:30:00Z")
        assert result.year == 2024

    def test_none_returns_min(self):
        from _client import _MIN_SORT_DATETIME

        result = datetime_sort_key(None)
        assert result == _MIN_SORT_DATETIME


class TestFormatOutput:
    def test_basic_output(self, capsys):
        format_output({"key": "value"}, compact=False)
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["data"]["key"] == "value"

    def test_with_metadata(self, capsys):
        format_output([], metadata={"count": 0}, compact=False)
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["metadata"]["count"] == 0

    def test_compact_truncates(self, capsys):
        format_output({"input": "a" * 1000}, compact=True)
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert len(output["data"]["input"]) < 1000
