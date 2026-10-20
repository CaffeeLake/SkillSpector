# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SQL parameter binding and adverse OH1 ownership regressions."""

from __future__ import annotations

import pytest

from skillspector.nodes.analyzers import static_patterns_output_handling as output_handling
from skillspector.python_ast import parse_python_source


def _database_source(statement: str, driver: str = "psycopg") -> str:
    return (
        f"import {driver}\n"
        f"connection = {driver}.connect('example')\n"
        "cursor = connection.cursor()\n"
        f"{statement}\n"
    )


@pytest.mark.parametrize(
    "statement,driver",
    [
        ('cursor.execute("SELECT name FROM users WHERE id = %s", (output,))', "psycopg"),
        ('cursor.execute("SELECT name FROM users WHERE id = %(id)s", {"id": output})', "psycopg2"),
        (
            'cursor.execute(query="SELECT name FROM users WHERE id = %s", params=(response_id,))',
            "psycopg",
        ),
        (
            "cursor.execute(\"SELECT name FROM users WHERE id = ? AND note = '50%'\", (output,))",
            "sqlite3",
        ),
        ('cursor.execute("SELECT name FROM users WHERE id = %s", (output,))', "psycopg2"),
        (
            "connection.execute(\"SELECT name FROM users WHERE id = ? AND note = '50%'\", (output,))",
            "sqlite3",
        ),
    ],
)
def test_bound_database_query_is_not_output_injection(statement: str, driver: str) -> None:
    findings = output_handling.analyze(_database_source(statement, driver), "query.py", "python")
    assert not [finding for finding in findings if finding.rule_id == "OH1"]


@pytest.mark.parametrize(
    "query",
    [
        '"SELECT name FROM users WHERE id = " + output',
        '"SELECT name FROM users WHERE id = %s" % output',
        '"SELECT name FROM users WHERE id = {}".format(output)',
        'f"SELECT name FROM users WHERE id = {output}"',
    ],
)
def test_query_interpolation_remains_detected_even_with_parameters(query: str) -> None:
    source = _database_source(f"cursor.execute({query}, (user_id,))")
    assert any(
        finding.rule_id == "OH1"
        for finding in output_handling.analyze(source, "query.py", "python")
    )


@pytest.mark.parametrize(
    "source",
    [
        'cursor.execute("SELECT name FROM users WHERE id = %s", (output,))',
        _database_source(
            'initialize_plugin()\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
        ),
        'custom.query("SELECT name FROM users WHERE id = %s", (output,))',
        'execute("SELECT name FROM users WHERE id = %s", (output,))',
        _database_source(
            'cursor = custom\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
        ),
        _database_source(
            'cursor.execute = custom\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
        ),
        'import psycopg\npsycopg.connect = custom\nconnection = psycopg.connect("example")\ncursor = connection.cursor()\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))',
        'import psycopg\ndef run(psycopg):\n    cursor = psycopg.connect("example").cursor()\n    cursor.execute("SELECT name FROM users WHERE id = %s", (output,))',
        'import psycopg\nconnection = psycopg.connect("example", cursor_factory=custom)\ncursor = connection.cursor()\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))',
        _database_source(
            'custom(cursor)\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
        ),
        _database_source(
            'other = cursor\nother.execute = custom\ncursor.execute("SELECT name FROM users WHERE id = %s", (output,))'
        ),
        _database_source(
            'cursor.execute("SELECT name FROM users WHERE id = %s", (output,))\nif incomplete:'
        ),
    ],
)
def test_unknown_mutated_or_malformed_database_owner_stays_conservative(source: str) -> None:
    assert any(
        finding.rule_id == "OH1"
        for finding in output_handling.analyze(source, "query.py", "python")
    )


def test_safe_call_cannot_hide_adjacent_interpolation_with_unicode_offsets() -> None:
    source = _database_source(
        'label = "café"; cursor.execute("SELECT name FROM users WHERE id = %s", (output,)); '
        'cursor.execute("SELECT name FROM users WHERE id = " + output)'
    )
    findings = [
        finding
        for finding in output_handling.analyze(source, "query.py", "python")
        if finding.rule_id == "OH1"
    ]
    assert len(findings) == 1
    assert " + output" in (findings[0].matched_text or "")
    assert findings[0].location.start_line == 4


def test_embedded_python_keeps_lexical_sql_fallback() -> None:
    content = (
        "```python\n"
        + _database_source('cursor.execute("SELECT name FROM users WHERE id = %s", (output,))')
        + "```\n"
    )
    assert any(
        finding.rule_id == "OH1"
        for finding in output_handling.analyze(content, "SKILL.md", "markdown")
    )


def test_mismatched_source_parse_keeps_lexical_fallback() -> None:
    source = _database_source('cursor.execute("SELECT name FROM users WHERE id = %s", (output,))')
    parsed = parse_python_source(source.replace("SELECT", "SELECT\u200b"), "query.py")
    findings = output_handling.analyze(source, "query.py", "python", python_ast=parsed)
    assert any(finding.rule_id == "OH1" for finding in findings)
