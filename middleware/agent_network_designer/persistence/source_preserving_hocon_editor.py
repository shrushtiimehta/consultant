# Copyright © 2025-2026 Cognizant Technology Solutions Corp, www.cognizant.com.
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
#
# END COPYRIGHT

"""Surgical updates for instructions and descriptions in an existing HOCON file."""

import json
import os
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from pyhocon import ConfigFactory

_PATH_LOCKS: dict[Path, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class _Token:
    """A significant HOCON token and its exact source span."""

    kind: str
    value: str
    start: int
    end: int


class SourcePreservingHoconEditor:
    """Patch supported agent fields without resolving includes or rebuilding the network."""

    DEFAULT_MAX_STEPS = 500
    DEFAULT_MAX_EXECUTION_SECONDS = 600

    @classmethod
    def update_file(cls, source_file: str, changes: dict[str, dict[str, str]]) -> str:
        """Validate and apply changes atomically, returning the updated HOCON text."""
        path = Path(source_file).resolve()
        with _PATH_LOCKS_GUARD:
            path_lock = _PATH_LOCKS.setdefault(path, threading.Lock())
        with path_lock:
            original = path.read_text(encoding="utf-8")
            updated = cls.update_text(original, changes)
            cls._validate_and_atomic_write(path, updated)
            return updated

    @classmethod
    def update_text(cls, text: str, changes: dict[str, dict[str, str]]) -> str:
        """Return HOCON text with only the requested agent fields changed."""
        updated = text
        for agent_name, fields in changes.items():
            for field_name, new_value in fields.items():
                if field_name not in {"instructions", "description"}:
                    raise ValueError(f"Unsupported agent field for source-preserving edit: {field_name}")
                updated = cls._replace_agent_field(updated, agent_name, field_name, new_value)
        return cls._ensure_execution_limits(updated)

    @classmethod
    def _replace_agent_field(cls, text: str, agent_name: str, field_name: str, new_value: str) -> str:
        tokens = cls._tokens(text)
        block_start, _ = cls._find_agent_block(tokens, agent_name)
        object_start = cls._token_index_at(tokens, block_start)
        object_end = cls._matching_index(tokens, object_start, "{", "}")

        if field_name == "instructions":
            value_index = cls._property_value_index(tokens, object_start, object_end, "instructions")
        else:
            function_index = cls._property_value_index(tokens, object_start, object_end, "function")
            function_index = cls._skip_substitutions(tokens, function_index, object_end)
            if function_index >= object_end or tokens[function_index].value != "{":
                raise ValueError(f"Could not locate function object for agent {agent_name!r}.")
            function_end = cls._matching_index(tokens, function_index, "{", "}")
            value_index = cls._property_value_index(tokens, function_index, function_end, "description")

        value_index = cls._skip_substitutions(tokens, value_index, object_end)
        if value_index >= len(tokens) or tokens[value_index].kind != "string":
            raise ValueError(f"Could not locate {field_name!r} string for agent {agent_name!r} in source HOCON.")
        token = tokens[value_index]
        replacement = cls._format_string_value(new_value)

        # Every agent in an AAOSA network needs ${aaosa_instructions} appended after its
        # instructions string, no exceptions -- if this agent never had it (missing from the
        # source file to begin with), add it here rather than relying on whoever wrote
        # new_value to have included it. Doesn't apply to ${expertise_scoping_instructions},
        # which is deliberately only on entry-point agents, so it can't be forced onto every
        # agent the same way -- leave that one to whoever authored new_value.
        trailing_addition = ""
        if field_name == "instructions":
            next_index = value_index + 1
            has_trailing_aaosa = (
                next_index < object_end
                and tokens[next_index].kind == "substitution"
                and tokens[next_index].value == "${aaosa_instructions}"
            )
            if not has_trailing_aaosa:
                trailing_addition = " ${aaosa_instructions}"

        return text[: token.start] + replacement + trailing_addition + text[token.end :]

    @staticmethod
    def _format_string_value(value: str) -> str:
        """Render a HOCON string value, preferring a triple-quoted block for multi-line text
        so patched instructions/descriptions stay human-readable instead of one escaped line.
        Falls back to a JSON-quoted single line for one-liners or text that itself contains
        `\"\"\"` (which would break the triple-quote delimiter)."""
        if "\n" in value and '"""' not in value:
            return f'"""\n{value.strip()}\n"""'
        return json.dumps(value, ensure_ascii=False)

    @classmethod
    def _find_agent_block(cls, tokens: list[_Token], agent_name: str) -> tuple[int, int]:
        root_start = cls._first_token(tokens, "{")
        root_end = cls._matching_index(tokens, root_start, "{", "}")
        tools_index = cls._property_value_index(tokens, root_start, root_end, "tools")
        tools_index = cls._skip_substitutions(tokens, tools_index, root_end)
        if tools_index >= root_end or tokens[tools_index].value != "[":
            raise ValueError("Could not locate the root tools array in source HOCON.")
        tools_end = cls._matching_index(tokens, tools_index, "[", "]")

        matches: list[tuple[int, int]] = []
        index = tools_index + 1
        while index < tools_end:
            if tokens[index].value == "{":
                agent_end = cls._matching_index(tokens, index, "{", "}")
                try:
                    name_index = cls._property_value_index(tokens, index, agent_end, "name")
                except ValueError:
                    index = agent_end + 1
                    continue
                name_index = cls._skip_substitutions(tokens, name_index, agent_end)
                if name_index < agent_end and tokens[name_index].kind in {"string", "bare"}:
                    if tokens[name_index].value == agent_name:
                        matches.append((tokens[index].start, tokens[agent_end].end))
                index = agent_end + 1
                continue
            index += 1

        if not matches:
            raise ValueError(f"Could not locate agent {agent_name!r} in the root tools array.")
        if len(matches) > 1:
            raise ValueError(f"Agent name {agent_name!r} is duplicated in the root tools array.")
        return matches[0]

    @staticmethod
    def _token_index_at(tokens: list[_Token], start: int) -> int:
        for index, token in enumerate(tokens):
            if token.start == start:
                return index
        raise ValueError("Could not resolve HOCON token boundary.")

    @staticmethod
    def _first_token(tokens: list[_Token], value: str) -> int:
        for index, token in enumerate(tokens):
            if token.value == value:
                return index
        raise ValueError("Source HOCON has no root object.")

    @staticmethod
    def _skip_substitutions(tokens: list[_Token], index: int, end: int) -> int:
        while index < end and tokens[index].kind == "substitution":
            index += 1
        return index

    @classmethod
    def _property_value_index(cls, tokens: list[_Token], start: int, end: int, key: str) -> int:
        depth = 0
        index = start + 1
        while index < end:
            token = tokens[index]
            if token.value in {"{", "["}:
                depth += 1
            elif token.value in {"}", "]"}:
                depth -= 1
            elif depth == 0 and token.kind in {"string", "bare"} and token.value == key:
                separator = index + 1
                if separator < end and tokens[separator].value in {":", "="}:
                    return separator + 1
            index += 1
        raise ValueError(f"Could not locate direct property {key!r} in HOCON object.")

    @staticmethod
    def _matching_index(tokens: list[_Token], start: int, opening: str, closing: str) -> int:
        if tokens[start].value != opening:
            raise ValueError(f"Expected {opening!r} at HOCON token boundary.")
        depth = 0
        for index in range(start, len(tokens)):
            if tokens[index].value == opening:
                depth += 1
            elif tokens[index].value == closing:
                depth -= 1
                if depth == 0:
                    return index
        raise ValueError(f"Unterminated HOCON {opening!r} block.")

    @classmethod
    def _ensure_execution_limits(cls, text: str) -> str:
        tokens = cls._tokens(text)
        root_start = cls._first_token(tokens, "{")
        root_end = cls._matching_index(tokens, root_start, "{", "}")
        additions = []
        for key, value in (
            ("max_steps", cls.DEFAULT_MAX_STEPS),
            ("max_execution_seconds", cls.DEFAULT_MAX_EXECUTION_SECONDS),
        ):
            try:
                cls._property_value_index(tokens, root_start, root_end, key)
            except ValueError:
                additions.append(f'    "{key}": {value},')
        if not additions:
            return text
        insertion = "\n" + "\n".join(additions) + "\n"
        position = tokens[root_start].end
        return text[:position] + insertion + text[position:]

    @staticmethod
    # Keeping the scanner's state transitions together makes its source spans auditable.
    # pylint: disable=too-many-branches,too-many-statements
    def _tokens(text: str) -> list[_Token]:
        """Lex significant HOCON tokens while ignoring whitespace and comments."""
        tokens: list[_Token] = []
        index = 0
        punctuation = "{}[]:=,"
        while index < len(text):
            if text[index].isspace():
                index += 1
                continue
            if text[index] == "#" or text.startswith("//", index):
                newline = text.find("\n", index)
                index = len(text) if newline < 0 else newline + 1
                continue
            if text.startswith('"""', index):
                end = text.find('"""', index + 3)
                if end < 0:
                    raise ValueError("Unterminated triple-quoted HOCON string.")
                tokens.append(_Token("string", text[index + 3 : end], index, end + 3))
                index = end + 3
                continue
            if text[index] == '"':
                end = index + 1
                escaped = False
                while end < len(text):
                    if text[end] == '"' and not escaped:
                        break
                    escaped = text[end] == "\\" and not escaped
                    if text[end] != "\\":
                        escaped = False
                    end += 1
                if end >= len(text):
                    raise ValueError("Unterminated quoted HOCON string.")
                raw = text[index : end + 1]
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid quoted HOCON string.") from exc
                tokens.append(_Token("string", value, index, end + 1))
                index = end + 1
                continue
            if text.startswith("${", index):
                end = text.find("}", index + 2)
                if end < 0:
                    raise ValueError("Unterminated HOCON substitution.")
                tokens.append(_Token("substitution", text[index : end + 1], index, end + 1))
                index = end + 1
                continue
            if text[index] in punctuation:
                tokens.append(_Token("punctuation", text[index], index, index + 1))
                index += 1
                continue
            end = index
            while end < len(text):
                if text[end].isspace() or text[end] in punctuation or text[end] == "#" or text.startswith("//", end):
                    break
                end += 1
            if end == index:
                raise ValueError(f"Unexpected HOCON character at offset {index}.")
            tokens.append(_Token("bare", text[index:end], index, end))
            index = end
        return tokens

    @classmethod
    def _validate_and_atomic_write(cls, path: Path, content: str) -> None:
        mode = stat.S_IMODE(path.stat().st_mode)
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", suffix=".hocon", dir=path.parent, delete=False
            ) as temporary:
                temporary.write(content)
                temporary_name = temporary.name
            # Syntax-check without resolving substitutions. Resolution depends on the
            # network's include graph and environment and is handled by normal loading.
            ConfigFactory.parse_file(temporary_name, resolve=False)
            os.chmod(temporary_name, mode)
            os.replace(temporary_name, path)
            temporary_name = ""
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
