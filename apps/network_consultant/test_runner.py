# Copyright © 2025-2026 Cognizant Technology Solutions Corp, www.cognizant.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# END COPYRIGHT
"""
Runs every ANTeGen-generated test fixture for a network and reports pass/fail per
fixture, without going through pytest. Reuses neuro_san's own data-driven test
driver -- the same one `make test-integration` uses -- so results match exactly
what CI would report.
"""

import glob
import logging
import os
import re
import time
from typing import Any
from typing import Callable
from unittest import TestCase

from leaf_common.time.timeout_reached_exception import TimeoutReachedException
from neuro_san.test.driver.data_driven_agent_test_driver import DataDrivenAgentTestDriver
from neuro_san.test.unittest.unit_test_assert_forwarder import UnitTestAssertForwarder

logger = logging.getLogger(__name__)

API_KEY_ERROR_MARKER = "API KEY error detected"

# Matches coded_tools/agent_network_consultant/read_thinking_trace.py's THINKING_DIR
# and the "--- <agent_origin> ---" section headers it parses.
IMPROVEMENT_THINKING_DIR = os.path.join("logs", "thinking_dir", "improvement")

# Matches ThinkingFileMessageProcessor._write_to_file's entry header exactly:
# f"\n[{message_type_str}{use_origin}] @ {timestamp_str}:\n"
_THINKING_ENTRY_HEADER = re.compile(r"^\[(?P<type>[A-Z_]+)[^\]]*\] @ .+:$", re.MULTILINE)


# Telemetry keys unique to neuro-san's own token/cost-accounting report -- never part of an
# agent's actual reasoning or AAOSA dialogue.
_COST_ACCOUNTING_KEYS = ("prompt_tokens", "completion_tokens", "total_cost", "total_tokens")


def _is_noise_paragraph(paragraph: str) -> bool:
    """A chat_context dump (conversation-continuation bookkeeping) or a token/cost-accounting
    report -- both are telemetry a diagnosing agent has no use for, not dialogue content."""
    if paragraph.startswith("chat_context:"):
        return True
    body = paragraph.strip("`").removeprefix("json").strip() if paragraph.startswith("```") else paragraph
    return body.startswith("{") and any(key in body for key in _COST_ACCOUNTING_KEYS)


def _strip_system_entries(raw_text: str) -> str:
    """Drop every [SYSTEM ...] entry (an agent's full instructions/system prompt) from one
    agent's raw thinking file, keeping everything else -- its own reasoning, the AAOSA
    inquiry/response exchange with its down-chain agents, tool calls/results, final answer.
    Also drops chat_context dumps and cost-accounting reports, wherever they appear."""
    headers = list(_THINKING_ENTRY_HEADER.finditer(raw_text))
    if not headers:
        return raw_text.strip()
    kept: list[str] = []
    for index, header in enumerate(headers):
        if header.group("type") == "SYSTEM":
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(raw_text)
        # Split header from body BEFORE paragraph-splitting: a noise block (chat_context dump,
        # cost report) can immediately follow the header with no blank line in between, which
        # would otherwise glue it onto the header into one paragraph that starts with "[TYPE...]"
        # instead of "{" or "```" -- invisible to _is_noise_paragraph.
        header_line = header.group(0)
        body = raw_text[header.end() : end].strip()
        paragraphs = [p for p in body.split("\n\n") if not _is_noise_paragraph(p.strip())]
        body = "\n\n".join(paragraphs).strip()
        entry = f"{header_line}\n{body}" if body else ""
        if entry:
            kept.append(entry)
    return "\n\n".join(kept)


def _write_consolidated_thinking(fixture_name: str, started: float) -> None:
    """Consolidate neuro-san's raw per-agent thinking files (written under
    AGENT_TEST_THINKING_BASIS while this fixture just ran) into the one file
    read_thinking_trace serves back to the consultant's diagnosing sub-agents: system prompts
    stripped, one `--- <agent_origin> ---` section per agent, only this run's own directories
    (older leftovers under the same basis dir are ignored via mtime).

    No-ops if AGENT_TEST_THINKING_BASIS isn't set (thinking files were never being written in
    the first place) or if this fixture produced none.

    Known upstream gap (neuro_san.test.driver.data_driven_agent_test_driver._setup_thinking_dir):
    its per-turn directory name has only second-level timestamp precision, so two turns of the
    same iteration that finish within the same wall-clock second collide on one directory --
    the later turn's setup rmtree()s the earlier turn's files before writing its own. Multi-turn
    fixtures can silently lose an earlier turn's thinking trace; nothing in this file can recover
    it after the fact. Not something to patch here -- it lives in the installed neuro-san package.
    """
    basis_dir = os.environ.get("AGENT_TEST_THINKING_BASIS")
    if not basis_dir:
        return
    run_dirs = sorted(
        d
        for d in glob.glob(os.path.join(basis_dir, f"*_{fixture_name}*"))
        if os.path.isdir(d) and os.path.getmtime(d) >= started
    )
    if not run_dirs:
        return

    sections: dict[str, list[str]] = {}
    for run_dir in run_dirs:
        for agent_file in sorted(os.listdir(run_dir)):
            path = os.path.join(run_dir, agent_file)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8", errors="replace") as file:
                raw = file.read()
            # First line is always "Agent: <origin_str>\n" (ThinkingFileMessageProcessor);
            # use that as the true origin name rather than the "/"->"__" sanitized filename.
            first_line, _, rest = raw.partition("\n")
            agent_origin = first_line[len("Agent: ") :].strip() if first_line.startswith("Agent: ") else agent_file
            filtered = _strip_system_entries(rest)
            if filtered:
                sections.setdefault(agent_origin, []).append(filtered)

    if not sections:
        return

    os.makedirs(IMPROVEMENT_THINKING_DIR, exist_ok=True)
    out_path = os.path.join(IMPROVEMENT_THINKING_DIR, f"{fixture_name}.txt")
    with open(out_path, "w", encoding="utf-8") as out_file:
        for agent_origin, chunks in sections.items():
            out_file.write(f"--- {agent_origin} ---\n")
            out_file.write("\n\n".join(chunks))
            out_file.write("\n\n")
    logger.info("Consolidated thinking trace written: %s (%d agent(s))", out_path, len(sections))
SUCCESS_RATIO_PATTERN = re.compile(r'("success_ratio"\s*:\s*")(\d+/\d+)(")')


def fixture_paths(fixtures_dir: str) -> list[str]:
    """:return: Sorted list of fixture HOCON paths under tests/fixtures/<fixtures_dir>/."""
    search_dir = os.path.join("tests", "fixtures", fixtures_dir)
    return sorted(glob.glob(os.path.join(search_dir, "*.hocon")))


def _set_success_ratio_for_paths(paths: list[str], ratio: str) -> dict[str, str]:
    """Overwrite success_ratio in place for exactly the given fixture paths."""
    originals: dict[str, str] = {}
    for path in paths:
        with open(path, encoding="utf-8") as fixture_file:
            text = fixture_file.read()
        match = SUCCESS_RATIO_PATTERN.search(text)
        if not match or match.group(2) == ratio:
            continue
        originals[path] = match.group(2)
        with open(path, "w", encoding="utf-8") as fixture_file:
            fixture_file.write(SUCCESS_RATIO_PATTERN.sub(rf"\g<1>{ratio}\g<3>", text, count=1))
        logger.info("success_ratio %s -> %s: %s", originals[path], ratio, path)
    return originals


def set_success_ratio_for_fixtures(fixtures_dir: str, fixture_names: list[str], ratio: str) -> dict[str, str]:
    """
    Overwrite success_ratio in place for specific fixtures only (by basename), e.g. those the
    consultant flagged CONFIDENT_FIX for -- letting most fixtures stay cheap (1/1) while only
    the ones worth the extra token spend get re-verified at a stricter ratio.

    :param fixtures_dir: Network path under tests/fixtures/, as in run_all_tests.
    :param fixture_names: Basenames (e.g. "foo.hocon") to change; others are left untouched.
    :param ratio: New value, e.g. "3/3".
    :return: {fixture_path: original_ratio} for every fixture actually changed, so the
             caller can restore it later via restore_success_ratios.
    """
    wanted = set(fixture_names)
    paths = [path for path in fixture_paths(fixtures_dir) if os.path.basename(path) in wanted]
    return _set_success_ratio_for_paths(paths, ratio)


def set_success_ratios(fixtures_dir: str, ratio: str) -> dict[str, str]:
    """
    Overwrite every fixture's top-level success_ratio in place.

    :param fixtures_dir: Network path under tests/fixtures/, as in run_all_tests.
    :param ratio: New value, e.g. "3/3".
    :return: {fixture_path: original_ratio} for every fixture actually changed, so the
             caller can restore it later via restore_success_ratios.
    """
    originals = _set_success_ratio_for_paths(fixture_paths(fixtures_dir), ratio)
    logger.info("set_success_ratios(%s, %s): changed %d/%d fixtures", fixtures_dir, ratio, len(originals),
                len(fixture_paths(fixtures_dir)))
    return originals


def restore_success_ratios(originals: dict[str, str]) -> None:
    """Undo set_success_ratios, restoring each fixture's original success_ratio."""
    for path, ratio in originals.items():
        with open(path, encoding="utf-8") as fixture_file:
            text = fixture_file.read()
        with open(path, "w", encoding="utf-8") as fixture_file:
            fixture_file.write(SUCCESS_RATIO_PATTERN.sub(rf"\g<1>{ratio}\g<3>", text, count=1))
        logger.info("success_ratio restored -> %s: %s", ratio, path)
    logger.info("restore_success_ratios: restored %d fixture(s)", len(originals))


class _ApiKeyErrorCapture(logging.Handler):
    """Catches neuro-san's own logged API-key errors, which it otherwise only logs and
    silently falls back from -- never raising an exception a caller could catch."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if API_KEY_ERROR_MARKER in message:
            self.messages.append(message.strip())


def run_fixture(fixture_path: str) -> dict[str, Any]:
    """
    :param fixture_path: Path to a single test fixture HOCON file.
    :return: Result with fixture/path/passed/message/infrastructure_error fields.
    """
    # one_test() raises a single AssertionError (summarizing every interaction/iteration
    # mismatch) only if the fixture's success_ratio wasn't met, so a plain try/except
    # is all the aggregation this needs.
    fixture_name = os.path.basename(fixture_path)
    asserts = UnitTestAssertForwarder(TestCase())
    driver = DataDrivenAgentTestDriver(asserts, test_name=fixture_name)
    result = {"fixture": fixture_name, "path": fixture_path, "infrastructure_error": False}

    logger.info("run_fixture start: %s", fixture_name)
    started = time.time()
    capture = _ApiKeyErrorCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(capture)
    try:
        driver.one_test(fixture_path)
        logger.info("run_fixture pass (%.1fs): %s", time.time() - started, fixture_name)
        return {**result, "passed": True, "message": None}
    except AssertionError as exc:
        cause = exc.__cause__ or exc
        if capture.messages:
            api_key_summary = "\n".join(capture.messages)
            logger.warning(
                "run_fixture infrastructure_error (%.1fs, API key error): %s", time.time() - started, fixture_name
            )
            return {
                **result,
                "passed": False,
                "message": f"{api_key_summary}\n\n(Original assertion, likely a symptom of the above: {cause})",
                "infrastructure_error": True,
            }
        logger.info("run_fixture fail (%.1fs): %s -- %s", time.time() - started, fixture_name, cause)
        return {**result, "passed": False, "message": str(cause)}
    except TimeoutReachedException as exc:
        # exc carries no message of its own (leaf_common never sets one) -- report the interaction's
        # own timeout budget so a human knows to raise timeout_in_seconds, not chase a phantom bug.
        limit = exc.timeout.get_limit_in_seconds()
        name = exc.timeout.get_name() or fixture_name
        message = (
            f"TIMEOUT_ISSUE: {fixture_name}: interaction {name!r} exceeded its {limit:.0f}s timeout -- "
            f"increase timeout_in_seconds in this fixture."
        )
        logger.warning("run_fixture infrastructure_error (%.1fs, timeout): %s", time.time() - started, fixture_name)
        return {**result, "passed": False, "message": message, "infrastructure_error": True}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        message = f"{type(exc).__name__}: {exc}"
        if capture.messages:
            api_key_summary = "\n".join(capture.messages)
            message = f"{api_key_summary}\n\n(Original exception, likely a symptom of the above: {message})"
        logger.warning(
            "run_fixture infrastructure_error (%.1fs): %s -- %s", time.time() - started, fixture_name, message
        )
        return {
            **result,
            "passed": False,
            "message": message,
            "infrastructure_error": True,
        }
    finally:
        root_logger.removeHandler(capture)
        _write_consolidated_thinking(fixture_name, started)


def run_all_tests(
    fixtures_dir: str, only_fixtures: list[str] = None, on_result: Callable[[dict, int], None] = None
) -> list[dict[str, Any]]:
    """
    :param fixtures_dir: Network path under tests/fixtures/, e.g. "generated/coffee_shop"
                (matches ANTeGen's target_agent_name, i.e. the hocon file path minus ".hocon").
    :param only_fixtures: If given, run only these basenames (e.g. ["foo.hocon"]) instead of
                every fixture in the directory -- lets a caller cheaply re-verify just the
                handful of fixtures it touched instead of paying for the full suite every round.
    :param on_result: If given, called as (result, batch_size) after each fixture finishes, so a
                caller can chart progress per test instead of only once the whole batch is done.
    :return: One result dict (see run_fixture) per fixture found.
    """
    paths = fixture_paths(fixtures_dir)
    if only_fixtures is not None:
        wanted = set(only_fixtures)
        paths = [path for path in paths if os.path.basename(path) in wanted]
    if not paths:
        search_dir = os.path.join("tests", "fixtures", fixtures_dir)
        logger.warning("run_all_tests: no fixtures found under %s (only_fixtures=%s)", search_dir, only_fixtures)
        return [
            {
                "fixture": "<fixture discovery>",
                "path": search_dir,
                "passed": False,
                "message": f"No test fixtures found under '{search_dir}'.",
                "infrastructure_error": True,
            }
        ]
    logger.info(
        "run_all_tests start: %d fixture(s) under %s%s",
        len(paths),
        fixtures_dir,
        f" (subset of {only_fixtures})" if only_fixtures is not None else "",
    )
    started = time.time()
    results = []
    for path in paths:
        result = run_fixture(path)
        results.append(result)
        if on_result is not None:
            on_result(result, len(paths))
    passed = sum(1 for r in results if r["passed"])
    logger.info(
        "run_all_tests done (%.1fs): %d/%d passing under %s", time.time() - started, passed, len(results), fixtures_dir
    )
    return results
