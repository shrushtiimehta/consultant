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
network_consultant -- same iterative test-and-fix loop as apps/network_improver, except the
fix step calls the specialized agent_network_consultant network instead of
agent_network_designer in modify mode. agent_network_designer is a general create/modify
tool that has to first figure out "is this structural or instructions-only"; consultant_editor
skips that and goes straight from a failing-test report to per-agent instruction fixes.

By default this runs in-process (--connection direct), no server needed. Pass --connection
http to instead talk to an already-running `ns run` server -- useful if you want this to share
a server with other clients, but NOT to watch the run in nsflow: nsflow's live view only shows
conversations started through its own UI/websocket, so a script hitting the neuro-san server's
plain chat API directly (this one) never appears there regardless of connection type.

Usage:
    python -m apps.network_consultant.runner --use-case "A coffee shop order-status bot"
    python -m apps.network_consultant.runner --hocon-file generated/coffee_shop.hocon \
        --direction "Preserve order lookup"
    python -m apps.network_consultant.runner --hocon-file generated/coffee_shop.hocon \
        --direction "Preserve order lookup" --connection http
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import PurePosixPath
from typing import Any
from typing import Optional
from unittest import TestCase

from leaf_common.time.timeout_reached_exception import TimeoutReachedException

import neuro_san_studio

# neuro_san_studio is now vendored into this repo's root and installed from there, but resolve
# toolbox_info.hocon via the installed package's __file__ rather than a path relative to this
# repo, so it still works if that ever changes back to a separate installed dependency.
_toolbox_dir = os.path.join(os.path.dirname(neuro_san_studio.__file__), "toolbox")

# Only used by --connection direct (the default), which loads and runs the network in this
# process instead of talking to a `ns run` server; harmless (and unused) otherwise.
os.environ.setdefault("AGENT_MANIFEST_FILE", "registries/manifest.hocon")
os.environ.setdefault("AGENT_TOOL_PATH", "coded_tools")
os.environ.setdefault("AGENT_TOOLBOX_INFO_FILE", os.path.join(_toolbox_dir, "toolbox_info.hocon"))
# get_toolbox.py (agent_network_designer's own toolbox lookup) reads this DIFFERENT env var,
# defaulting to a repo-relative path that doesn't exist here -- point it at the installed
# package's copy too.
os.environ.setdefault(
    "AGENT_NETWORK_DESIGNER_TOOLBOX_INFO_FILE",
    os.path.join(_toolbox_dir, "agent_network_designer_toolbox_info.hocon"),
)
# DataDrivenAgentTestDriver (used by run_all_tests) only writes per-interaction thinking files,
# and only uses the fuller "MAXIMAL" chat filter, when this is set -- otherwise it silently
# skips both. See _setup_thinking_dir in neuro_san's data_driven_agent_test_driver.py.
os.environ.setdefault("AGENT_TEST_THINKING_BASIS", "/tmp/network_consultant_test_thinking")

# These imports intentionally follow the direct-session environment defaults above.
# pylint: disable=wrong-import-position
from neuro_san.client.agent_session_factory import AgentSessionFactory  # noqa: E402
from neuro_san.client.streaming_input_processor import StreamingInputProcessor  # noqa: E402
from neuro_san.test.driver.data_driven_agent_test_driver import DataDrivenAgentTestDriver  # noqa: E402
from neuro_san.test.unittest.unit_test_assert_forwarder import UnitTestAssertForwarder  # noqa: E402

from coded_tools.agent_network_consultant.network_scratchpad import clear_for_hocon_file  # noqa: E402

# Not __name__: this module runs as "__main__" via `python -m`, which would otherwise
# produce an unhelpful logger name.
logger = logging.getLogger("network_consultant")

# --- Fixture test running (formerly apps/network_consultant/test_runner.py) ---------------
# Runs every ANTeGen-generated test fixture for a network and reports pass/fail per fixture,
# without going through pytest. Reuses neuro_san's own data-driven test driver -- the same one
# `make test-integration` uses -- so results match exactly what CI would report.

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

    :param fixtures_dir: Network path under tests/fixtures/, matching _fixture_paths.
    :param fixture_names: Basenames (e.g. "foo.hocon") to change; others are left untouched.
    :param ratio: New value, e.g. "3/3".
    :return: {fixture_path: original_ratio} for every fixture actually changed, so the
             caller can restore it later via restore_success_ratios.
    """
    wanted = set(fixture_names)
    paths = [path for path in _fixture_paths(fixtures_dir) if os.path.basename(path) in wanted]
    return _set_success_ratio_for_paths(paths, ratio)


def restore_success_ratios(originals: dict[str, str]) -> None:
    """Undo set_success_ratio_for_fixtures, restoring each fixture's original success_ratio."""
    for path, ratio in originals.items():
        with open(path, encoding="utf-8") as fixture_file:
            text = fixture_file.read()
        with open(path, "w", encoding="utf-8") as fixture_file:
            fixture_file.write(SUCCESS_RATIO_PATTERN.sub(rf"\g<1>{ratio}\g<3>", text, count=1))
        logger.info("success_ratio restored -> %s: %s", ratio, path)
    logger.info("restore_success_ratios: restored %d fixture(s)", len(originals))


class _ApiKeyErrorCapture(logging.Handler):
    """Catches neuro-san's own logged API-key errors, which it otherwise only logs and
    silently falls back from -- never raising an exception a caller could catch.

    # ponytail: attaches to the single process-wide root logger, so with run_all_tests'
    # concurrent fixtures this instance can also receive another fixture's log record --
    # neither contextvars nor threading.local propagate into the inner ThreadPoolExecutor
    # workers that actually emit these records (verified empirically), so there's no clean
    # way to attribute a record to "which fixture" without patching vendored neuro_san.
    # Bounded impact: a misattributed record only mislabels one fixture's failure as
    # infrastructure_error for one iteration, and the self-improve loop's max_iterations
    # retries self-correct it. Revisit only if that misclassification shows up in practice.
    """

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


# How many fixtures run_all_tests runs concurrently. Mirrors the ThreadPoolExecutor pattern
# DataDrivenAgentTestDriver.one_test() already uses to run a single fixture's repeats in parallel.
MAX_CONCURRENT_FIXTURES = 5


def run_all_tests(fixtures_dir: str, only_fixtures: list[str] = None) -> list[dict[str, Any]]:
    """
    :param fixtures_dir: Network path under tests/fixtures/, e.g. "generated/coffee_shop"
                (matches ANTeGen's target_agent_name, i.e. the hocon file path minus ".hocon").
    :param only_fixtures: If given, run only these basenames (e.g. ["foo.hocon"]) instead of
                every fixture in the directory -- lets a caller cheaply re-verify just the
                handful of fixtures it touched instead of paying for the full suite every round.
    :return: One result dict (see run_fixture) per fixture found.
    """
    paths = _fixture_paths(fixtures_dir)
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
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FIXTURES) as executor:
        futures = [executor.submit(run_fixture, path) for path in paths]
        for future in as_completed(futures):
            results.append(future.result())
    passed = sum(1 for r in results if r["passed"])
    logger.info(
        "run_all_tests done (%.1fs): %d/%d passing under %s", time.time() - started, passed, len(results), fixtures_dir
    )
    return results


# --- End of former test_runner.py content --------------------------------------------------

# Ratio a fixture gets bumped to once consultant_editor is CONFIDENT its fix holds up under
# repeated runs. Everything else stays at whatever cheap ratio (usually 1/1) it already had --
# re-running every fixture at 3/3 every round is not worth the token cost.
CONFIDENT_SUCCESS_RATIO = "3/3"

# Generous by design: the goal is to actually improve the network, not stop the moment progress
# looks slow. max-iterations is a safety ceiling, not a target -- override with --max-iterations.
DEFAULT_MAX_ITERATIONS = 20
PLATEAU_STRIKES = 3
# Good enough to move on, checked ONLY against a full-suite result -- never against a subset
# re-check while the network is still climbing. Fixing continues until the failing subset is
# clean; the full sweep that follows is then accepted at this rate instead of demanding 100%.
GOOD_ENOUGH_RATIO = 0.8
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080
THINKING_FILE = "/tmp/network_consultant_thinking.txt"
# StreamingInputProcessor only attaches its ThinkingFileMessageProcessor when BOTH
# thinking_file and thinking_dir are non-None -- a bare thinking_file is silently ignored.
THINKING_DIR = "/tmp/network_consultant_thinking"


def open_session(agent_name: str, connection: str, host: str, port: int):
    """Open a session against one of this studio's own networks -- "http" talks to a running
    `ns run` server (visible in nsflow); "direct" runs the network in this process instead."""
    logger.info("Opening session: agent=%s connection=%s host=%s port=%d", agent_name, connection, host, port)
    # use_direct governs how THIS network's own external-agent references (e.g. "/agent_network_editor")
    # get resolved. In "direct" mode there's no real server listening, so those must also resolve
    # in-process (True) -- with use_direct=False they'd try an actual HTTP call to host:port and
    # silently fail, leaving the agent with none of its own sub-tools.
    session = AgentSessionFactory().create_session(
        session_type=connection,
        agent_name=agent_name,
        hostname=host,
        port=port,
        use_direct=(connection == "direct"),
        metadata={"user_id": os.environ.get("USER", "network_consultant")},
    )
    thread = {
        "last_chat_response": None,
        "prompt": "",
        "timeout": 6000.0,
        "num_input": 0,
        "user_input": None,
        "sly_data": None,
        "chat_filter": {"chat_filter_type": "MAXIMAL"},
    }
    return session, thread


def _unwrap_json_error(response: str) -> str:
    """This network's own config sets error_formatter=json with error_fragments including
    "Error:" -- so whenever a response's text happens to contain "Error:" (e.g. relaying a
    sub-agent's tool-error verbatim, which is completely normal/expected here), neuro-san
    wraps the WHOLE response into {"error": "<escaped text>", "tool": ...}, often fenced in a
    ```json block. That JSON-escapes the original newlines into literal \\n, which breaks every
    line-based prefix check downstream (TOOL_ISSUE:, STRUCTURAL_CHANGE_REQUIRED:, etc., since
    none of them are at the start of a physical line anymore). Unwrap it back to plain text
    with real newlines whenever this envelope is detected; return the input unchanged otherwise.
    """
    if not response:
        return response
    text = response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json") :]
        text = text.strip()
    if not text.startswith("{"):
        return response
    try:
        parsed = json.loads(text)
    except ValueError:
        return response
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
        return parsed["error"]
    return response


def chat(session, thread: dict, message: str, sly_data: dict = None) -> tuple:
    """Send one message on an existing thread; returns (response_text, updated_thread)."""
    if sly_data:
        thread["sly_data"] = {**(thread.get("sly_data") or {}), **sly_data}
    os.makedirs(THINKING_DIR, exist_ok=True)
    processor = StreamingInputProcessor("DEFAULT", THINKING_FILE, session, THINKING_DIR)
    thread["user_input"] = message
    logger.info("chat -> sending message (%d chars)", len(message))
    started = time.time()
    thread = processor.process_once(thread)
    response = _unwrap_json_error(thread.get("last_chat_response"))
    logger.info(
        "chat <- response received (%.1fs, %d chars)", time.time() - started, len(response or "")
    )
    return response, thread


# Set by nsflow's backend job runner when this script is launched as a detached subprocess
# with no interactive stdin -- input() would just hang forever waiting for a terminal that
# doesn't exist. When set, clarification questions are exchanged via files in this directory
# instead (see _ask_headless).
NSFLOW_JOB_ID = os.environ.get("NSFLOW_JOB_ID")
NSFLOW_JOB_DIR = os.environ.get("NSFLOW_JOB_DIR")
HEADLESS_POLL_INTERVAL_SECONDS = 1.0


class _ProgressTracker:
    """Persist chart-ready test checkpoints for an nsflow-launched run.

    The runner frequently re-tests only the fixtures that were failing.  The chart still needs
    to show progress against the *whole* suite, so this tracker retains the last known state of
    untested fixtures and assigns newly passing fixtures to a new blue cohort on each iteration.
    A full-suite Before/After checkpoint resets that assumption with authoritative results.
    """

    def __init__(self):
        self.path = (
            os.path.join(NSFLOW_JOB_DIR, f"{NSFLOW_JOB_ID}.progress.jsonl")
            if NSFLOW_JOB_ID and NSFLOW_JOB_DIR
            else None
        )
        self.check_number = 0
        self.fixture_cohorts: dict[str, int] = {}
        self.next_cohort = 1

    def record(
        self,
        results: list[dict[str, Any]],
        checkpoint: str,
        total_fixture_count: int,
        improvement_iteration: Optional[int] = None,
    ) -> None:
        """Update cumulative state and, for nsflow jobs, append one complete checkpoint."""
        passing = {result["fixture"] for result in results if result.get("passed")}
        tested = {result["fixture"] for result in results}

        if checkpoint in {"generated", "before", "after"}:
            # These checkpoints always come from the complete suite and therefore replace every
            # inferred state left over from targeted re-tests.
            self.fixture_cohorts = {fixture: 0 for fixture in passing}
            self.next_cohort = 1
            segments = [len(passing)]
        else:
            cohort = self.next_cohort
            for fixture in tested:
                if fixture in passing:
                    if fixture not in self.fixture_cohorts:
                        self.fixture_cohorts[fixture] = cohort
                else:
                    self.fixture_cohorts.pop(fixture, None)
            self.next_cohort += 1
            segments = [
                sum(1 for fixture_cohort in self.fixture_cohorts.values() if fixture_cohort == cohort_index)
                for cohort_index in range(self.next_cohort)
            ]

        if not self.path:
            return
        self.check_number += 1
        entry = {
            "check": self.check_number,
            "checkpoint": checkpoint,
            "improvement_iteration": improvement_iteration,
            "passed": min(len(self.fixture_cohorts), total_fixture_count),
            "total": total_fixture_count,
            "segments": segments,
        }
        with open(self.path, "a", encoding="utf-8") as progress_file:
            progress_file.write(json.dumps(entry) + "\n")


def _ask_headless(question: str) -> str:
    """Write `question` to a file nsflow's backend surfaces in the UI, then block until a
    human answers it there (a file appears in the same directory), and return that answer."""
    question_path = os.path.join(NSFLOW_JOB_DIR, f"{NSFLOW_JOB_ID}.question.txt")
    answer_path = os.path.join(NSFLOW_JOB_DIR, f"{NSFLOW_JOB_ID}.answer.txt")
    with open(question_path, "w", encoding="utf-8") as question_file:
        question_file.write(question)
    try:
        while not os.path.exists(answer_path):
            time.sleep(HEADLESS_POLL_INTERVAL_SECONDS)
        with open(answer_path, encoding="utf-8") as answer_file:
            answer = answer_file.read().strip()
        os.remove(answer_path)
        return answer
    finally:
        if os.path.exists(question_path):
            os.remove(question_path)


CLARIFICATION_PREFIX = "NEEDS_CLARIFICATION:"
STRUCTURAL_CHANGE_PREFIX = "STRUCTURAL_CHANGE_REQUIRED:"
CONFIDENT_FIX_PREFIX = "CONFIDENT_FIX:"
RETEST_ONLY_PREFIX = "RETEST_ONLY:"
TOOL_ISSUE_PREFIX = "TOOL_ISSUE:"


def _write_tool_issues(tool_issues: list[str]) -> None:
    """Persist reported tool issues to a file nsflow's backend surfaces in the UI (mirrors
    _ask_headless's question file) -- a no-op when not running as an nsflow job."""
    if not (NSFLOW_JOB_ID and NSFLOW_JOB_DIR):
        return
    issues_path = os.path.join(NSFLOW_JOB_DIR, f"{NSFLOW_JOB_ID}.tool_issues.txt")
    with open(issues_path, "w", encoding="utf-8") as issues_file:
        issues_file.write("\n".join(tool_issues))


def _write_git_branch(branch: str) -> None:
    """Persist which branch --git-versions is committing this run's snapshots to, so nsflow's UI
    can surface it (mirrors _write_tool_issues) -- a no-op when not running as an nsflow job."""
    if not (NSFLOW_JOB_ID and NSFLOW_JOB_DIR):
        return
    branch_path = os.path.join(NSFLOW_JOB_DIR, f"{NSFLOW_JOB_ID}.git_branch.txt")
    with open(branch_path, "w", encoding="utf-8") as branch_file:
        branch_file.write(branch)


# --git-versions commits land here: <prefix>/<network>/<run-id>, never on whatever branch the
# person running this already has checked out.
GIT_VERSIONS_BRANCH_PREFIX = "consultant-versions"


def _start_git_versioning(network_name: str, run_id: str) -> Optional[str]:
    """Set up an isolated git worktree checked out to a dedicated
    consultant-versions/<network>/<run-id> branch, for committing/pushing a snapshot of the
    network's hocon file at each meaningful checkpoint -- without ever touching whatever branch
    or uncommitted changes the person running this already has checked out (no `git checkout`
    against the real working tree, ever). Returns the worktree's path, or None (after logging a
    warning) if this isn't inside a usable git repo -- versioning is then skipped for the rest of
    this run rather than failing it outright over a nice-to-have.
    """
    branch = f"{GIT_VERSIONS_BRANCH_PREFIX}/{network_name.replace('/', '-')}/{run_id}"
    worktree_dir = tempfile.mkdtemp(prefix="network_consultant_git_")
    try:
        subprocess.run(
            ["git", "worktree", "add", "-B", branch, worktree_dir, "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        logger.warning("--git-versions requested but could not set up a git worktree (%s); skipping.", detail)
        shutil.rmtree(worktree_dir, ignore_errors=True)
        return None
    logger.info("Versioning network hocon snapshots to git branch %r.", branch)
    _write_git_branch(branch)
    return worktree_dir


def _commit_hocon_version(worktree_dir: Optional[str], hocon_file: str, message: str) -> None:
    """Copy the network's current hocon content into the versioning worktree, commit it there if
    it differs from the branch's last commit, and push. A no-op if versioning was never started
    (worktree setup failed, or --git-versions wasn't passed). The "did it change" check compares
    content directly against the branch's own last commit (`git show HEAD:...`) rather than
    `git diff --cached --quiet` -- the latter trusts the working tree's file-stat cache to skip
    re-hashing, which can misjudge a file rewritten within the same on-disk mtime tick as its
    last stage (this loop's own checkpoints can land less than a second apart). Push/commit
    failures are logged and swallowed -- a rejected push or a network blip shouldn't take down
    the fix loop over this."""
    if worktree_dir is None:
        return
    relative_path = os.path.join("registries", hocon_file)
    with open(relative_path, encoding="utf-8") as source_file:
        new_content = source_file.read()
    last_committed = subprocess.run(
        ["git", "-C", worktree_dir, "show", f"HEAD:{relative_path}"], capture_output=True, text=True
    )
    if last_committed.returncode == 0 and last_committed.stdout == new_content:
        return  # Identical to the branch's last commit -- nothing new to save.
    dest_path = os.path.join(worktree_dir, relative_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as dest_file:
        dest_file.write(new_content)
    try:
        subprocess.run(["git", "-C", worktree_dir, "add", relative_path], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", worktree_dir, "commit", "-m", message], check=True, capture_output=True, text=True
        )
        subprocess.run(
            ["git", "-C", worktree_dir, "push", "-u", "origin", "HEAD"], check=True, capture_output=True, text=True
        )
        logger.info("Committed and pushed a version snapshot: %s", message)
    except subprocess.CalledProcessError as exc:
        logger.warning("Could not commit/push a version snapshot (%s); continuing without it.", exc.stderr or exc)


def _stop_git_versioning(worktree_dir: Optional[str]) -> None:
    """Remove the versioning worktree created by _start_git_versioning, if any. The branch itself
    (and everything committed to it) is left alone -- only the temporary checkout goes away."""
    if worktree_dir is None:
        return
    subprocess.run(["git", "worktree", "remove", "--force", worktree_dir], capture_output=True)


# One-shot cache: a UI-triggered "Generate Tests" run (max_iterations=0) tests the fresh
# fixtures once anyway, so a UI-triggered "Self-Improve" launched right after can reuse that
# result as its own iteration 1 instead of paying to re-run every fixture a second time. Gated
# on NSFLOW_JOB_ID/NSFLOW_JOB_DIR throughout -- plain CLI usage never writes or reads this cache,
# so it behaves exactly as before (max_iterations=0 does no test run at all).
GENTESTS_CACHE_DIR = "/tmp/network_consultant_gentests_cache"


def _gentests_cache_paths(network_name: str) -> tuple[str, str]:
    """(results_json_path, thinking_traces_dir) for this network's cached baseline, if any."""
    os.makedirs(GENTESTS_CACHE_DIR, exist_ok=True)
    safe_name = network_name.replace("/", "_")
    return (
        os.path.join(GENTESTS_CACHE_DIR, f"{safe_name}.json"),
        os.path.join(GENTESTS_CACHE_DIR, f"{safe_name}_thinking"),
    )


def _gentests_cache_fingerprint(network_name: str, hocon_path: str) -> str:
    """Hash of everything a test run's outcome for this network actually depends on: the
    network's own HOCON plus the current content of every one of its fixture files. A hocon-only
    hash would miss a fixture being added, edited, or deleted (e.g. by a fresh Generate Tests
    call, or a human editing tests/fixtures/ by hand) between the run that wrote this cache and
    the one that would consume it -- the fixtures wouldn't match what was actually tested, but
    the hocon hash alone would still say "unchanged". Hashing both means the cache is only ever
    reused when literally nothing that could change the result has moved since.
    """
    hasher = hashlib.sha256()
    with open(hocon_path, encoding="utf-8") as hocon_file:
        hasher.update(hocon_file.read().encode("utf-8"))
    for fixture_path in _fixture_paths(network_name):
        hasher.update(fixture_path.encode("utf-8"))
        with open(fixture_path, encoding="utf-8") as fixture_file:
            hasher.update(fixture_file.read().encode("utf-8"))
    return hasher.hexdigest()


def _save_gentests_cache(network_name: str, hocon_path: str, results: list) -> None:
    """Cache a generate-tests-only run's results, fingerprinted to the network's current content,
    for one-shot reuse by the next Self-Improve run against this exact network. Also copies each
    fixture's consolidated thinking trace (see IMPROVEMENT_THINKING_DIR above) -- without
    it, a self-improve run that skips its own re-test would leave the diagnosing sub-agents with
    only the bare assertion message instead of the full per-agent reasoning a fresh run gives
    them via read_thinking_trace."""
    results_path, thinking_dir = _gentests_cache_paths(network_name)
    tmp_path = f"{results_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as results_file:
        json.dump(
            {"fingerprint": _gentests_cache_fingerprint(network_name, hocon_path), "results": results}, results_file
        )
    os.replace(tmp_path, results_path)
    shutil.rmtree(thinking_dir, ignore_errors=True)
    if os.path.isdir(IMPROVEMENT_THINKING_DIR):
        shutil.copytree(IMPROVEMENT_THINKING_DIR, thinking_dir)


def _load_gentests_cache(network_name: str, hocon_path: str) -> list:
    """Return cached results (restoring their thinking traces into IMPROVEMENT_THINKING_DIR) if
    the network's hocon and every one of its fixtures still match what was cached; None otherwise
    (which means the caller must actually run the tests). Always consumes (deletes) the cache --
    a stale, mismatched, or already-used baseline is never reused, so at most the very next
    Self-Improve run after a Generate Tests run benefits."""
    results_path, thinking_dir = _gentests_cache_paths(network_name)
    if not os.path.exists(results_path):
        return None
    try:
        with open(results_path, encoding="utf-8") as results_file:
            cached = json.load(results_file)
    except (json.JSONDecodeError, OSError):
        cached = None
    os.remove(results_path)
    matches = cached is not None and cached.get("fingerprint") == _gentests_cache_fingerprint(network_name, hocon_path)
    if matches and os.path.isdir(thinking_dir):
        shutil.copytree(thinking_dir, IMPROVEMENT_THINKING_DIR, dirs_exist_ok=True)
    shutil.rmtree(thinking_dir, ignore_errors=True)
    return cached.get("results") if matches else None


def _restore_best_hocon(hocon_path: str, best_text: str, best_iteration: int) -> None:
    """Roll the network HOCON back to the version that produced the best result, discarding the
    later edits that never beat it. No-op if the file already is that version."""
    if best_text is None:
        return
    with open(hocon_path, encoding="utf-8") as current_file:
        if current_file.read() == best_text:
            return
    with open(hocon_path, "w", encoding="utf-8") as out_file:
        out_file.write(best_text)
    print(f"[network_consultant] Rolled {hocon_path} back to its iteration-{best_iteration} version "
          "(the best result seen); the edits after that one never improved on it.")


def _good_enough(passed: int, total: int) -> bool:
    """Whether a FULL-suite result clears the bar to stop fixing this network and move on."""
    return total > 0 and passed >= GOOD_ENOUGH_RATIO * total


def extract_prefixed(response: str, prefix: str) -> list[str]:
    """Return the payload of every line in `response` that starts with `prefix`."""
    return [
        line[len(prefix) :].strip() for line in (response or "").splitlines() if line.strip().startswith(prefix)
    ]

# Signature of consultant_editor retrying a tool call that can never succeed -- e.g. when the
# target network's HOCON uses a style (no root braces, "=" instead of ":") that
# SourcePreservingHoconEditor cannot parse. Left unhandled, this retries indefinitely.
PARSE_ERROR_MARKERS = ("could not be parsed", "Could not locate direct property")
PARSE_ERROR_REPEAT_THRESHOLD = 3


class _ParseErrorCapture(logging.Handler):
    """Watches for the recurring 'model output could not be parsed' signature during one
    chat() call, so a doomed retry loop can be recognized and stopped instead of run out."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if any(marker in message for marker in PARSE_ERROR_MARKERS):
            self.messages.append(message.strip())


class StuckPatchError(Exception):
    """Raised when consultant_editor is stuck retrying an unfixable tool-call parse error
    against a specific network HOCON file."""

    def __init__(self, hocon_file: str, messages: list[str]):
        self.hocon_file = hocon_file
        self.messages = messages
        super().__init__(
            f"consultant_editor is stuck patching {hocon_file} -- its source-preserving editor "
            "doesn't support this file's brace-less/'=' HOCON style. Skipping."
        )


def _guarded_chat(session, thread: dict, message: str, hocon_file: str, sly_data: dict = None) -> tuple:
    """chat(), but raises StuckPatchError if the parse-error signature repeats during the call
    instead of letting consultant_editor retry a doomed tool call indefinitely."""
    capture = _ParseErrorCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(capture)
    try:
        response, thread = chat(session, thread, message, sly_data=sly_data)
    finally:
        root_logger.removeHandler(capture)
    if len(capture.messages) >= PARSE_ERROR_REPEAT_THRESHOLD:
        raise StuckPatchError(hocon_file, capture.messages)
    return response, thread


def normalize_hocon_reference(value: str) -> str:
    """Return a safe registries-relative HOCON reference for network and fixture lookup."""
    normalized = value.strip().replace("\\", "/")
    if normalized.startswith("registries/"):
        normalized = normalized[len("registries/") :]
    path = PurePosixPath(normalized)
    if path.is_absolute() or path.suffix != ".hocon" or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("HOCON file must be a safe .hocon path relative to registries/.")
    return path.as_posix()


def consult(session, thread: dict, message: str, hocon_file: str, fixture_paths: dict[str, str]) -> tuple:
    """
    Send one diagnosis/follow-up message to consultant_editor, and keep answering any
    NEEDS_CLARIFICATION questions it comes back with (asking the actual person running this
    script) until it returns a turn with no more open questions.

    :return: (final_response_text, updated_thread)
    """
    logger.info("consult start: hocon_file=%s fixtures=%d", hocon_file, len(fixture_paths))
    response, thread = _guarded_chat(
        session,
        thread,
        message,
        hocon_file,
        sly_data={"agent_network_hocon_file": hocon_file, "test_fixture_paths": fixture_paths},
    )
    while True:
        questions = extract_prefixed(response, CLARIFICATION_PREFIX)
        if not questions:
            logger.info("consult done: no more open questions")
            return response, thread

        logger.info("consult: %d clarification question(s) raised", len(questions))
        print("[network_consultant] The consultant needs clarification before it can continue:")
        answers = []
        for question in questions:
            if NSFLOW_JOB_ID and NSFLOW_JOB_DIR:
                answer = _ask_headless(question)
            else:
                print(f"  ? {question}")
                answer = input("    your answer: ").strip()
            logger.info("consult: Q=%r A=%r", question, answer)
            answers.append(f"Q: {question}\nA: {answer}")

        follow_up = "This message answers the clarification question(s) you just asked:\n\n" + "\n\n".join(answers)
        response, thread = _guarded_chat(session, thread, follow_up, hocon_file)


def _consult_all_passing(session, thread: dict, direction: str, total_fixture_count: int, hocon_file: str) -> None:
    """Give consultant_editor one chance to act on `direction` (e.g. token reduction) even when
    there's nothing failing to fix -- otherwise the front man is never invoked at all, and its
    "run token_reduction_advisor even with no failures" instruction never gets a chance to fire."""
    if not direction:
        return
    try:
        response, _ = consult(session, thread, all_passing_prompt(direction, total_fixture_count), hocon_file, {})
        logger.info("consultant_editor response: %s", response)
    except StuckPatchError as exc:
        logger.error(str(exc))
        _write_tool_issues([str(exc)])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("consultant_editor call failed unexpectedly: %s: %s", type(exc).__name__, exc)


def _fixture_paths(network_name: str) -> list[str]:
    """Sorted paths of every generated fixture under tests/fixtures/<network_name>/, if any.
    Sorted so a fingerprint hashing these in order is stable regardless of directory listing
    order."""
    search_dir = os.path.join("tests", "fixtures", network_name)
    return sorted(glob.glob(os.path.join(search_dir, "*.hocon")))


def has_existing_fixtures(network_name: str) -> bool:
    """Whether tests/fixtures/<network_name>/ already has any generated fixture."""
    return bool(_fixture_paths(network_name))


def diagnosis_prompt(failures: list, direction: str, total_fixture_count: int, is_subset_check: bool) -> str:
    """Builds the failing-test report handed to consultant_editor, including each fixture's
    current content -- needed since it may decide to correct the fixture itself, not just the
    network's instructions.

    :param total_fixture_count: How many fixtures exist in the network's full suite.
    :param is_subset_check: Whether this round only re-checked a subset (the fixtures still
        failing last round) instead of running the full suite.
    """
    lines = ["User's intended behavior and approximate vision:", direction, ""]
    if is_subset_check:
        lines.append(
            f"This round only re-checked {len(failures)} of the {total_fixture_count} total fixtures in the "
            "suite (the ones still failing last round) -- not a full run. The following are failing:"
        )
    else:
        lines.append(f"The full suite of {total_fixture_count} fixtures was run. The following are failing:")
    lines.append("")
    for failure in failures:
        with open(failure["path"], encoding="utf-8") as fixture_file:
            fixture_content = fixture_file.read()
        lines.append(f"### Fixture file: {failure['fixture']}")
        lines.append(f"Failure: {failure['message'].strip()}")
        lines.append("Current fixture content:")
        lines.append(fixture_content.strip())
        lines.append("")
    lines.append(
        "By default, next round only re-checks whichever fixtures are still failing after your fix -- not the "
        "full suite. If you want a different set re-checked next round instead (e.g. only a specific one you're "
        "unsure about, while skipping others you don't expect to have changed), you may output "
        "`RETEST_ONLY: <fixture>` lines (one per fixture) to override the default. Omit these lines entirely to "
        "just accept the default (re-check exactly what's still failing)."
    )
    return "\n".join(lines)


def all_passing_prompt(direction: str, total_fixture_count: int) -> str:
    """Report handed to consultant_editor when every fixture already passes -- there's nothing
    to fix, but the user's direction (e.g. "reduce token usage") may still call for the
    token_reduction_advisor pass, which only ever runs if the front man is actually invoked."""
    return (
        "User's intended behavior and approximate vision:\n"
        f"{direction}\n\n"
        f"All {total_fixture_count} fixtures in the test suite are currently passing. There are no failures to "
        "fix. If the direction above calls for something to still be done (e.g. reducing token usage), do that "
        "now; otherwise say plainly that there is nothing to do."
    )


def main():  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """Run the iterative generate, test, diagnose, and repair workflow."""
    # Root stays at WARNING so third-party loggers (neuro-san's manifest loading, etc.) don't
    # flood the output -- only this app's own loggers are bumped to INFO.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    logger.setLevel(logging.INFO)
    # Re-announces every "serve": false manifest entry (26 of them) at WARNING level, on every
    # session open -- real but useless noise for this tool, drowning out our own progress logs.
    logging.getLogger("ServedManifestConfigFilter").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-case", help="Use-case description for a brand new network.")
    parser.add_argument("--hocon-file", help="Existing network hocon (relative to registries/) to iterate on instead.")
    parser.add_argument(
        "--direction",
        help="Intended behavior for an existing network; helps distinguish network defects from bad tests.",
    )
    parser.add_argument("--test-level", default="normal", choices=["minimum", "normal", "max"])
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--success-ratio",
        default=CONFIDENT_SUCCESS_RATIO,
        help=f"Ratio (e.g. 'N/M') a fixture is bumped to once consultant_editor is CONFIDENT its fix holds up "
        f"under repeated runs (default: {CONFIDENT_SUCCESS_RATIO}). Everything else stays cheap.",
    )
    parser.add_argument(
        "--connection",
        default="direct",
        choices=["http", "direct"],
        help="'direct' (default) runs the network in this process, no server needed. "
        "'http' talks to an already-running `ns run` server instead.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="`ns run` server host (--connection http only).")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="`ns run` server port (--connection http only)."
    )
    parser.add_argument(
        "--git-versions",
        action="store_true",
        help="Commit the network hocon to a dedicated "
        f"{GIT_VERSIONS_BRANCH_PREFIX}/<network>/<run-id> branch and push it to origin at each "
        "meaningful test checkpoint (baseline, each retest, final confirmation), so every "
        "version tried is preserved in git history. Off by default -- this pushes to your "
        "'origin' remote repeatedly during the run, so only enable it when you actually want that.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"\d+/\d+", args.success_ratio):
        parser.error(f"--success-ratio must look like 'N/M' (e.g. '3/3'), got {args.success_ratio!r}.")
    if not args.use_case and not args.hocon_file:
        parser.error("Provide --use-case (to create a network) or --hocon-file (to iterate on an existing one).")
    if args.hocon_file and not args.direction:
        parser.error(
            "--direction is required with --hocon-file so test defects are not guessed from current behavior."
        )
    if args.hocon_file:
        try:
            args.hocon_file = normalize_hocon_reference(args.hocon_file)
        except ValueError as exc:
            parser.error(str(exc))

    consultant_session, consultant_thread = open_session(
        "agent_network_consultant", args.connection, args.host, args.port
    )

    hocon_file = args.hocon_file
    if not hocon_file:
        logger.info("Designing a new network (use_case=%r)...", args.use_case)
        designer_session, designer_thread = open_session(
            "agent_network_designer", args.connection, args.host, args.port
        )
        response, designer_thread = chat(designer_session, designer_thread, args.use_case)
        network_name = (designer_thread.get("sly_data") or {}).get("agent_network_name")
        logger.info("Designer response: %s", response)
        if not network_name:
            logger.error(
                "Designer did not return an agent_network_name; cannot continue. Its response may explain why:\n%s",
                response,
            )
            return
        try:
            hocon_file = normalize_hocon_reference(f"generated/{network_name}.hocon")
        except ValueError as exc:
            logger.error("Designer returned an unsafe agent_network_name (%r): %s", network_name, exc)
            return
    network_name = os.path.splitext(hocon_file)[0]
    direction = args.direction or args.use_case
    logger.info("Target network: %s (hocon_file=%s)", network_name, hocon_file)
    # This is a fresh run, not a continuation of a prior one -- clear any scratchpad notes
    # network_behavior_fixer left behind last time so they don't leak into this run. Left alone
    # for the rest of main() so it persists across this run's own iterations below.
    clear_for_hocon_file(hocon_file)
    # Same reasoning for consolidated thinking traces: wipe last run's leftovers so a diagnosing
    # sub-agent can never read a stale trace for a fixture this run hasn't gotten to yet.
    shutil.rmtree(IMPROVEMENT_THINKING_DIR, ignore_errors=True)

    if has_existing_fixtures(network_name):
        logger.info("Existing test fixtures found for %s; skipping ANTeGen.", network_name)
    else:
        logger.info("Generating tests (ANTeGen, test_level=%s)...", args.test_level)
        testgen_session, testgen_thread = open_session(
            "agent_network_test_generator", args.connection, args.host, args.port
        )
        response, testgen_thread = chat(
            testgen_session,
            testgen_thread,
            f"Generate test cases for {network_name} with {args.test_level} coverage",
        )
        logger.info("ANTeGen response: %s", response)

    original_ratios: dict[str, str] = {}
    progress_tracker = _ProgressTracker()
    # None = run the full suite; otherwise a list of basenames to re-check cheaply instead of
    # paying for every fixture every round.
    retest_only = None
    total_fixture_count = None
    improvement_iteration = 0
    git_worktree = None
    try:
        best_failure_count = None
        stale_rounds = 0
        # hocon_file is registries-relative; the snapshot below is the file text that produced
        # the best score so far, so a plateau can roll the later dead-end edits back off.
        hocon_path = os.path.join("registries", hocon_file)
        best_hocon_text = None
        best_hocon_iteration = None

        if args.max_iterations == 0:
            # Generate Tests, triggered from the UI: run the fresh fixtures once and cache the
            # result so a Self-Improve run started right after doesn't pay to re-run every
            # fixture a second time. Plain CLI usage (no nsflow job env vars) keeps the old
            # behavior exactly: generate and stop, no test run at all.
            if NSFLOW_JOB_ID and NSFLOW_JOB_DIR:
                logger.info("Baseline check (Generate Tests, no fix loop)...")
                results = run_all_tests(network_name)
                _save_gentests_cache(network_name, hocon_path, results)
                failures = [r for r in results if not r["passed"]]
                progress_tracker.record(results, "generated", len(results))
                logger.info("Baseline: %d/%d passing.", len(results) - len(failures), len(results))
            return

        git_worktree = (
            _start_git_versioning(network_name, NSFLOW_JOB_ID or time.strftime("%Y%m%d-%H%M%S"))
            if args.git_versions
            else None
        )

        for iteration in range(1, args.max_iterations + 1):
            logger.info("--- Iteration %d/%d: running tests ---", iteration, args.max_iterations)
            is_subset_check = retest_only is not None
            cached_results = (
                _load_gentests_cache(network_name, hocon_path)
                if iteration == 1 and NSFLOW_JOB_ID and NSFLOW_JOB_DIR
                else None
            )
            if cached_results is not None:
                logger.info("Reusing the Generate Tests baseline (network unchanged since) -- skipping re-test.")
                results = cached_results
            else:
                results = run_all_tests(network_name, only_fixtures=retest_only)
            if not is_subset_check:
                total_fixture_count = len(results)
            infrastructure_errors = [result for result in results if result.get("infrastructure_error")]
            if infrastructure_errors:
                logger.error("Test infrastructure failed; no network or fixture changes were attempted:")
                for error in infrastructure_errors:
                    logger.error("  - %s: %s", error["fixture"], error["message"])
                return
            if iteration == 1:
                progress_tracker.record(results, "before", total_fixture_count)
            else:
                improvement_iteration += 1
                progress_tracker.record(results, "iteration", total_fixture_count, improvement_iteration)
            failures = [r for r in results if not r["passed"]]
            logger.info(
                "%d/%d fixtures passing%s.",
                len(results) - len(failures),
                len(results),
                " (subset re-check)" if retest_only is not None else "",
            )
            _commit_hocon_version(
                git_worktree,
                hocon_file,
                f"{'Before' if iteration == 1 else f'Iteration {iteration}'}: "
                f"{len(results) - len(failures)}/{len(results)} passing"
                f"{' (subset re-check)' if is_subset_check else ''}",
            )
            if not failures:
                if retest_only is not None:
                    logger.info("Subset re-check passed; running full suite once to confirm no regressions...")
                    results = run_all_tests(network_name)
                    total_fixture_count = len(results)
                    infrastructure_errors = [result for result in results if result.get("infrastructure_error")]
                    if infrastructure_errors:
                        logger.error("Full-suite confirmation could not complete because test infrastructure failed.")
                        for error in infrastructure_errors:
                            logger.error("  - %s: %s", error["fixture"], error["message"])
                        return
                    failures = [r for r in results if not r["passed"]]
                    progress_tracker.record(results, "after", total_fixture_count)
                    _commit_hocon_version(
                        git_worktree,
                        hocon_file,
                        f"Iteration {iteration} (full-suite confirmation): "
                        f"{total_fixture_count - len(failures)}/{total_fixture_count} passing",
                    )
                    if failures:
                        passed_count = total_fixture_count - len(failures)
                        if _good_enough(passed_count, total_fixture_count):
                            print(
                                f"[network_consultant] {passed_count}/{total_fixture_count} passing "
                                f"(>= {GOOD_ENOUGH_RATIO:.0%}) on the full suite -- good enough, moving on."
                            )
                            for failure in failures:
                                logger.info(
                                    "  - still failing: %s: %s", failure["fixture"], failure["message"].strip()
                                )
                            return
                        # Not good enough -- regressions elsewhere in the suite; keep going against those.
                        retest_only = None
                        is_subset_check = False

                if not failures:
                    logger.info("All tests passing. Network is satisfiable.")
                    with open(hocon_path, encoding="utf-8") as before_file:
                        hocon_before_consult = before_file.read()
                    _consult_all_passing(
                        consultant_session, consultant_thread, direction, total_fixture_count, hocon_file
                    )
                    with open(hocon_path, encoding="utf-8") as after_file:
                        hocon_after_consult = after_file.read()
                    if hocon_after_consult == hocon_before_consult:
                        logger.info(
                            "consultant_editor made no changes; already all passing, nothing to re-verify. Stopping."
                        )
                        progress_tracker.record(results, "after", total_fixture_count)
                        return
                    logger.info("Re-running full suite to verify that change didn't break anything...")
                    results = run_all_tests(network_name)
                    total_fixture_count = len(results)
                    infrastructure_errors = [result for result in results if result.get("infrastructure_error")]
                    if infrastructure_errors:
                        logger.error("Final confirmation could not complete because test infrastructure failed.")
                        for error in infrastructure_errors:
                            logger.error("  - %s: %s", error["fixture"], error["message"])
                        return
                    failures = [r for r in results if not r["passed"]]
                    progress_tracker.record(results, "after", total_fixture_count)
                    _commit_hocon_version(
                        git_worktree,
                        hocon_file,
                        f"After: {total_fixture_count - len(failures)}/{total_fixture_count} passing",
                    )
                    if not failures:
                        logger.info("Still all passing after verification. Stopping.")
                        return
                    passed_count = total_fixture_count - len(failures)
                    if _good_enough(passed_count, total_fixture_count):
                        print(
                            f"[network_consultant] {passed_count}/{total_fixture_count} passing "
                            f"(>= {GOOD_ENOUGH_RATIO:.0%}) on the full suite -- good enough, moving on."
                        )
                        for failure in failures:
                            logger.info("  - still failing: %s: %s", failure["fixture"], failure["message"].strip())
                        return
                    logger.warning(
                        "That change introduced %d regression(s); continuing to fix them instead of stopping.",
                        len(failures),
                    )
                    retest_only = None
                    is_subset_check = False

            # A targeted re-test's `failures` list is smaller than the full suite by design;
            # compare plateau progress using the tracker's cumulative whole-suite state so an
            # After check and the following iteration remain comparable.
            effective_failure_count = total_fixture_count - len(progress_tracker.fixture_cohorts)
            improved = best_failure_count is None or effective_failure_count < best_failure_count
            if best_failure_count is not None and effective_failure_count >= best_failure_count:
                stale_rounds += 1
            else:
                stale_rounds = 0
            best_failure_count = (
                min(effective_failure_count, best_failure_count)
                if best_failure_count is not None
                else effective_failure_count
            )
            if improved:
                # Snapshot the HOCON that produced this best-so-far score -- read now, before the
                # editor touches it again, so a later plateau can roll the useless edits back off.
                # Iteration 1 snapshots the original, which is the right floor to fall back to.
                with open(hocon_path, encoding="utf-8") as best_file:
                    best_hocon_text = best_file.read()
                best_hocon_iteration = iteration

            if stale_rounds >= PLATEAU_STRIKES:
                print(
                    f"[network_consultant] Tried hard for {iteration} rounds, but this isn't working -- "
                    f"{len(failures)}/{len(results)} fixtures still failing. Giving up here."
                )
                logger.warning(
                    "No improvement for %d consecutive rounds. Stopping with %d/%d still failing:",
                    PLATEAU_STRIKES,
                    len(failures),
                    len(results),
                )
                for failure in failures:
                    logger.warning("  - %s: %s", failure["fixture"], failure["message"].strip())
                _restore_best_hocon(hocon_path, best_hocon_text, best_hocon_iteration)
                logger.info("Re-running the full suite on the restored best version for the final After checkpoint...")
                results = run_all_tests(network_name)
                infrastructure_errors = [result for result in results if result.get("infrastructure_error")]
                if not infrastructure_errors:
                    progress_tracker.record(results, "after", len(results))
                else:
                    logger.error("Final confirmation could not complete because test infrastructure failed.")
                return

            logger.info("Consulting consultant_editor to fix failing agents' instructions...")
            failure_fixture_paths = {failure["fixture"]: failure["path"] for failure in failures}
            try:
                response, consultant_thread = consult(
                    consultant_session,
                    consultant_thread,
                    diagnosis_prompt(failures, direction, total_fixture_count, is_subset_check),
                    hocon_file,
                    failure_fixture_paths,
                )
            except StuckPatchError as exc:
                logger.error(str(exc))
                _write_tool_issues([str(exc)])
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("consultant_editor call failed unexpectedly: %s: %s", type(exc).__name__, exc)
                _write_tool_issues([f"{type(exc).__name__}: {exc}"])
                return
            logger.info("consultant_editor response: %s", response)
            # Commit/push the edit itself the moment it's made -- not the checkpoint AFTER the
            # next test run confirms it. Waiting for that confirmation is why the first-ever
            # edit (made here, while `iteration` is still 1) used to only show up in git once
            # iteration 2's test round ran, one full round later than the edit that produced it.
            _commit_hocon_version(
                git_worktree, hocon_file, f"Change {iteration}: fixing {len(failures)} failing fixture(s)"
            )
            if any(line.strip().startswith(STRUCTURAL_CHANGE_PREFIX) for line in (response or "").splitlines()):
                logger.warning("A structural change requires explicit Designer review. Stopping safely.")
                return

            tool_issues = extract_prefixed(response, TOOL_ISSUE_PREFIX)
            if tool_issues:
                print("[network_consultant] A required coded tool is broken -- this needs a human code fix, not "
                      "an instructions/fixture change. Stopping so you can fix it and re-run:")
                for issue in tool_issues:
                    print(f"  ! {issue}")
                logger.warning("Tool issue(s) reported; stopping for a human fix: %s", tool_issues)
                _write_tool_issues(tool_issues)
                return

            confident_fixtures = extract_prefixed(response, CONFIDENT_FIX_PREFIX)
            if confident_fixtures:
                new_originals = set_success_ratio_for_fixtures(network_name, confident_fixtures, args.success_ratio)
                original_ratios.update(new_originals)
                logger.info(
                    "consultant_editor is confident in %d fix(es); bumped to %s for next round: %s",
                    len(confident_fixtures),
                    args.success_ratio,
                    confident_fixtures,
                )

            # Next round, only re-check what we just worked on -- cheap, targeted re-verification
            # instead of the whole suite. A full sweep still runs once before declaring success.
            # consultant_editor can override this default via explicit RETEST_ONLY: lines.
            requested_retest = extract_prefixed(response, RETEST_ONLY_PREFIX)
            if requested_retest:
                retest_only = requested_retest
                logger.info("consultant_editor requested a specific retest set: %s", retest_only)
            else:
                retest_only = [failure["fixture"] for failure in failures]

        logger.warning("Reached max iterations (%d) without a full pass.", args.max_iterations)
        _restore_best_hocon(hocon_path, best_hocon_text, best_hocon_iteration)
        logger.info("Re-running the full suite on the restored best version for the final After checkpoint...")
        results = run_all_tests(network_name)
        infrastructure_errors = [result for result in results if result.get("infrastructure_error")]
        if not infrastructure_errors:
            progress_tracker.record(results, "after", len(results))
        else:
            logger.error("Final confirmation could not complete because test infrastructure failed.")
    finally:
        logger.info("Restoring original success_ratio values...")
        restore_success_ratios(original_ratios)
        _stop_git_versioning(git_worktree)


if __name__ == "__main__":
    main()
