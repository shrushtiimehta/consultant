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
network_consultant's fixture-running engine: run every ANTeGen-generated test fixture for a
network and report pass/fail per fixture, without going through pytest. Reuses neuro_san's own
data-driven test driver -- the same one `make test-integration` uses -- so results match
exactly what CI would report.

Imported by run.py, which adds the consult-and-fix loop on top. The dependency only points
that way: nothing here knows the loop exists, so a caller that just wants to run a suite pays
for none of it.

The direct-session environment defaults below live in THIS module rather than in run.py
because they must be set before neuro-san is imported, and this is the module both sides
import first.
"""

import glob
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest import TestCase

import neuro_san_studio

# This repo has no local neuro_san_studio/ source tree -- it only depends on neuro-san-studio
# as an installed package -- so toolbox_info.hocon paths must be resolved to wherever pip/uv
# put it, not assumed relative to this repo's root.
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

# These imports intentionally follow the direct-session environment defaults above: neuro-san
# reads AGENT_MANIFEST_FILE and friends at import time, so setting them afterwards is too late.
# wrong-import-order is disabled for the same reason -- neuro_san_studio has to be imported
# before them to locate the toolbox, which no sort order allows.
# pylint: disable=wrong-import-position,wrong-import-order
from leaf_common.time.timeout_reached_exception import TimeoutReachedException  # noqa: E402
from neuro_san.test.driver.data_driven_agent_test_driver import DataDrivenAgentTestDriver  # noqa: E402
from neuro_san.test.unittest.unit_test_assert_forwarder import UnitTestAssertForwarder  # noqa: E402

# Not __name__: run.py runs as "__main__" via `python -m`, and both halves logging under one
# name keeps the job log a single readable stream for nsflow to tail.
logger = logging.getLogger("network_consultant")

# Ceiling on how many fixtures run at once. Each one nests its own pool underneath -- neuro-san's
# DataDrivenAgentTestDriver.one_test runs a fixture's success_ratio iterations in parallel too --
# so peak live agent sessions is this times the ratio, and an uncapped 15-fixture suite at 3/3
# was 45 of them: straight into provider rate limits, which surface here as fixture failures and
# send consultant off to "fix" a network that was fine.
MAX_PARALLEL_FIXTURES = 7
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

# Set by nsflow's backend job runner when this script is launched as a detached subprocess
# with no interactive stdin -- input() would just hang forever waiting for a terminal that
# doesn't exist. When set, clarification questions are exchanged via files in this directory
# instead (see _ask_headless).
NSFLOW_JOB_ID = os.environ.get("NSFLOW_JOB_ID")
NSFLOW_JOB_DIR = os.environ.get("NSFLOW_JOB_DIR")


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

    # A fixture with success_ratio > 1 (e.g. "3/3") runs several iterations of the SAME
    # interactions concurrently (see DataDrivenAgentTestDriver.one_test), each iteration
    # getting its own thinking directories, named "..._<iteration_index>". Consolidating every
    # iteration would glue N near-identical retries together under one section per agent --
    # wasted tokens, and it obscures which attempt a diagnosing agent is even looking at. Keep
    # only the earliest iteration's directories (every turn of it, since turns of the SAME
    # iteration share that suffix) -- a failing fixture almost always fails the same way across
    # iterations, so one is representative. A single-iteration fixture (no suffix at all) is
    # unaffected: every directory shares the same (empty) key below.
    def _iteration_of(run_dir: str) -> str:
        match = re.search(r"_(\d+)$", os.path.basename(run_dir))
        return match.group(1) if match else ""

    first_iteration = _iteration_of(run_dirs[0])
    run_dirs = [d for d in run_dirs if _iteration_of(d) == first_iteration]

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
    logger.info(
        "set_success_ratios(%s, %s): changed %d/%d fixtures",
        fixtures_dir,
        ratio,
        len(originals),
        len(fixture_paths(fixtures_dir)),
    )
    return originals


def restore_success_ratios(originals: dict[str, str]) -> None:
    """Undo set_success_ratios, restoring each fixture's original success_ratio.

    Never raises, and never gives up partway. This runs from main()'s `finally` and from the
    SIGTERM handler, where an exception on one fixture would abandon every fixture after it
    still bumped -- and, from the `finally`, would replace whatever actually went wrong with a
    confusing error from the cleanup path. A fixture can legitimately be gone by now (deleted
    while the run was in flight), and a missing file needs no restoring anyway.
    """
    restored = 0
    for path, ratio in originals.items():
        try:
            with open(path, encoding="utf-8") as fixture_file:
                text = fixture_file.read()
            with open(path, "w", encoding="utf-8") as fixture_file:
                fixture_file.write(SUCCESS_RATIO_PATTERN.sub(rf"\g<1>{ratio}\g<3>", text, count=1))
        except OSError as error:
            logger.warning("Could not restore success_ratio %s on %s: %s", ratio, path, error)
            continue
        restored += 1
        logger.info("success_ratio restored -> %s: %s", ratio, path)
    logger.info("restore_success_ratios: restored %d/%d fixture(s)", restored, len(originals))


def _run_real_ratio_suite(fixtures_dir: str, original_ratios: dict[str, str], ratio: str) -> list[dict[str, Any]]:
    """Run the full suite for an authoritative Before/After bar. Those bars always score against
    each fixture's own success_ratio -- that is the network's real score -- so lift any
    CONFIDENT_FIX bump for the duration of the run, then put it straight back: the loop often
    continues afterwards, and a fix the consultant vouched for should stay on its stricter ratio
    when it does.

    `original_ratios` is updated in place, so it keeps tracking exactly what needs undoing.
    """
    bumped_paths = list(original_ratios)
    restore_success_ratios(original_ratios)
    original_ratios.clear()
    try:
        return run_all_tests(fixtures_dir)
    finally:
        original_ratios.update(_set_success_ratio_for_paths(bumped_paths, ratio))


class _ApiKeyErrorCapture(logging.Handler):
    """Catches neuro-san's own logged API-key errors, which it otherwise only logs and
    silently falls back from -- never raising an exception a caller could catch.

    A handler is added to the shared root logger, so with run_all_tests running fixtures
    concurrently, every concurrently-running fixture's handler would otherwise see every OTHER
    fixture's log records too -- filtering to this handler's own thread keeps one fixture's
    capture from picking up another's API-key error."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []
        self._thread_id = threading.get_ident()

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self._thread_id:
            return
        message = record.getMessage()
        if API_KEY_ERROR_MARKER in message:
            self.messages.append(message.strip())


class _ScorecardAssertForwarder(UnitTestAssertForwarder):
    """Records every acceptance criterion a fixture checks, not just the first that failed.

    neuro-san evaluates all of them, then DataDrivenAgentTestDriver.one_test raises from
    `asserts[0]` and discards the rest -- so a consultant fixing a fixture saw one failure of
    several and only learned about the next a whole round later. Observed on cpg_agents:
    `'owners' not found`, fixed, then `'KPIs' not found` the following round, when three
    criteria had been failing from the start. It was also never told which criteria already
    PASS, so nothing stopped a fix from trading one for another.

    The hook is free: AssertCapture wraps this object as its `basis` and calls straight through
    it for every criterion, so overriding the evaluators' assert methods sees all of them.

    Tallied per criterion rather than grouped per iteration on purpose. A fixture's
    success_ratio runs its iterations in parallel over a shared pool, so a thread can serve
    two iterations and thread identity does not identify one -- and counting attempts is the
    more useful answer anyway: "met on 1 of 3 attempts" is a flaky criterion, which reads
    identically to a hard failure when only the first exception survives.
    """

    def __init__(self, test_case: TestCase):
        super().__init__(test_case)
        # criterion -> [met, attempts]. Insertion-ordered, so a report reads in fixture order.
        self._tally: dict[str, list[int]] = {}
        # `entry[0] += 1` is load-add-store, not atomic: without this, parallel iterations of
        # the same fixture drop counts.
        self._lock = threading.Lock()

    def _record(self, criterion: str, method: str, *args) -> None:
        """Run one criterion's real assertion via the base class, tallying it either way.

        The parent method is resolved here, in an ordinary method body, so plain zero-argument
        `super()` works. Passing a name rather than a callable is what makes that possible: a
        lambda has no `__class__` cell, so `super()` inside one has to be spelled out with the
        class -- which then silently binds to the wrong class if this one is ever renamed or
        subclassed.

        :param criterion: How this check should read in a report
        :param method: The AssertForwarder method to delegate to
        :param args: That method's arguments, forwarded unchanged
        """
        try:
            getattr(super(), method)(*args)
        except AssertionError:
            self._note(criterion, met=False)
            raise  # AssertCapture still needs the exception to collect it
        self._note(criterion, met=True)

    def _note(self, criterion: str, met: bool) -> None:
        with self._lock:
            entry = self._tally.setdefault(criterion, [0, 0])
            entry[0] += int(met)
            entry[1] += 1

    def criteria_counts(self) -> tuple[int, int]:
        """:return: (criteria met on every attempt, criteria checked).

        "Met on every attempt" rather than "met at least once": a criterion that holds only
        sometimes is not one the network can be said to satisfy, and counting it as met would
        let flakiness read as progress -- the exact confusion the per-attempt tally exists to
        expose.
        """
        with self._lock:
            total = len(self._tally)
            met = sum(1 for met_count, attempts in self._tally.values() if met_count == attempts)
        return met, total

    def scorecard(self) -> list[tuple[str, int, int]]:
        """:return: (criterion, times met, times attempted), in the order first checked."""
        with self._lock:
            return [(criterion, met, attempts) for criterion, (met, attempts) in self._tally.items()]

    # --- the ten stock tests, per neuro_san/test/evaluators/ ----------------------------------
    # Method names are neuro-san's AssertForwarder interface, so they keep its camelCase.
    def assertGist(self, gist, acceptance_criteria, text_sample, msg=None):  # noqa: N802
        self._record(acceptance_criteria, "assertGist", gist, acceptance_criteria, text_sample, msg)

    def assertNotGist(self, gist, acceptance_criteria, text_sample, msg=None):  # noqa: N802
        self._record(f"NOT: {acceptance_criteria}", "assertNotGist", gist, acceptance_criteria, text_sample, msg)

    def assertIn(self, member, container, msg=None):  # noqa: N802
        self._record(f"contains {member!r}", "assertIn", member, container, msg)

    def assertNotIn(self, member, container, msg=None):  # noqa: N802
        self._record(f"does not contain {member!r}", "assertNotIn", member, container, msg)

    def assertEqual(self, first, second, msg=None):  # noqa: N802
        # gist_agent_evaluator asserts `assertEqual(only_one, True)` to catch an equivocating
        # judge. That is an internal sanity check, not one of the fixture's criteria.
        if isinstance(first, bool) and second is True:
            super().assertEqual(first, second, msg)
            return
        self._record(f"equals {first!r}", "assertEqual", first, second, msg)

    def assertNotEqual(self, first, second, msg=None):  # noqa: N802
        self._record(f"does not equal {first!r}", "assertNotEqual", first, second, msg)

    def assertGreater(self, first, second, msg=None):  # noqa: N802
        self._record(f"greater than {first!r}", "assertGreater", first, second, msg)

    def assertGreaterEqual(self, first, second, msg=None):  # noqa: N802
        self._record(f"not less than {first!r}", "assertGreaterEqual", first, second, msg)

    def assertLess(self, first, second, msg=None):  # noqa: N802
        self._record(f"less than {first!r}", "assertLess", first, second, msg)

    def assertLessEqual(self, first, second, msg=None):  # noqa: N802
        self._record(f"not greater than {first!r}", "assertLessEqual", first, second, msg)


def _scorecard_message(cause: BaseException, scorecard: list[tuple[str, int, int]]) -> str:
    """Turn a fixture's criteria into a report naming EVERY failure, not just the first.

    neuro-san checks all of them and then re-raises only asserts[0], so a consultant fixing a
    fixture used to see one failure of several and learn about the next a whole round later.

    Only the failures are listed. The count still says how many criteria there were in total
    ("2 of 6"), so the scale is visible without the message carrying -- and persisting -- the
    full text of everything that already works.
    """
    if not scorecard:
        return str(cause)

    failing = [(name, met, total) for name, met, total in scorecard if met < total]
    if not failing:
        return str(cause)

    lines = [f"{len(failing)} of {len(scorecard)} acceptance criteria failing:"]
    for name, met, total in failing:
        # "met on 1 of 3" is a flaky criterion, not a broken one -- rewriting an agent that is
        # already right a third of the time makes it worse, and the bump to a stricter ratio
        # then turns a 1-in-3 into a 1-in-27.
        suffix = f"  (met on {met} of {total} attempts)" if met else ""
        lines.append(f"  - {name}{suffix}")
    lines.append("")
    lines.append(f"First failure in detail:\n{cause}")
    return "\n".join(lines)


def run_fixture(fixture_path: str) -> dict[str, Any]:
    """
    :param fixture_path: Path to a single test fixture HOCON file.
    :return: Result with fixture/path/passed/message/infrastructure_error fields.
    """
    # one_test() raises a single AssertionError (summarizing every interaction/iteration
    # mismatch) only if the fixture's success_ratio wasn't met, so a plain try/except
    # is all the aggregation this needs.
    fixture_name = os.path.basename(fixture_path)
    asserts = _ScorecardAssertForwarder(TestCase())
    driver = DataDrivenAgentTestDriver(asserts, test_name=fixture_name)

    def verdict(passed: bool, message: str | None, infrastructure_error: bool = False) -> dict[str, Any]:
        """Build this fixture's result, so every exit reports the same shape.

        An infrastructure error keeps zero criteria: a timeout or an API-key fault never reached
        a verdict, so it must not look like -- or be SCORED as -- a round in which every
        criterion happened to fail. Stating that here makes it a rule; previously it held only
        because three of the five exits happened to leave the counts out.
        """
        met, total = (0, 0) if infrastructure_error else asserts.criteria_counts()
        return {
            "fixture": fixture_name,
            "path": fixture_path,
            "passed": passed,
            "message": message,
            "infrastructure_error": infrastructure_error,
            "criteria_passed": met,
            "criteria_total": total,
        }

    logger.info("run_fixture start: %s", fixture_name)
    started = time.time()
    capture = _ApiKeyErrorCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(capture)
    try:
        driver.one_test(fixture_path)
        logger.info("run_fixture pass (%.1fs): %s", time.time() - started, fixture_name)
        return verdict(True, None)
    except AssertionError as exc:
        cause = exc.__cause__ or exc
        if capture.messages:
            api_key_summary = "\n".join(capture.messages)
            logger.warning(
                "run_fixture infrastructure_error (%.1fs, API key error): %s", time.time() - started, fixture_name
            )
            return verdict(
                False,
                f"{api_key_summary}\n\n(Original assertion, likely a symptom of the above: {cause})",
                infrastructure_error=True,
            )
        message = _scorecard_message(cause, asserts.scorecard())
        met, total = asserts.criteria_counts()
        logger.info(
            "run_fixture fail (%.1fs): %s (%d/%d criteria) -- %s",
            time.time() - started,
            fixture_name,
            met,
            total,
            cause,
        )
        return verdict(False, message)
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
        return verdict(False, message, infrastructure_error=True)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        message = f"{type(exc).__name__}: {exc}"
        if capture.messages:
            api_key_summary = "\n".join(capture.messages)
            message = f"{api_key_summary}\n\n(Original exception, likely a symptom of the above: {message})"
        logger.warning(
            "run_fixture infrastructure_error (%.1fs): %s -- %s", time.time() - started, fixture_name, message
        )
        return verdict(False, message, infrastructure_error=True)
    finally:
        root_logger.removeHandler(capture)
        _write_consolidated_thinking(fixture_name, started)


def run_all_tests(fixtures_dir: str, only_fixtures: list[str] = None) -> list[dict[str, Any]]:
    """
    :param fixtures_dir: Network path under tests/fixtures/, e.g. "generated/coffee_shop"
                (matches ANTeGen's target_agent_name, i.e. the hocon file path minus ".hocon").
    :param only_fixtures: If given, run only these basenames (e.g. ["foo.hocon"]) instead of
                every fixture in the directory -- lets a caller cheaply re-verify just the
                handful of fixtures it touched instead of paying for the full suite every round.
    :return: One result dict (see run_fixture) per fixture found, in fixture_paths order
                (independent of the order fixtures actually finished in).
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
    # Run fixtures concurrently (one thread each) instead of one at a time -- run_fixture
    # is already safe to call this way: _write_consolidated_thinking is keyed by fixture name,
    # and _ApiKeyErrorCapture filters to its own thread.
    with ThreadPoolExecutor(max_workers=min(len(paths), MAX_PARALLEL_FIXTURES)) as executor:
        results = list(executor.map(run_fixture, paths))
    passed = sum(1 for r in results if r["passed"])
    logger.info(
        "run_all_tests done (%.1fs): %d/%d passing under %s",
        time.time() - started,
        passed,
        len(results),
        fixtures_dir,
    )
    # Written here rather than at each call site: every path that produces per-fixture
    # verdicts goes through this function, so one call covers the whole runner.
    _write_fixture_results(results)
    return results


def _write_fixture_results(results: list[dict[str, Any]]) -> None:
    """Record each fixture's verdict for an nsflow-launched run, so the UI can show pass/fail
    and the reason per test instead of leaving them to be scraped out of the log.

    Merged into the existing map rather than replacing it: the runner frequently re-tests only
    the fixtures that were failing, and a subset re-check says nothing about the ones it didn't
    run -- they keep their last known verdict, exactly as _ProgressTracker carries them forward
    for the chart. A no-op outside an nsflow job, like the tracker's own write.

    :param results: run_fixture result dicts for the fixtures this round actually ran
    """
    if not (NSFLOW_JOB_ID and NSFLOW_JOB_DIR):
        return
    path = os.path.join(NSFLOW_JOB_DIR, f"{NSFLOW_JOB_ID}.results.json")
    try:
        with open(path, encoding="utf-8") as results_file:
            merged: dict[str, Any] = json.load(results_file)
    except (OSError, json.JSONDecodeError):
        # Absent on the first round, and a half-written file is not worth failing a test run
        # over -- this round's results are about to replace what matters in it anyway.
        merged = {}
    for result in results:
        merged[result["fixture"]] = {
            "passed": bool(result.get("passed")),
            "message": result.get("message"),
            "infrastructure_error": bool(result.get("infrastructure_error")),
        }
    try:
        with open(path, "w", encoding="utf-8") as results_file:
            json.dump(merged, results_file, indent=2)
    except OSError as error:
        logger.warning("Could not write fixture results to %s: %s", path, error)
