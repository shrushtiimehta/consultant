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
tool that has to first figure out "is this structural or instructions-only"; consultant
skips that and goes straight from a failing-test report to per-agent instruction fixes.

The fixture-running engine lives next door in test_runner.py; this module is the loop that
drives it -- generate, test, diagnose, repair, re-test.

By default this runs in-process (--connection direct), no server needed. Pass --connection
http to instead talk to an already-running `ns run` server -- useful if you want this to share
a server with other clients, but NOT to watch the run in nsflow: nsflow's live view only shows
conversations started through its own UI/websocket, so a script hitting the neuro-san server's
plain chat API directly (this one) never appears there regardless of connection type.

Usage:
    python -m apps.network_consultant.run --use-case "A coffee shop order-status bot"
    python -m apps.network_consultant.run --hocon-file generated/coffee_shop.hocon \
        --direction "Preserve order lookup"
    python -m apps.network_consultant.run --hocon-file generated/coffee_shop.hocon \
        --direction "Preserve order lookup" --connection http
"""

import argparse
import logging
import os
import re
import shutil
import signal
import time
from pathlib import PurePosixPath
from typing import Any
from typing import Optional

# isort: off
# pylint: disable=wrong-import-order
# Import order here is load-bearing, not cosmetic: test_runner sets the direct-session
# environment defaults (AGENT_MANIFEST_FILE and friends) that neuro-san reads at IMPORT time,
# so it has to precede the neuro-san imports below. That is the reverse of isort's
# third-party-before-first-party rule, hence the off/on fence rather than a reordering.
from apps.network_consultant.test_runner import IMPROVEMENT_THINKING_DIR
from apps.network_consultant.test_runner import NSFLOW_JOB_DIR
from apps.network_consultant.test_runner import NSFLOW_JOB_ID
from apps.network_consultant.test_runner import _run_real_ratio_suite
from apps.network_consultant.test_runner import _write_fixture_results
from apps.network_consultant.test_runner import fixture_paths
from apps.network_consultant.test_runner import restore_success_ratios
from apps.network_consultant.test_runner import run_all_tests
from apps.network_consultant.test_runner import set_success_ratio_for_fixtures
from apps.network_consultant.generated_tests_cache import GeneratedTestsCache
from apps.network_consultant.git_versioning import GIT_VERSIONS_BRANCH_PREFIX
from apps.network_consultant.git_versioning import GitVersioning
from apps.network_consultant.progress_tracker import _ProgressTracker
from apps.network_consultant.scoring import ConsultantScoring
from apps.network_consultant.scoring import extract_prefixed
from apps.network_consultant.session import chat
from apps.network_consultant.session import HEADLESS_POLL_INTERVAL_SECONDS
from apps.network_consultant.session import open_session

from coded_tools.agent_network_consultant.network_scratchpad import clear_for_hocon_file

# isort: on

# Not __name__: this module runs as "__main__" via `python -m`, which would otherwise produce an
# unhelpful logger name. Shared with test_runner.py so the job log is one readable stream.
logger = logging.getLogger("network_consultant")

# Ratio a fixture gets bumped to once consultant is CONFIDENT its fix holds up under
# repeated runs. Everything else stays at whatever cheap ratio (usually 1/1) it already had --
# re-running every fixture at 3/3 every round is not worth the token cost. The authoritative
# Before/After bars lift the bump and use each fixture's own ratio: that is the network's
# real score, not the stricter one a vouched-for fix is being held to.
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


# =============================================================================================
# Orchestration
# =============================================================================================


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
TOOL_ISSUE_PREFIX = "TOOL_ISSUE:"
# A criterion asking for a fact the network has no way to obtain, because a tool it depends on
# returns nothing. Distinct from TOOL_ISSUE (that is an exception, this tool works fine and
# politely has no data) and from AGENT FIX (no instruction rewrite can conjure the fact).
UNGROUNDED_PREFIX = "UNGROUNDED:"


def _write_tool_issues(tool_issues: list[str]) -> None:
    """Persist reported tool issues to a file nsflow's backend surfaces in the UI (mirrors
    _ask_headless's question file) -- a no-op when not running as an nsflow job."""
    if not (NSFLOW_JOB_ID and NSFLOW_JOB_DIR):
        return
    issues_path = os.path.join(NSFLOW_JOB_DIR, f"{NSFLOW_JOB_ID}.tool_issues.txt")
    with open(issues_path, "w", encoding="utf-8") as issues_file:
        issues_file.write("\n".join(tool_issues))


def _write_ungrounded(entries: list[str]) -> None:
    """Persist reported ungrounded criteria where nsflow's backend can surface them (mirrors
    _write_tool_issues) -- a no-op when not running as an nsflow job."""
    if not (NSFLOW_JOB_ID and NSFLOW_JOB_DIR):
        return
    path = os.path.join(NSFLOW_JOB_DIR, f"{NSFLOW_JOB_ID}.ungrounded.txt")
    with open(path, "w", encoding="utf-8") as ungrounded_file:
        ungrounded_file.write("\n".join(entries))


# Signature of consultant retrying a tool call that can never succeed -- e.g. when the
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
    """Raised when consultant is stuck retrying an unfixable tool-call parse error
    against a specific network HOCON file."""

    def __init__(self, hocon_file: str, messages: list[str]):
        self.hocon_file = hocon_file
        self.messages = messages
        super().__init__(
            f"consultant is stuck patching {hocon_file} -- its source-preserving editor "
            "doesn't support this file's brace-less/'=' HOCON style. Skipping."
        )


def _guarded_chat(session, thread: dict, message: str, hocon_file: str, sly_data: dict = None) -> tuple:
    """chat(), but raises StuckPatchError if the parse-error signature repeats during the call
    instead of letting consultant retry a doomed tool call indefinitely."""
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


# Populated by main() so the SIGTERM handler can undo the run's on-disk side effects. Module
# level because a signal handler cannot reach main()'s locals. `original_ratios` is mutated in
# place throughout the run, so binding the same dict object here keeps it current.
_CLEANUP: dict[str, Any] = {"original_ratios": {}, "git_worktree": None}


def _handle_sigterm(_signum, _frame) -> None:
    """Undo what the run changed on disk, then exit.

    nsflow's Stop sends SIGTERM and escalates to SIGKILL 5 seconds later. Python's default
    SIGTERM terminates the process WITHOUT unwinding, so main()'s `finally` never ran: every
    Stop left the CONFIDENT_FIX bumps on disk (fixtures stuck at the stricter ratio, so the
    next run measured them 3x harder for no reason) and a stale git worktree behind.

    Cleaning up here rather than raising is deliberate. An exception would unwind through
    ThreadPoolExecutor.__exit__, which waits for every in-flight fixture -- minutes -- and
    would be SIGKILLed long before the cleanup ran. Rewriting a handful of success_ratio
    lines takes milliseconds and fits the 5 seconds comfortably.
    """
    logger.warning("Stop requested (SIGTERM) -- restoring fixture success ratios before exit.")
    try:
        restore_success_ratios(_CLEANUP["original_ratios"])
        GitVersioning.stop_git_versioning(_CLEANUP["git_worktree"])
    finally:
        # os._exit, not sys.exit: SystemExit would unwind into the very executor wait avoided above.
        os._exit(128 + signal.SIGTERM)


def normalize_hocon_reference(value: str) -> str:
    """Return a safe registries-relative HOCON reference for network and fixture lookup."""
    normalized = value.strip().replace("\\", "/")
    if normalized.startswith("registries/"):
        normalized = normalized[len("registries/") :]
    path = PurePosixPath(normalized)
    if path.is_absolute() or path.suffix != ".hocon" or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("HOCON file must be a safe .hocon path relative to registries/.")
    return path.as_posix()


def consult(session, thread: dict, message: str, hocon_file: str, fixture_paths_: dict[str, str]) -> tuple:
    """
    Send one diagnosis/follow-up message to consultant, and keep answering any
    NEEDS_CLARIFICATION questions it comes back with (asking the actual person running this
    script) until it returns a turn with no more open questions.

    :return: (final_response_text, updated_thread)
    """
    logger.info("consult start: hocon_file=%s fixtures=%d", hocon_file, len(fixture_paths_))
    response, thread = _guarded_chat(
        session,
        thread,
        message,
        hocon_file,
        sly_data={"agent_network_hocon_file": hocon_file, "test_fixture_paths": fixture_paths_},
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
    """Give consultant one chance to act on `direction` (e.g. token reduction) even when
    there's nothing failing to fix -- otherwise the front man is never invoked at all, and its
    "run token_reduction_advisor even with no failures" instruction never gets a chance to fire."""
    if not direction:
        return
    try:
        response, _ = consult(session, thread, all_passing_prompt(direction, total_fixture_count), hocon_file, {})
        logger.info("consultant response: %s", response)
    except StuckPatchError as exc:
        logger.error(str(exc))
        _write_tool_issues([str(exc)])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("consultant call failed unexpectedly: %s: %s", type(exc).__name__, exc)


def has_existing_fixtures(network_name: str) -> bool:
    """Whether tests/fixtures/<network_name>/ already has any generated fixture."""
    return bool(fixture_paths(network_name))


def diagnosis_prompt(
    failures: list, direction: str, total_fixture_count: int, is_subset_check: bool, ungrounded: str = "stop"
) -> str:
    """Builds the failing-test report handed to consultant, including each fixture's
    current content -- needed since it may decide to correct the fixture itself, not just the
    network's instructions.

    :param total_fixture_count: How many fixtures exist in the network's full suite.
    :param ungrounded: What the caller wants done about criteria no tool can satisfy --
                "stop" to report and halt, "continue" to drop them and keep going.
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
        "If a criterion asks for a concrete fact the network cannot obtain -- because a tool it "
        "depends on returns NO DATA (an unwired retrieval agent, an empty index, a stub asking the "
        "caller to supply the source) -- that is UNGROUNDED, not an agent defect. Confirm it in the "
        "thinking trace by finding the tool's own reply and checking it carries no content. No "
        "instruction rewrite can ever satisfy such a criterion, so do NOT classify it as AGENT FIX; "
        "that is the trap that burns every remaining round rewriting agents that were never at fault. "
        "Report one line per criterion: `UNGROUNDED: <fixture>: <toolname>: <the exact criterion "
        "text>`, naming a tool that actually EXISTS in the network you were given -- copy it "
        "character-for-character, never invent a plausible-sounding retriever."
    )
    if ungrounded == "continue":
        lines.append(
            "For each UNGROUNDED criterion, ALSO call `fixture_expectation_fixer`, naming the exact "
            "criteria to remove and saying they are ungrounded, then carry on fixing anything else "
            "that fixture gets wrong."
        )
    else:
        lines.append(
            "Output the UNGROUNDED lines and change neither the network nor the fixture for them; "
            "the run stops so a human can wire up the missing data source."
        )
    lines.append(
        "Never satisfy an ungrounded criterion by having the network state the fact anyway -- a "
        "fabricated answer is worse than a failing test."
    )
    lines.append(
        "Next round re-checks whichever fixtures are still failing after your fix -- not the full suite. "
        "A full sweep still runs once before success is declared."
    )
    return "\n".join(lines)


def all_passing_prompt(direction: str, total_fixture_count: int) -> str:
    """Report handed to consultant when every fixture already passes -- there's nothing
    to fix, but the user's direction (e.g. "reduce token usage") may still call for the
    token_reduction_advisor pass, which only ever runs if the front man is actually invoked."""
    return (
        "User's intended behavior and approximate vision:\n"
        f"{direction}\n\n"
        f"All {total_fixture_count} fixtures in the test suite are currently passing. There are no failures to "
        "fix. If the direction above calls for something to still be done (e.g. reducing token usage), do that "
        "now; otherwise say plainly that there is nothing to do."
    )


class NetworkConsultantApp:  # pylint: disable=too-few-public-methods
    """Run the Network Consultant command-line workflow."""

    @staticmethod
    # The state machine deliberately keeps its ordered early-stop decisions together.
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    # pylint: disable=too-many-return-statements,too-many-nested-blocks
    def main():
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
        parser.add_argument(
            "--hocon-file", help="Existing network hocon (relative to registries/) to iterate on instead."
        )
        parser.add_argument(
            "--direction",
            help="Intended behavior for an existing network; helps distinguish network defects from bad tests.",
        )
        parser.add_argument("--test-level", default="normal", choices=["minimum", "normal", "max"])
        parser.add_argument(
            "--test-guidance",
            default="",
            help="Free text steering what the test generator writes tests ABOUT (e.g. 'the vendor "
            "onboarding path'). Read only by the generator -- --direction is the consultant's "
            "statement of intended behavior and is a different thing.",
        )
        parser.add_argument(
            "--force-generate",
            action="store_true",
            help="Generate tests even when the network already has fixtures. Without this, "
            "generation is skipped whenever any fixture exists, which is right for the fix loop "
            "(it should reuse the suite, not pay to rebuild it) but makes an explicit 'generate' "
            "request silently do nothing. Regenerating overwrites same-named fixtures and adds "
            "new ones; it never deletes.",
        )
        parser.add_argument(
            "--ungrounded",
            default="stop",
            choices=["stop", "continue"],
            help="What to do when the consultant reports a criterion that no tool in the network can "
            "satisfy. 'stop' (default) ends the run so the missing data source can be wired up -- no "
            "instruction rewrite can ever make such a criterion pass. 'continue' tells the consultant "
            "to drop those criteria from their fixtures and keep improving everything else.",
        )
        parser.add_argument(
            "--only-fixtures",
            nargs="+",
            help="Run only these fixture basenames (e.g. 'foo.hocon') instead of the whole suite.",
        )
        parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
        parser.add_argument(
            "--success-ratio",
            default=CONFIDENT_SUCCESS_RATIO,
            help=f"Ratio (e.g. 'N/M') a fixture is bumped to once consultant is CONFIDENT its fix holds up "
            f"under repeated runs (default: {CONFIDENT_SUCCESS_RATIO}). Everything else stays cheap, and the "
            f"Before/After bars always score on each fixture's own ratio.",
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
                    "Designer did not return an agent_network_name; cannot continue. "
                    "Its response may explain why:\n%s",
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

        # The skip is right for the fix loop -- it should reuse the suite rather than pay to rebuild
        # it -- but it made an explicit "generate tests" request silently do nothing on any network
        # that already had some, so that request passes --force-generate.
        if not args.force_generate and has_existing_fixtures(network_name):
            logger.info("Existing test fixtures found for %s; skipping ANTeGen.", network_name)
        else:
            logger.info("Generating tests (ANTeGen, test_level=%s)...", args.test_level)
            testgen_session, testgen_thread = open_session(
                "agent_network_test_generator", args.connection, args.host, args.port
            )
            testgen_request = f"Generate test cases for {network_name} with {args.test_level} coverage"
            if args.test_guidance.strip():
                testgen_request += f". Focus on: {args.test_guidance.strip()}"
            logger.info("ANTeGen request: %s", testgen_request)
            response, testgen_thread = chat(testgen_session, testgen_thread, testgen_request)
            logger.info("ANTeGen response: %s", response)

        original_ratios: dict[str, str] = {}
        # Same dict object the handler restores from; it is mutated in place from here on.
        _CLEANUP["original_ratios"] = original_ratios
        signal.signal(signal.SIGTERM, _handle_sigterm)
        # None = run the full suite; otherwise a list of basenames to re-check cheaply instead of
        # paying for every fixture every round.
        retest_only = None
        total_fixture_count = None
        git_worktree = None
        try:
            # (fixtures passing, criteria passing), best-first -- see _round_score.
            best_score: Optional[tuple[int, int]] = None
            stale_rounds = 0
            # Tracked apart from the two above, which only ever see full-suite results: a subset
            # re-check's failure count is not comparable with the whole suite's, and running out of
            # ideas on a subset ends the subset, not the run. See the go-wide block below.
            subset_best_score: Optional[tuple[int, int]] = None
            subset_stale_rounds = 0
            progress_tracker = _ProgressTracker()
            # Only the fix rounds are numbered on the chart's x-axis; the Before/After bookends are
            # labelled by name instead, so they don't consume an iteration number.
            improvement_iteration = 0
            # hocon_file is registries-relative; the snapshot below is the file text that produced
            # the best score so far, so a plateau can roll the later dead-end edits back off.
            hocon_path = os.path.join("registries", hocon_file)
            best_hocon_text = None
            best_hocon_iteration = None

            if args.max_iterations == 0:
                # Run the suite once and stop -- this serves both "Generate Tests" (run the fresh
                # fixtures so the user sees pass/fail immediately) and "Run tests" (fixtures already
                # exist, so generation above was a no-op). Plain CLI usage (no nsflow job env vars)
                # keeps the old behavior exactly: generate and stop, no test run at all.
                if NSFLOW_JOB_ID and NSFLOW_JOB_DIR:
                    is_subset = bool(args.only_fixtures)
                    logger.info(
                        "Test run, no fix loop%s...",
                        f" (subset of {args.only_fixtures})" if is_subset else "",
                    )
                    results = run_all_tests(network_name, only_fixtures=args.only_fixtures)
                    failures = [r for r in results if not r["passed"]]
                    # Both of these describe the WHOLE suite, so neither may be fed a subset:
                    # the cache becomes a later Self-Improve run's Before baseline, and the chart's
                    # denominator would collapse to however many fixtures this run happened to pick.
                    # A subset run reports itself through the per-fixture results file instead.
                    if not is_subset:
                        GeneratedTestsCache.save(network_name, hocon_path, results)
                        # Its own bar, not a Before: "Before" belongs to the Self-Improve run, which
                        # reuses this very result (via the gentests cache) to draw it there.
                        progress_tracker.record(results, "generated", len(results))
                    logger.info("Result: %d/%d passing.", len(results) - len(failures), len(results))
                return

            git_worktree = (
                # A timestamp reads far better in a branch list than NSFLOW_JOB_ID's raw hex --
                # the job id itself is already logged alongside the branch name for correlation.
                GitVersioning.start_git_versioning(network_name, time.strftime("%Y%m%d-%H%M%S"))
                if args.git_versions
                else None
            )
            _CLEANUP["git_worktree"] = git_worktree

            for iteration in range(1, args.max_iterations + 1):
                logger.info("--- Iteration %d/%d: running tests ---", iteration, args.max_iterations)
                is_subset_check = retest_only is not None
                cached_results = (
                    GeneratedTestsCache.load(network_name, hocon_path)
                    if iteration == 1 and NSFLOW_JOB_ID and NSFLOW_JOB_DIR
                    else None
                )
                if cached_results is not None:
                    logger.info("Reusing the Generate Tests baseline (network unchanged since) -- skipping re-test.")
                    results = cached_results
                    # run_all_tests normally writes these; this path skips it, and without them the
                    # UI shows nothing for a run that already knows exactly which fixtures fail.
                    _write_fixture_results(results)
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
                GitVersioning.commit_hocon_version(
                    git_worktree,
                    hocon_file,
                    f"{'Before' if iteration == 1 else f'Iteration {iteration}'}: "
                    f"{len(results) - len(failures)}/{len(results)} passing"
                    f"{' (subset re-check)' if is_subset_check else ''}",
                )
                # total_fixture_count is fixed at the full-suite size (set on the first, non-subset
                # iteration) -- a subset re-check's "passed" is everything outside that subset (assumed
                # still passing) plus whatever of the subset just passed, so the chart's denominator
                # never shrinks: the tracker above carries fixtures this round didn't run forward at
                # their last known state instead of dropping them out of the chart entirely.

                if not failures:
                    if retest_only is not None:
                        logger.info("Subset re-check passed; running full suite once to confirm no regressions...")
                        results = _run_real_ratio_suite(network_name, original_ratios, args.success_ratio)
                        total_fixture_count = len(results)
                        infrastructure_errors = [result for result in results if result.get("infrastructure_error")]
                        if infrastructure_errors:
                            logger.error(
                                "Full-suite confirmation could not complete because test infrastructure failed."
                            )
                            for error in infrastructure_errors:
                                logger.error("  - %s: %s", error["fixture"], error["message"])
                            return
                        failures = [r for r in results if not r["passed"]]
                        progress_tracker.record(results, "after", total_fixture_count)
                        GitVersioning.commit_hocon_version(
                            git_worktree,
                            hocon_file,
                            f"Iteration {iteration} (full-suite confirmation): "
                            f"{total_fixture_count - len(failures)}/{total_fixture_count} passing",
                        )
                        if failures:
                            passed_count = total_fixture_count - len(failures)
                            if ConsultantScoring.good_enough(passed_count, total_fixture_count):
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
                            # consultant made no edit (e.g. it had nothing to do) -- the full
                            # suite already passed just before this call, so re-running it again would
                            # burn a whole extra round of fixture tests to reconfirm an unchanged file.
                            # The one exception: a CONFIDENT_FIX bump is in force, so that run scored
                            # some fixtures at the stricter ratio -- not the real score an After bar
                            # reports, so it does have to be redone on the fixtures' own ratios.
                            if original_ratios:
                                logger.info(
                                    "consultant made no changes; re-scoring the full suite on the fixtures' "
                                    "own ratios for the After bar."
                                )
                                results = _run_real_ratio_suite(network_name, original_ratios, args.success_ratio)
                                total_fixture_count = len(results)
                            else:
                                logger.info("consultant made no changes; skipping the redundant re-verification run.")
                            progress_tracker.record(results, "after", total_fixture_count)
                            return
                        logger.info("Re-running full suite to verify that change didn't break anything...")
                        results = _run_real_ratio_suite(network_name, original_ratios, args.success_ratio)
                        total_fixture_count = len(results)
                        infrastructure_errors = [result for result in results if result.get("infrastructure_error")]
                        if infrastructure_errors:
                            logger.error("Final confirmation could not complete because test infrastructure failed.")
                            for error in infrastructure_errors:
                                logger.error("  - %s: %s", error["fixture"], error["message"])
                            return
                        failures = [r for r in results if not r["passed"]]
                        progress_tracker.record(results, "after", total_fixture_count)
                        GitVersioning.commit_hocon_version(
                            git_worktree,
                            hocon_file,
                            f"After: {total_fixture_count - len(failures)}/{total_fixture_count} passing",
                        )
                        if not failures:
                            logger.info("Still all passing after verification. Stopping.")
                            return
                        passed_count = total_fixture_count - len(failures)
                        if ConsultantScoring.good_enough(passed_count, total_fixture_count):
                            print(
                                f"[network_consultant] {passed_count}/{total_fixture_count} passing "
                                f"(>= {GOOD_ENOUGH_RATIO:.0%}) on the full suite -- good enough, moving on."
                            )
                            for failure in failures:
                                logger.info(
                                    "  - still failing: %s: %s", failure["fixture"], failure["message"].strip()
                                )
                            return
                        logger.warning(
                            "That change introduced %d regression(s); continuing to fix them instead of stopping.",
                            len(failures),
                        )
                        retest_only = None
                        is_subset_check = False

                # A subset that has stopped improving is a reason to stop trusting the subset, not to
                # end the run. Edits made while focused on a handful of fixtures regress ones the
                # subset never runs -- and the chart has been extrapolating over those for rounds. So
                # go wide: measure everything, chart that as an authoritative After bar, and keep
                # going against whatever is really failing.
                if is_subset_check and subset_stale_rounds >= PLATEAU_STRIKES:
                    logger.warning(
                        "The failing subset hasn't improved for %d rounds; re-checking the whole suite.",
                        PLATEAU_STRIKES,
                    )
                    results = _run_real_ratio_suite(network_name, original_ratios, args.success_ratio)
                    infrastructure_errors = [result for result in results if result.get("infrastructure_error")]
                    if infrastructure_errors:
                        logger.error("Full-suite re-check could not complete because test infrastructure failed.")
                        for error in infrastructure_errors:
                            logger.error("  - %s: %s", error["fixture"], error["message"])
                        return
                    total_fixture_count = len(results)
                    failures = [r for r in results if not r["passed"]]
                    passed_count = total_fixture_count - len(failures)
                    progress_tracker.record(results, "after", total_fixture_count)
                    GitVersioning.commit_hocon_version(
                        git_worktree, hocon_file, f"After: {passed_count}/{total_fixture_count} passing"
                    )
                    logger.info("Full suite: %d/%d passing.", passed_count, total_fixture_count)
                    if ConsultantScoring.good_enough(passed_count, total_fixture_count):
                        print(
                            f"[network_consultant] {passed_count}/{total_fixture_count} passing "
                            f"(>= {GOOD_ENOUGH_RATIO:.0%}) on the full suite -- good enough, moving on."
                        )
                        for failure in failures:
                            logger.info("  - still failing: %s: %s", failure["fixture"], failure["message"].strip())
                        return
                    # Under the bar, so carry on -- against the real failures now, not the stuck
                    # subset. Whether THAT has run out of road is the full-suite bookkeeping's call.
                    logger.warning(
                        "%d/%d passing is below %.0f%%; continuing against the full set of failures.",
                        passed_count,
                        total_fixture_count,
                        GOOD_ENOUGH_RATIO * 100,
                    )
                    retest_only = None
                    is_subset_check = False
                    subset_best_score = None
                    subset_stale_rounds = 0

                score = ConsultantScoring.round_score(results)
                if is_subset_check:
                    # Strictly separate from the full-suite counters: a subset's score covers only
                    # the fixtures it ran, so it must never move best_score, take the HOCON
                    # snapshot, or vote on giving up. Its strikes send the loop wide above.
                    if subset_best_score is not None and score <= subset_best_score:
                        subset_stale_rounds += 1
                    else:
                        subset_stale_rounds = 0
                    subset_best_score = score if subset_best_score is None else max(score, subset_best_score)
                else:
                    improved = best_score is None or score > best_score
                    if not improved:
                        stale_rounds += 1
                    else:
                        stale_rounds = 0
                    best_score = score if best_score is None else max(score, best_score)
                    logger.info(
                        "Round score: %d/%d fixtures, %d/%d criteria (best so far %d/%d fixtures, %d criteria).",
                        score[0],
                        len(results),
                        score[1],
                        sum(r.get("criteria_total", 0) for r in results),
                        best_score[0],
                        len(results),
                        best_score[1],
                    )
                    if improved:
                        # Snapshot the HOCON that produced this best-so-far score -- read now, before
                        # the editor touches it again, so a later plateau can roll the useless edits
                        # back off. Iteration 1 snapshots the original: the right floor to fall back to.
                        with open(hocon_path, encoding="utf-8") as best_file:
                            best_hocon_text = best_file.read()
                        best_hocon_iteration = iteration

                # Only full-suite rounds ever reach PLATEAU_STRIKES here (subsets are handled above),
                # so this is the whole network genuinely refusing to move -- the one case that ends
                # the run below 80% instead of iterating again.
                if stale_rounds >= PLATEAU_STRIKES:
                    print(
                        f"[network_consultant] Tried hard for {iteration} rounds, but this isn't working -- "
                        f"{len(failures)}/{len(results)} fixtures still failing on the full suite. Giving up here."
                    )
                    logger.warning(
                        "No full-suite improvement for %d consecutive rounds. Stopping with %d/%d still failing:",
                        PLATEAU_STRIKES,
                        len(failures),
                        len(results),
                    )
                    for failure in failures:
                        logger.warning("  - %s: %s", failure["fixture"], failure["message"].strip())
                    ConsultantScoring.restore_best_hocon(hocon_path, best_hocon_text, best_hocon_iteration)
                    logger.info("Re-running the full suite on the restored best version for the final After bar...")
                    results = _run_real_ratio_suite(network_name, original_ratios, args.success_ratio)
                    infrastructure_errors = [result for result in results if result.get("infrastructure_error")]
                    if not infrastructure_errors:
                        progress_tracker.record(results, "after", len(results))
                    else:
                        logger.error("Final confirmation could not complete because test infrastructure failed.")
                    return

                logger.info("Consulting consultant to fix failing agents' instructions...")
                failure_fixture_paths = {failure["fixture"]: failure["path"] for failure in failures}
                try:
                    response, consultant_thread = consult(
                        consultant_session,
                        consultant_thread,
                        diagnosis_prompt(failures, direction, total_fixture_count, is_subset_check, args.ungrounded),
                        hocon_file,
                        failure_fixture_paths,
                    )
                except StuckPatchError as exc:
                    logger.error(str(exc))
                    _write_tool_issues([str(exc)])
                    return
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error("consultant call failed unexpectedly: %s: %s", type(exc).__name__, exc)
                    _write_tool_issues([f"{type(exc).__name__}: {exc}"])
                    return
                logger.info("consultant response: %s", response)
                # Commit/push the edit itself the moment it's made -- not the checkpoint AFTER the
                # next test run confirms it. Waiting for that confirmation is why the first-ever
                # edit (made here, while `iteration` is still 1) used to only show up in git once
                # iteration 2's test round ran, one full round later than the edit that produced it.
                GitVersioning.commit_hocon_version(
                    git_worktree, hocon_file, f"Change {iteration}: fixing {len(failures)} failing fixture(s)"
                )
                if any(line.strip().startswith(STRUCTURAL_CHANGE_PREFIX) for line in (response or "").splitlines()):
                    logger.warning("A structural change requires explicit Designer review. Stopping safely.")
                    return

                tool_issues = extract_prefixed(response, TOOL_ISSUE_PREFIX)
                if tool_issues:
                    print(
                        "[network_consultant] A required coded tool is broken -- this needs a human code fix, not "
                        "an instructions/fixture change. Stopping so you can fix it and re-run:"
                    )
                    for issue in tool_issues:
                        print(f"  ! {issue}")
                    logger.warning("Tool issue(s) reported; stopping for a human fix: %s", tool_issues)
                    _write_tool_issues(tool_issues)
                    return

                ungrounded = extract_prefixed(response, UNGROUNDED_PREFIX)
                if ungrounded:
                    # Not a defect the loop can fix: the tool the criterion depends on has no data,
                    # so the network is right to decline to invent the fact. Left unreported, every
                    # remaining iteration goes on rewriting agents that were never at fault --
                    # observed on industry/intranet_agents_with_tools, four rounds at ~5.5 min each.
                    print("[network_consultant] Some criteria ask for facts no tool in this network can supply:")
                    for entry in ungrounded:
                        print(f"  ? {entry}")
                    logger.warning("Ungrounded criteria reported: %s", ungrounded)
                    _write_ungrounded(ungrounded)
                    if args.ungrounded == "stop":
                        print(
                            "[network_consultant] Stopping -- wire up the data source, or re-run with "
                            "--ungrounded continue to drop those criteria and keep improving the rest."
                        )
                        return

                confident_fixtures = extract_prefixed(response, CONFIDENT_FIX_PREFIX)
                if confident_fixtures:
                    new_originals = set_success_ratio_for_fixtures(
                        network_name, confident_fixtures, args.success_ratio
                    )
                    original_ratios.update(new_originals)
                    logger.info(
                        "consultant is confident in %d fix(es); bumped to %s for next round: %s",
                        len(confident_fixtures),
                        args.success_ratio,
                        confident_fixtures,
                    )

                # Next round, only re-check what we just worked on -- cheap, targeted re-verification
                # instead of the whole suite. A full sweep still runs once before declaring success.
                #
                # This set is NOT the consultant's to choose. It used to be able to override it with
                # RETEST_ONLY: lines, and on industry/cpg_agents it emitted three CONFIDENT_FIX lines
                # and a RETEST_ONLY naming a fourth fixture -- so the three it had just fixed and
                # vouched for were bumped to 3/3 and then never run again. Two rounds (285s) measured
                # one fixture, the chart read 2/6 then 3/6, and the closing full sweep found 5/6: the
                # loop was two fixtures better than it could see, and a plateau in that state would
                # have rolled back real fixes. An agent must not be able to excuse its own work from
                # the test that grades it.
                retest_only = [failure["fixture"] for failure in failures]

            logger.warning("Reached max iterations (%d) without a full pass.", args.max_iterations)
            ConsultantScoring.restore_best_hocon(hocon_path, best_hocon_text, best_hocon_iteration)
            logger.info("Re-running the full suite on the restored best version for the final After bar...")
            results = _run_real_ratio_suite(network_name, original_ratios, args.success_ratio)
            infrastructure_errors = [result for result in results if result.get("infrastructure_error")]
            if not infrastructure_errors:
                progress_tracker.record(results, "after", len(results))
            else:
                logger.error("Final confirmation could not complete because test infrastructure failed.")
        finally:
            logger.info("Restoring original success_ratio values...")
            restore_success_ratios(original_ratios)
            GitVersioning.stop_git_versioning(git_worktree)


if __name__ == "__main__":
    NetworkConsultantApp.main()
