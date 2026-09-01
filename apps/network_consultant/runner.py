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
from pathlib import PurePosixPath
from typing import Callable
from typing import Optional

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

# These imports intentionally follow the direct-session environment defaults above.
# pylint: disable=wrong-import-position
from neuro_san.client.agent_session_factory import AgentSessionFactory  # noqa: E402
from neuro_san.client.streaming_input_processor import StreamingInputProcessor  # noqa: E402

from apps.network_consultant.test_runner import IMPROVEMENT_THINKING_DIR  # noqa: E402
from apps.network_consultant.test_runner import restore_success_ratios  # noqa: E402
from apps.network_consultant.test_runner import run_all_tests  # noqa: E402
from apps.network_consultant.test_runner import set_success_ratio_for_fixtures  # noqa: E402
from coded_tools.agent_network_consultant.network_scratchpad import clear_for_hocon_file  # noqa: E402

# Not __name__: this module runs as "__main__" via `python -m`, which would otherwise
# produce an unhelpful logger name.
logger = logging.getLogger("network_consultant")

# Ratio a fixture gets bumped to once consultant is CONFIDENT its fix holds up under
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
# Per-network iteration history (pass/total per round), so a chart can be built after the fact
# even for a plain CLI run that never sets the nsflow job env vars below.
PROGRESS_DIR = "/tmp/network_consultant_progress"


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

# Pushed to a dedicated remote rather than `origin` -- these are throwaway per-run snapshots,
# not something to land on whatever repo `origin` happens to point at (e.g. a shared upstream
# project). Override via env var for a different personal remote.
GIT_VERSIONS_REMOTE_NAME = "network-consultant-versions"
GIT_VERSIONS_REMOTE_URL: str = os.environ.get(
    "NETWORK_CONSULTANT_GIT_VERSIONS_REMOTE", "https://github.com/shrushtiimehta/consultant.git"
)


def _ensure_git_versions_remote() -> None:
    """Add GIT_VERSIONS_REMOTE_NAME pointing at GIT_VERSIONS_REMOTE_URL if it isn't already
    configured, or repoint it if some other URL is there under that name -- so a change to
    NETWORK_CONSULTANT_GIT_VERSIONS_REMOTE takes effect without manual `git remote` surgery."""
    existing = subprocess.run(
        ["git", "remote", "get-url", GIT_VERSIONS_REMOTE_NAME], capture_output=True, text=True
    )
    if existing.returncode == 0:
        if existing.stdout.strip() != GIT_VERSIONS_REMOTE_URL:
            subprocess.run(
                ["git", "remote", "set-url", GIT_VERSIONS_REMOTE_NAME, GIT_VERSIONS_REMOTE_URL], check=True
            )
        return
    subprocess.run(["git", "remote", "add", GIT_VERSIONS_REMOTE_NAME, GIT_VERSIONS_REMOTE_URL], check=True)


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
        _ensure_git_versions_remote()
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
    logger.info(
        "Versioning network hocon snapshots to branch %r on remote %r (%s).",
        branch,
        GIT_VERSIONS_REMOTE_NAME,
        GIT_VERSIONS_REMOTE_URL,
    )
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
            ["git", "-C", worktree_dir, "push", "-u", GIT_VERSIONS_REMOTE_NAME, "HEAD"],
            check=True,
            capture_output=True,
            text=True,
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


_PROGRESS_STEPS: dict[str, int] = {}
_PROGRESS_PATHS_WRITTEN: set[str] = set()


def _progress_paths(network_name: str) -> list[str]:
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    paths = [os.path.join(PROGRESS_DIR, f"{network_name.replace('/', '_')}.progress.jsonl")]
    if NSFLOW_JOB_ID and NSFLOW_JOB_DIR:
        paths.append(os.path.join(NSFLOW_JOB_DIR, f"{NSFLOW_JOB_ID}.progress.jsonl"))
    return paths


def _start_progress_batch(network_name: str) -> None:
    """Start a new bar: the next _write_progress call(s) for this network begin a fresh row
    instead of editing the previous batch's. Call once per run_all_tests round (a subset
    re-check and its follow-up full-suite confirmation are separate batches, so each gets its
    own bar) -- NOT once per test within that round."""
    for path in _progress_paths(network_name):
        _PROGRESS_STEPS[path] = _PROGRESS_STEPS.get(path, 0) + 1


def _write_progress(network_name: str, passed: int, total: int, complete: bool, full_suite: bool = True) -> None:
    """Update the current batch's progress row in PROGRESS_DIR (always) and, when running as an
    nsflow job, in the job dir too (surfaced there as a live progress chart). Call after each
    test in a batch so the bar edits live instead of only appearing once the batch finishes.
    `complete` marks whether this was the batch's last test, so the chart can render a
    still-running bar differently (e.g. faded) from a finished one. `full_suite` marks whether
    this batch checked every fixture (the initial baseline and the final confirmation) as
    opposed to only the fixtures that were still failing last round -- the chart uses this to
    tell an authoritative before/after result apart from an in-progress retest of just what's
    broken. The first write to a given file in this process ignores whatever it finds on disk --
    a fresh run shouldn't inherit a prior run's tail rows -- every write after that preserves
    earlier bars."""
    for path in _progress_paths(network_name):
        step = _PROGRESS_STEPS.setdefault(path, 1)
        rows = []
        if path in _PROGRESS_PATHS_WRITTEN and os.path.exists(path):
            with open(path, encoding="utf-8") as existing:
                rows = [json.loads(line) for line in existing if line.strip()]
        rows = [row for row in rows if row["iteration"] != step]
        rows.append(
            {"iteration": step, "passed": passed, "total": total, "complete": complete, "full_suite": full_suite}
        )
        # Write to a temp file and rename over the target -- os.replace is atomic on POSIX and
        # Windows, so a concurrent reader (nsflow polling this same file) always sees either the
        # complete old content or the complete new content, never a half-written file.
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as progress_file:
            for row in rows:
                progress_file.write(json.dumps(row) + "\n")
        os.replace(tmp_path, path)
        _PROGRESS_PATHS_WRITTEN.add(path)


def _progress_chart_writer(
    network_name: str, is_subset_check: bool, total_fixture_count: int
) -> Callable[[dict, int], None]:
    """Return a run_all_tests on_result callback that updates the current batch's bar as each
    test finishes: the first test shows e.g. 1/1, the second 1/2 or 2/2, and so on, up to the
    batch's full size -- call _start_progress_batch first so this batch gets its own bar. A
    subset check's fixtures outside the batch are assumed still passing, so both the running
    numerator and denominator start from that baseline (total_fixture_count minus this batch's
    size) instead of zero."""
    state = {"done": 0, "passed": 0}

    def on_result(result: dict, batch_size: int) -> None:
        state["done"] += 1
        state["passed"] += 1 if result["passed"] else 0
        baseline = (total_fixture_count - batch_size) if is_subset_check else 0
        _write_progress(
            network_name,
            baseline + state["passed"],
            baseline + state["done"],
            state["done"] == batch_size,
            full_suite=not is_subset_check,
        )

    return on_result


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
    fixture's consolidated thinking trace (see test_runner.IMPROVEMENT_THINKING_DIR) -- without
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
    Send one diagnosis/follow-up message to consultant, and keep answering any
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
    """Builds the failing-test report handed to consultant, including each fixture's
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


def main():  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """Run the iterative generate, test, diagnose, and repair workflow."""
    # Root stays at WARNING so third-party loggers (neuro-san's manifest loading, etc.) don't
    # flood the output -- only this app's own loggers are bumped to INFO.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    logger.setLevel(logging.INFO)
    logging.getLogger("apps.network_consultant.test_runner").setLevel(logging.INFO)
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
        help=f"Ratio (e.g. 'N/M') a fixture is bumped to once consultant is CONFIDENT its fix holds up "
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
    # None = run the full suite; otherwise a list of basenames to re-check cheaply instead of
    # paying for every fixture every round.
    retest_only = None
    total_fixture_count = None
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
            # Generate Tests, triggered from the UI: run the fresh fixtures once so the user
            # sees a pass/fail chart immediately, and cache the result so a Self-Improve run
            # started right after doesn't pay to re-run every fixture a second time. Plain CLI
            # usage (no nsflow job env vars) keeps the old behavior exactly: generate and stop,
            # no test run at all.
            if NSFLOW_JOB_ID and NSFLOW_JOB_DIR:
                logger.info("Baseline check (Generate Tests, no fix loop)...")
                _start_progress_batch(network_name)
                results = run_all_tests(
                    network_name, on_result=_progress_chart_writer(network_name, False, None)
                )
                _save_gentests_cache(network_name, hocon_path, results)
                failures = [r for r in results if not r["passed"]]
                logger.info("Baseline: %d/%d passing.", len(results) - len(failures), len(results))
            return

        git_worktree = (
            # A timestamp reads far better in a branch list than NSFLOW_JOB_ID's raw hex --
            # the job id itself is already logged alongside the branch name for correlation.
            _start_git_versioning(network_name, time.strftime("%Y%m%d-%H%M%S"))
            if args.git_versions
            else None
        )

        for iteration in range(1, args.max_iterations + 1):
            logger.info("--- Iteration %d/%d: running tests ---", iteration, args.max_iterations)
            is_subset_check = retest_only is not None
            _start_progress_batch(network_name)
            cached_results = (
                _load_gentests_cache(network_name, hocon_path)
                if iteration == 1 and NSFLOW_JOB_ID and NSFLOW_JOB_DIR
                else None
            )
            if cached_results is not None:
                logger.info("Reusing the Generate Tests baseline (network unchanged since) -- skipping re-test.")
                results = cached_results
                _write_progress(network_name, sum(1 for r in results if r["passed"]), len(results), True)
            else:
                results = run_all_tests(
                    network_name,
                    only_fixtures=retest_only,
                    on_result=_progress_chart_writer(network_name, is_subset_check, total_fixture_count),
                )
            if not is_subset_check:
                total_fixture_count = len(results)
            infrastructure_errors = [result for result in results if result.get("infrastructure_error")]
            if infrastructure_errors:
                logger.error("Test infrastructure failed; no network or fixture changes were attempted:")
                for error in infrastructure_errors:
                    logger.error("  - %s: %s", error["fixture"], error["message"])
                return
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
            # total_fixture_count is fixed at the full-suite size (set on the first, non-subset
            # iteration) -- a subset re-check's "passed" is everything outside that subset (assumed
            # still passing) plus whatever of the subset just passed, so the chart's denominator
            # never shrinks and passing count only ever grows toward it. The on_result callback
            # above already wrote a row for every fixture in this batch, including this final tally.

            if not failures:
                if retest_only is not None:
                    logger.info("Subset re-check passed; running full suite once to confirm no regressions...")
                    _start_progress_batch(network_name)
                    results = run_all_tests(
                        network_name, on_result=_progress_chart_writer(network_name, False, total_fixture_count)
                    )
                    total_fixture_count = len(results)
                    failures = [r for r in results if not r["passed"]]
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
                                logger.info("  - still failing: %s: %s", failure["fixture"], failure["message"].strip())
                            return
                        # Not good enough -- regressions elsewhere in the suite; keep going against those.
                        retest_only = None
                        is_subset_check = False

                if not failures:
                    logger.info("All tests passing. Network is satisfiable.")
                    with open(hocon_path, encoding="utf-8") as before_file:
                        hocon_before_consult = before_file.read()
                    _consult_all_passing(consultant_session, consultant_thread, direction, total_fixture_count, hocon_file)
                    with open(hocon_path, encoding="utf-8") as after_file:
                        hocon_after_consult = after_file.read()
                    if hocon_after_consult == hocon_before_consult:
                        # consultant made no edit (e.g. it had nothing to do) -- the full
                        # suite already passed just before this call, so re-running it again would
                        # burn a whole extra round of fixture tests to reconfirm an unchanged file.
                        logger.info("consultant made no changes; skipping the redundant re-verification run.")
                        return
                    logger.info("Re-running full suite to verify that change didn't break anything...")
                    _start_progress_batch(network_name)
                    results = run_all_tests(
                        network_name, on_result=_progress_chart_writer(network_name, False, total_fixture_count)
                    )
                    total_fixture_count = len(results)
                    failures = [r for r in results if not r["passed"]]
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

            improved = best_failure_count is None or len(failures) < best_failure_count
            if best_failure_count is not None and len(failures) >= best_failure_count:
                stale_rounds += 1
            else:
                stale_rounds = 0
            best_failure_count = (
                min(len(failures), best_failure_count) if best_failure_count is not None else len(failures)
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
                return

            logger.info("Consulting consultant to fix failing agents' instructions...")
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
                logger.error("consultant call failed unexpectedly: %s: %s", type(exc).__name__, exc)
                _write_tool_issues([f"{type(exc).__name__}: {exc}"])
                return
            logger.info("consultant response: %s", response)
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
                    "consultant is confident in %d fix(es); bumped to %s for next round: %s",
                    len(confident_fixtures),
                    args.success_ratio,
                    confident_fixtures,
                )

            # Next round, only re-check what we just worked on -- cheap, targeted re-verification
            # instead of the whole suite. A full sweep still runs once before declaring success.
            # consultant can override this default via explicit RETEST_ONLY: lines.
            requested_retest = extract_prefixed(response, RETEST_ONLY_PREFIX)
            if requested_retest:
                retest_only = requested_retest
                logger.info("consultant requested a specific retest set: %s", retest_only)
            else:
                retest_only = [failure["fixture"] for failure in failures]

        logger.warning("Reached max iterations (%d) without a full pass.", args.max_iterations)
        _restore_best_hocon(hocon_path, best_hocon_text, best_hocon_iteration)
    finally:
        logger.info("Restoring original success_ratio values...")
        restore_success_ratios(original_ratios)
        _stop_git_versioning(git_worktree)


if __name__ == "__main__":
    main()
