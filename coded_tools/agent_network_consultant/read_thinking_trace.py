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

import logging
import os
import re
from pathlib import Path
from typing import Any
from typing import Union

from neuro_san.interfaces.coded_tool import CodedTool

from coded_tools.agent_network_editor.and_logger import AndLogger

# Matches apps/network_consultant/test_runner.py's IMPROVEMENT_THINKING_DIR and the
# "--- <agent_origin> ---" section headers _write_consolidated_thinking writes.
THINKING_DIR = Path("logs/thinking_dir/improvement").resolve()
SECTION_HEADER = re.compile(r"^--- (.+) ---$", re.MULTILINE)

# Set by nsflow's backend (network_consultant_endpoints.py's _start_job) on the current
# process's environment when this whole session is running as one of its background jobs --
# unset for plain CLI usage (`python -m apps.network_consultant.runner` directly), in which
# case there is no per-job log file to read at all.
NSFLOW_JOB_ID = os.environ.get("NSFLOW_JOB_ID")
NSFLOW_JOB_DIR = os.environ.get("NSFLOW_JOB_DIR")
JOB_LOG_TAIL_LINES = 200


class ReadThinkingTrace(CodedTool):
    """
    CodedTool that lets the consultant's diagnosing sub-agents (network_behavior_fixer,
    fixture_expectation_fixer, structural_change_assessor) read back the per-agent reasoning
    trace saved for a failing fixture (see apps/network_consultant/test_runner.py's
    _write_consolidated_thinking), instead of every agent's full trace being force-fed into its
    context up front. Call with no `agent_name` first to see which agents have a trace worth
    reading; call again with `agent_name` to fetch that one agent's trace. Fixture-scoped only --
    for round-level context (all fixtures, not just one), see ReadJobLog instead.
    """

    def _parse_sections(self, content: str) -> dict[str, str]:
        """Split a consolidated thinking file back into {agent_origin: trace}."""
        headers = list(SECTION_HEADER.finditer(content))
        sections: dict[str, str] = {}
        for index, header in enumerate(headers):
            start = header.end()
            end = headers[index + 1].start() if index + 1 < len(headers) else len(content)
            sections[header.group(1)] = content[start:end].strip()
        return sections

    async def async_invoke(self, args: dict[str, Any], sly_data: dict[str, Any]) -> Union[dict[str, Any], str]:
        """
        :param args: A dictionary with the following keys:
                "fixture_name": the fixture's basename (e.g. "midday_coffee_all_open.hocon").
                "agent_name": optional; one of the agent origins returned by a prior call with
                    this same fixture_name and no agent_name. Omit to just list what's available.

        :param sly_data: Unused by this implementation.

        :return:
            With no agent_name: {"available_agents": [...]} -- the agent origins with a saved
                trace for this fixture.
            With agent_name: {"agent_name": ..., "trace": ...} -- that agent's full trace.
            otherwise: A text string error message.
        """
        logger = AndLogger(logging.getLogger(self.__class__.__name__))

        fixture_name: str = args.get("fixture_name", "")
        if not fixture_name:
            return "Error: No 'fixture_name' provided."

        safe_name = re.sub(r"[^\w.\-]", "_", Path(fixture_name).name)
        trace_path = (THINKING_DIR / f"{safe_name}.txt").resolve()
        try:
            trace_path.relative_to(THINKING_DIR)
        except ValueError:
            return "Error: fixture_name resolves outside the thinking-trace directory."
        if not trace_path.is_file():
            return f"Error: No saved thinking trace found for fixture '{fixture_name}'."

        logger.info("Reading thinking trace: %s", trace_path)
        content = trace_path.read_text(encoding="utf-8")
        sections = self._parse_sections(content)

        agent_name: str = args.get("agent_name", "")
        if not agent_name:
            return {"available_agents": list(sections.keys())}
        if agent_name not in sections:
            return (
                f"Error: No trace for agent '{agent_name}' in fixture '{fixture_name}'. "
                f"Available agents: {list(sections.keys())}"
            )
        return {"agent_name": agent_name, "trace": sections[agent_name]}


class ReadJobLog(CodedTool):
    """
    CodedTool that lets consultant read the tail of its own current nsflow job's raw
    log -- round-level context (iteration progress, which fixtures passed/failed this round,
    any traceback) rather than one fixture's own trace. The job is always the one this session
    is currently running as. For per-fixture, per-agent traces instead, see ReadThinkingTrace --
    the two are deliberately separate tools so this one only ever reaches sub-agents that need
    it and vice versa.
    """

    async def async_invoke(self, args: dict[str, Any], sly_data: dict[str, Any]) -> Union[dict[str, Any], str]:
        """
        :param args: A dictionary with the following key:
                "tail_lines": optional int; how many lines of log tail to return. Defaults to
                    JOB_LOG_TAIL_LINES when omitted.

        :param sly_data: Unused by this implementation.

        :return:
            {"job_log_tail": <the requested tail>} on success.
            otherwise: A text string error message (not running as an nsflow job, or its log
                file isn't available for some other reason).
        """
        logger = AndLogger(logging.getLogger(self.__class__.__name__))

        if not (NSFLOW_JOB_ID and NSFLOW_JOB_DIR):
            return "Error: Not running as an nsflow job -- there is no per-job log to read."

        tail_lines = args.get("tail_lines") or JOB_LOG_TAIL_LINES

        log_path = os.path.join(NSFLOW_JOB_DIR, f"{NSFLOW_JOB_ID}.log")
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
                lines = log_file.readlines()[-tail_lines:]
        except FileNotFoundError:
            return f"Error: Job log not found: {log_path}"

        logger.info("Reading job log: %s", log_path)
        return {"job_log_tail": "".join(lines)}
