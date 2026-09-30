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

"""A criterion no tool in the network can satisfy is not an agent defect."""

from pathlib import Path

from apps.network_consultant import run
from apps.network_consultant.run import UNGROUNDED_PREFIX
from apps.network_consultant.run import diagnosis_prompt
from apps.network_consultant.run import extract_prefixed

REPLY = """\
hr_check_vacation_balance_today.hocon: fixed AbsenceManagement
UNGROUNDED: it_password_reset_gsd_ticket.hocon: URLProvider: Includes the GSD URL
UNGROUNDED: legal_ambiguous.hocon: URLProvider: Includes at least one internal URL
"""

FAILURE = {"fixture": "a.hocon", "message": "boom", "path": None}


def _failures(tmp_path):
    fixture = tmp_path / "a.hocon"
    fixture.write_text("{}", encoding="utf-8")
    return [{**FAILURE, "path": str(fixture)}]


def test_ungrounded_lines_are_extracted():
    assert extract_prefixed(REPLY, UNGROUNDED_PREFIX) == [
        "it_password_reset_gsd_ticket.hocon: URLProvider: Includes the GSD URL",
        "legal_ambiguous.hocon: URLProvider: Includes at least one internal URL",
    ]


def test_they_are_written_where_nsflow_can_surface_them(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "NSFLOW_JOB_ID", "job1")
    monkeypatch.setattr(run, "NSFLOW_JOB_DIR", str(tmp_path))
    run._write_ungrounded(["a.hocon: URLProvider: Includes the GSD URL"])
    assert "URLProvider" in (tmp_path / "job1.ungrounded.txt").read_text(encoding="utf-8")


def test_no_file_outside_an_nsflow_job(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "NSFLOW_JOB_ID", None)
    monkeypatch.setattr(run, "NSFLOW_JOB_DIR", str(tmp_path))
    run._write_ungrounded(["x"])
    assert list(tmp_path.iterdir()) == []


def test_the_prompt_forbids_calling_it_an_agent_fix(tmp_path):
    """The whole point: misclassifying this as AGENT FIX is what burned four rounds."""
    prompt = diagnosis_prompt(_failures(tmp_path), "keep behaviour", 6, False)
    assert "UNGROUNDED" in prompt
    assert "do NOT classify it as AGENT FIX" in prompt
    assert "fabricated answer is worse than a failing test" in prompt


def test_the_prompt_carries_the_chosen_policy(tmp_path):
    stop = diagnosis_prompt(_failures(tmp_path), "d", 6, False, "stop")
    keep = diagnosis_prompt(_failures(tmp_path), "d", 6, False, "continue")
    assert "change neither the network nor the fixture" in stop
    assert "fixture_expectation_fixer" not in stop
    assert "fixture_expectation_fixer" in keep


def test_the_consultant_registry_declares_the_class_and_the_fixer_exception():
    """Runner-side reporting is useless if the consultant never emits the line.

    Deliberately pins the CONTRACT, not the prose: the failure-class wording is edited by hand
    and a test that locks the exact sentences just breaks every time someone tightens them.
    What must hold is that the class exists, that it names the output format the runner parses
    with UNGROUNDED_PREFIX, and that fixture_expectation_fixer has its exception -- it otherwise
    refuses whenever the network is behaving correctly, which is exactly this case.
    """
    consultant = Path("registries/agent_network_consultant.hocon").read_text(encoding="utf-8")
    assert "UNGROUNDED:" in consultant
    assert "`UNGROUNDED: <fixture>: <toolname>:" in consultant
    assert "read_thinking_trace" in consultant
    assert "2b. EXCEPTION" in consultant
