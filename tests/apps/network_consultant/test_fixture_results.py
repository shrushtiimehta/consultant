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

"""The per-fixture results file the UI reads pass/fail and failure reasons from."""

import json

from apps.network_consultant import test_runner


def _job(monkeypatch, tmp_path):
    monkeypatch.setattr(test_runner, "NSFLOW_JOB_ID", "job1")
    monkeypatch.setattr(test_runner, "NSFLOW_JOB_DIR", str(tmp_path))
    return tmp_path / "job1.results.json"


def test_a_subset_round_keeps_the_verdicts_it_did_not_re_run(monkeypatch, tmp_path):
    """The whole point of merging: the runner re-tests only what was failing, and a re-check
    of one fixture says nothing about the others -- they must not vanish from the UI."""
    path = _job(monkeypatch, tmp_path)
    test_runner._write_fixture_results(
        [
            {"fixture": "a.hocon", "passed": True, "message": None, "infrastructure_error": False},
            {"fixture": "b.hocon", "passed": False, "message": "'owners' not found", "infrastructure_error": False},
        ]
    )
    test_runner._write_fixture_results(
        [{"fixture": "b.hocon", "passed": True, "message": None, "infrastructure_error": False}]
    )

    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert set(recorded) == {"a.hocon", "b.hocon"}
    assert recorded["a.hocon"]["passed"] is True
    assert recorded["b.hocon"]["passed"] is True
    assert recorded["b.hocon"]["message"] is None


def test_failure_reason_and_infrastructure_flag_survive(monkeypatch, tmp_path):
    path = _job(monkeypatch, tmp_path)
    test_runner._write_fixture_results(
        [{"fixture": "c.hocon", "passed": False, "message": "TIMEOUT_ISSUE: ...", "infrastructure_error": True}]
    )
    recorded = json.loads(path.read_text(encoding="utf-8"))["c.hocon"]
    assert recorded == {"passed": False, "message": "TIMEOUT_ISSUE: ...", "infrastructure_error": True}


def test_no_file_written_outside_an_nsflow_job(monkeypatch, tmp_path):
    """Plain CLI use must not litter -- and has nowhere to write to anyway."""
    monkeypatch.setattr(test_runner, "NSFLOW_JOB_ID", None)
    monkeypatch.setattr(test_runner, "NSFLOW_JOB_DIR", str(tmp_path))
    test_runner._write_fixture_results([{"fixture": "a.hocon", "passed": True}])
    assert list(tmp_path.iterdir()) == []


def test_a_corrupt_file_does_not_fail_the_test_run(monkeypatch, tmp_path):
    """The runner may be mid-write when something else reads it; losing old verdicts beats
    raising out of a suite that has already finished running."""
    path = _job(monkeypatch, tmp_path)
    path.write_text("{not json", encoding="utf-8")
    test_runner._write_fixture_results([{"fixture": "a.hocon", "passed": True}])
    assert json.loads(path.read_text(encoding="utf-8"))["a.hocon"]["passed"] is True
