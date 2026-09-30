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

"""A round's score, which the plateau counter and the best-so-far snapshot both read."""

from apps.network_consultant.run import PLATEAU_STRIKES
from apps.network_consultant.scoring import ConsultantScoring


def _round(*fixtures):
    """(passed, criteria_passed, criteria_total) per fixture."""
    return [
        {"fixture": f"f{i}.hocon", "passed": passed, "criteria_passed": met, "criteria_total": total}
        for i, (passed, met, total) in enumerate(fixtures)
    ]


def test_progress_within_failing_fixtures_is_visible():
    """The intranet_agents_with_tools shape: no fixture flips for three rounds while the
    network goes from meeting 4 of 15 criteria to 14. Counting fixtures alone saw nothing."""
    rounds = [
        _round((False, 2, 5), (False, 1, 5), (False, 1, 5)),  # 4/15
        _round((False, 4, 5), (False, 3, 5), (False, 3, 5)),  # 10/15
        _round((False, 5, 5), (False, 5, 5), (False, 4, 5)),  # 14/15
    ]
    scores = [ConsultantScoring.round_score(r) for r in rounds]

    assert [s[1] for s in scores] == [4, 10, 14]
    assert scores[0] < scores[1] < scores[2], "each round must register as an improvement"

    # And so the plateau counter never fires on it.
    best, stale = None, 0
    for score in scores:
        stale = 0 if best is None or score > best else stale + 1
        best = score if best is None else max(best, score)
    assert stale == 0 < PLATEAU_STRIKES


def test_a_passing_fixture_outranks_more_criteria_elsewhere():
    """Fixtures stay the primary term -- passing them is the goal, not maximising criteria."""
    two_passing = _round((True, 3, 3), (True, 3, 3), (False, 0, 9))
    more_criteria = _round((False, 2, 3), (False, 2, 3), (False, 8, 9))

    assert ConsultantScoring.round_score(two_passing) == (2, 6)
    assert ConsultantScoring.round_score(more_criteria) == (0, 12)
    assert ConsultantScoring.round_score(two_passing) > ConsultantScoring.round_score(more_criteria)


def test_a_genuine_plateau_still_strikes_out():
    """The counter must keep working -- the fix is sensitivity, not disabling it."""
    stuck = _round((False, 2, 5), (False, 1, 5))
    best, stale = None, 0
    for _ in range(PLATEAU_STRIKES + 1):
        score = ConsultantScoring.round_score(stuck)
        stale = 0 if best is None or score > best else stale + 1
        best = score if best is None else max(best, score)
    assert stale >= PLATEAU_STRIKES


def test_a_regression_does_not_count_as_improvement():
    before, after = _round((True, 5, 5), (False, 3, 5)), _round((False, 4, 5), (False, 3, 5))
    assert ConsultantScoring.round_score(after) < ConsultantScoring.round_score(before)


def test_results_without_criteria_counts_still_score():
    """Infrastructure errors never reach a verdict and carry no criteria."""
    assert ConsultantScoring.round_score([{"fixture": "a", "passed": False, "infrastructure_error": True}]) == (0, 0)
