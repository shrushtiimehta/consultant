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

"""Characterization tests for consultant-owned instruction change tracking."""

import json

from coded_tools.agent_network_consultant.state import ConsultantState
from coded_tools.agent_network_consultant.write_all_instructions import ConsultantWriteAllInstructions
from coded_tools.agent_network_editor.constants import AGENT_NETWORK_DEFINITION


class TestConsultantWriteAllInstructions:
    """Prove the consultant changes only fields returned by the writer."""

    def test_records_only_applied_fields(self) -> None:
        """Track the exact surgical patch while retaining other agent fields."""
        sly_data = {
            AGENT_NETWORK_DEFINITION: {
                "greeter": {
                    "instructions": "Say hello.",
                    "description": "Greets users.",
                    "tools": ["clock"],
                }
            }
        }
        response = json.dumps({"instructions": "Greet the user warmly."})

        error = ConsultantWriteAllInstructions._apply_writer_response(  # pylint: disable=protected-access
            "greeter", response, sly_data
        )

        assert error == ""
        assert sly_data.get(AGENT_NETWORK_DEFINITION).get("greeter") == {
            "instructions": "Greet the user warmly.",
            "description": "Greets users.",
            "tools": ["clock"],
        }
        assert sly_data.get(ConsultantState.AGENT_NETWORK_CHANGES) == {
            "greeter": {"instructions": "Greet the user warmly."}
        }

    def test_empty_update_is_a_no_op(self) -> None:
        """Keep the definition unchanged when the writer reports no necessary change."""
        definition = {"greeter": {"instructions": "Say hello.", "description": "Greets users."}}
        sly_data = {AGENT_NETWORK_DEFINITION: definition}

        error = ConsultantWriteAllInstructions._apply_writer_response(  # pylint: disable=protected-access
            "greeter", "{}", sly_data
        )

        assert error == ""
        assert sly_data.get(AGENT_NETWORK_DEFINITION) == definition
        assert sly_data.get(ConsultantState.AGENT_NETWORK_CHANGES) is None
