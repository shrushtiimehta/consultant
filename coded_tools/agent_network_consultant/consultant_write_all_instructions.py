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

"""Consultant-owned instruction writer that records surgical source changes."""

import logging
from typing import Any

from coded_tools.agent_network_consultant.consultant_state import ConsultantState
from coded_tools.agent_network_editor.and_logger import AndLogger
from coded_tools.agent_network_editor.constants import AGENT_NETWORK_DEFINITION
from coded_tools.agent_network_instructions_editor.write_all_instructions import WriteAllInstructions


class ConsultantWriteAllInstructions(WriteAllInstructions):
    """Preserve the shared writer behavior while recording changed fields."""

    @staticmethod
    def apply_writer_response(agent_name: str, response: str, sly_data: dict[str, Any]) -> str:
        """Apply validated instruction fields and record only the changed source fields."""
        network_def: dict[str, Any] = sly_data.get(AGENT_NETWORK_DEFINITION)
        if not network_def:
            return "Error: No network in sly data!"
        if agent_name not in network_def:
            return f"Error: Agent not found: {agent_name}"
        network_agent = network_def.get(agent_name, {})
        if network_agent.get("instructions") is None:
            return f"Error: Agent has no instructions field: {agent_name}. It is a function agent."

        updates, error = WriteAllInstructions._validate_writer_fields(response)
        if error:
            return error

        logger = AndLogger(logging.getLogger(ConsultantWriteAllInstructions.__name__))
        if not updates:
            logger.info("Writer reported no change needed for '%s'", agent_name)
            return ""

        changes: dict[str, dict[str, str]] = sly_data.setdefault(ConsultantState.AGENT_NETWORK_CHANGES, {})
        for field, value in updates.items():
            network_agent.update({field: value})
            changes.setdefault(agent_name, {}).update({field: value})
            logger.info("Set %s for '%s' (%d chars)", field, agent_name, len(value))

        sly_data.update({AGENT_NETWORK_DEFINITION: network_def})
        return ""

    @staticmethod
    def _apply_writer_response(agent_name: str, response: str, sly_data: dict[str, Any]) -> str:
        """Override the shared writer hook without changing its invocation lifecycle."""
        return ConsultantWriteAllInstructions.apply_writer_response(agent_name, response, sly_data)
