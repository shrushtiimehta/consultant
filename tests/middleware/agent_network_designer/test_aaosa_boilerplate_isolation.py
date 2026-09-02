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

"""AAOSA boilerplate must not reach an editor, nor get duplicated on write-back.

Both halves of the same concern: what the editor READS must be the agent's own text
(or it copies the protocol vocabulary into instructions it writes), and what the editor
WRITES must keep the source's ${...} tokens exactly as they were.
"""

import json

from middleware.agent_network_designer.agent_network_definition_middleware import AgentNetworkDefinitionMiddleware
from middleware.agent_network_designer.persistence.source_preserving_hocon_editor import SourcePreservingHoconEditor

# The resolved shape of registries/aaosa.hocon's aaosa_call, which every AAOSA agent gets
# merged into its "function" via ${aaosa_call}. The mode description is the leak vector.
AAOSA_PARAMETERS = {
    "type": "object",
    "properties": {
        "inquiry": {"type": "string", "description": "The inquiry"},
        "mode": {
            "type": "string",
            "description": (
                "'Determine' to ask the agent if the inquiry belongs to it.\n"
                "'Fulfill' to ask the agent to fulfill the inquiry, if it can.\n"
                "'Follow up' to ask the agent to respond to a follow up."
            ),
        },
    },
    "required": ["inquiry", "mode"],
}

OWN_PARAMETERS = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "An employee inquiry."}},
    "required": ["query"],
}

RESOLVED_CONFIG = {
    "aaosa_call": {"description": "Depending on the mode, ...", "parameters": AAOSA_PARAMETERS},
    "aaosa_instructions": "When you receive an inquiry, you will: ... <Determine | Fulfill> ...",
    "tools": [
        # Entry point: declares its own parameters, so they must survive.
        {"name": "front_man", "function": {"description": "Entry.", "parameters": OWN_PARAMETERS}},
        # Down-chain: parameters come entirely from ${aaosa_call}.
        {"name": "HR_agent", "function": {"description": "HR.", "parameters": AAOSA_PARAMETERS}},
    ],
}

# One entry-point agent (expertise scoping first, then aaosa) and one plain down-chain agent.
SOURCE_HOCON = """{
    include "registries/aaosa.hocon",

    "tools": [
        {
            "name": "front_man",
            "function": {
                "description": "Entry."
            },
            "instructions": \"\"\"
Old front man text.
            \"\"\" ${expertise_scoping_instructions} ${aaosa_instructions},
            "tools": ["HR_agent"]
        },
        {
            "name": "HR_agent",
            "function": ${aaosa_call}{
                "description": "HR."
            },
            "instructions": \"\"\"
Old HR text.
            \"\"\" ${aaosa_instructions},
        }
    ]
}
"""


class TestDiagnosticContextDropsAaosaSchema:
    """_build_diagnostic_context must not hand the AAOSA mode vocabulary to an editor."""

    def test_aaosa_parameters_are_dropped_but_real_ones_kept(self):
        network_def = {"front_man": {"instructions": "Own text."}, "HR_agent": {"instructions": "Own text."}}
        context = AgentNetworkDefinitionMiddleware._build_diagnostic_context(RESOLVED_CONFIG, network_def)

        blob = json.dumps(context)
        assert "Determine" not in blob
        assert "Fulfill" not in blob
        assert "Follow up" not in blob

        agents = {agent["name"]: agent for agent in context["tools"]}
        # The borrowed schema goes; the agent's own schema stays, description untouched.
        assert "parameters" not in agents["HR_agent"]["function"]
        assert agents["HR_agent"]["function"]["description"] == "HR."
        assert agents["front_man"]["function"]["parameters"] == OWN_PARAMETERS


class TestSourcePreservingEditorKeepsSubstitutionsIntact:
    """A patched field leaves the source's trailing ${...} run exactly as it was."""

    def test_entry_point_agent_does_not_gain_a_second_aaosa_substitution(self):
        """The run is `${expertise_scoping_instructions} ${aaosa_instructions}` -- scanning only
        the next token sees the wrong one and appends a duplicate."""
        updated = SourcePreservingHoconEditor.update_text(
            SOURCE_HOCON, {"front_man": {"instructions": "New front man text."}}
        )

        front_man_block = updated.split('"name": "HR_agent"')[0]
        assert front_man_block.count("${aaosa_instructions}") == 1
        assert front_man_block.count("${expertise_scoping_instructions}") == 1
        assert "New front man text." in front_man_block
        # Appended past the run, so aaosa still comes last as every registry writes it.
        assert "${expertise_scoping_instructions} ${aaosa_instructions}" in front_man_block

    def test_down_chain_agent_keeps_its_single_substitution(self):
        updated = SourcePreservingHoconEditor.update_text(SOURCE_HOCON, {"HR_agent": {"instructions": "New HR text."}})

        hr_block = updated.split('"name": "HR_agent"')[1]
        assert hr_block.count("${aaosa_instructions}") == 1
        assert "New HR text." in hr_block

    def test_missing_aaosa_substitution_is_added(self):
        """The append still has to happen when the source genuinely lacks it."""
        source = SOURCE_HOCON.replace('\\"\\"\\" ${aaosa_instructions},', '\\"\\"\\",')
        updated = SourcePreservingHoconEditor.update_text(source, {"HR_agent": {"instructions": "New HR text."}})

        hr_block = updated.split('"name": "HR_agent"')[1]
        assert hr_block.count("${aaosa_instructions}") == 1
