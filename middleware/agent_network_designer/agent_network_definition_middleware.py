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

import asyncio
import json
import os
import re
from copy import deepcopy
from json import JSONDecodeError
from logging import getLogger
from pathlib import Path
from re import Match
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import override

from botocore.exceptions import ClientError
from botocore.exceptions import NoCredentialsError
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from langchain.agents.middleware.types import ContextT
from langchain.agents.middleware.types import ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain.agents.middleware.types import ResponseT
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage
from leaf_common.resolution.resolver_util import ResolverUtil
from neuro_san.interfaces.agent_progress_reporter import AgentProgressReporter
from neuro_san.internals.persistence.abstract_async_config_restorer import AbstractAsyncConfigRestorer
from pyhocon import ConfigFactory
from pyhocon.config_tree import ConfigQuotedString
from pyhocon.config_tree import ConfigSubstitution
from pyhocon.config_tree import ConfigTree
from pyhocon.config_tree import ConfigValues
from pyhocon.exceptions import ConfigException
from pyparsing.exceptions import ParseException

from coded_tools.agent_network_editor.and_logger import AndLogger
from coded_tools.agent_network_editor.connectivity_dictionary_converter import ConnectivityDictionaryConverter
from coded_tools.agent_network_editor.constants import AGENT_NETWORK_DEFINITION
from coded_tools.agent_network_editor.constants import AGENT_NETWORK_DIAGNOSTIC_CONTEXT
from coded_tools.agent_network_editor.constants import AGENT_NETWORK_NAME
from coded_tools.agent_network_editor.constants import AGENT_NETWORK_SOURCE_FILE
from coded_tools.agent_network_editor.progress_handler import ProgressHandler
from coded_tools.agent_network_editor.sly_data_lock import SlyDataLock
from middleware.agent_network_designer.persistence.file_system_agent_network_persistor import DEFAULT_REGISTRIES_DIR

AGENT_NETWORK_HOCON_FILE: str = "agent_network_hocon_file"
AGENT_RESERVATIONS: str = "agent_reservations"
RESERVATION_ID: str = "reservation_id"
SKIP_DESIGNER: str = "skip_designer"


class AgentNetworkDefinitionMiddleware(AgentMiddleware):
    """
    Middleware that reads the agent network definition from sly_data and injects it
    into the system prompt before each model call.

    This allows the LLM to reason about the current agent network structure without
    requiring it to be passed explicitly through the chat stream.

    This middleware also anchors the progress-throttling contract of the editor
    coded tools: its aafter_agent hook flushes any progress report that
    ProgressHandler's throttle suppressed during the run (see flush_pending()).
    A network that wires the editor tools without registering this middleware
    silently loses that end-of-run flush.
    """

    def __init__(self, sly_data: dict[str, Any], progress_reporter: AgentProgressReporter | None = None) -> None:
        """
        Initialize agent network definition middleware.

        :param sly_data: A dictionary whose keys are defined by the agent hierarchy,
                but whose values are meant to be kept out of the chat stream.

                This dictionary is largely to be treated as read-only.
                It is possible to add key/value pairs to this dict that do not
                yet exist as a bulletin board, as long as the responsibility
                for which coded_tool publishes new entries is well understood
                by the agent chain implementation and the coded_tool implementation
                adding the data is not invoke()-ed more than once.

                Keys expected for this implementation are:
                    "agent_network_definition": an outline of an agent network
        :param progress_reporter: An optional AgentProgressReporter instance for
                reporting agent_network_definition to the client.
        """
        self.progress_reporter: AgentProgressReporter | None = progress_reporter
        self.sly_data = sly_data

        self.logger: AndLogger = AndLogger(getLogger(self.__class__.__name__))
        # Initialize agent network definition
        self.network_def: dict[str, Any] | list[dict[str, Any]] | None = None
        # Initialize an error message to store issues encountered during loading from HOCON file or S3 reservation.
        self.error_message: str = ""

    @override
    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        """
        Resolve and normalize the agent network definition before each model call.

        If loading from a HOCON file or S3 reservation fails, or if the agent network name is
        missing or invalid, reports the error back to the client and jumps to end.

        If skip_designer is set, normalizes the definition and jumps to end immediately so the
        persistence middleware can save the user-modified network without LLM involvement.

        Note that this is done before model, not before agent, because the definition may change
        between each model call (e.g., when the agent calls a tool that updates the network definition).

        :param state: Current agent state
        :param runtime: Runtime context
        :return: Dict with error message or skip notification and jump directive, or None to proceed normally
        """
        # Reset error_message before each resolve. In practice this is unreachable since a previous load
        # failure jumps to end and terminates the loop, but resetting here is a precaution against
        # stale errors persisting if the control flow ever changes.
        self.error_message = ""
        self.network_def = await self._resolve_network_def()
        agent_network_name: str = self.sly_data.get(AGENT_NETWORK_NAME)

        # Type check agent network name, but only report an error if no prior load error occurred,
        # since a load failure (e.g. missing HOCON file) may have prevented the name from being set.
        if agent_network_name is not None and not isinstance(agent_network_name, str) and not self.error_message:
            self.error_message = f"Error: {AGENT_NETWORK_NAME} has to be str. Got {type(agent_network_name).__name__}"
            self.logger.error(self.error_message)

        # If a network definition was resolved but the name is missing, report an actionable error.
        # Agent network name is required for persistence (saving the result back to S3 or disk).
        # This can happen when the user passes agent_network_definition directly without agent_network_name.
        # It does not apply to HOCON or S3 loading, which derive the name automatically.
        # Also skip if a prior error is already set (e.g. type error above) to avoid overwriting it.
        if self.network_def and not agent_network_name and not self.error_message:
            self.error_message = (
                f'Error: "{AGENT_NETWORK_NAME}" is missing from sly_data.\n'
                f'To edit an existing agent network, provide both "{AGENT_NETWORK_DEFINITION}" '
                f'and "{AGENT_NETWORK_NAME}" in sly_data.\n'
                f'Alternatively, provide the network via "{AGENT_NETWORK_HOCON_FILE}" or '
                f'"{AGENT_RESERVATIONS}" (with "{RESERVATION_ID}"), which supply the name automatically.'
            )
            self.logger.error(self.error_message)

        if self.error_message:
            # Loading errors (HOCON file or S3 reservation) only occur in the top-level agent_network_designer
            # network, not in its subnetworks, since loading is only triggered from the main network's sly_data.
            # Therefore, this jump will only fire in the agent_network_designer agent itself.
            return {
                "messages": [AIMessage(self.error_message)],
                "jump_to": "end",
            }

        # Normalize to dict format here so both the skip_designer and normal (awrap_model_call) paths
        # always receive a dict. Without this, a connectivity-list definition would reach the persistence
        # middleware as a list and crash validators that expect a dict (e.g. network_def.items()).
        if self.network_def:
            self.network_def = self._normalize_network_def(self.network_def)

            # This is used for manual editing where users modify the agent network definition and only want to use the
            # agent network designer to persist the changes, skipping the LLM entirely.
            # Strict boolean check to match the schema (type: boolean); "false" as a string would be truthy otherwise.
            if self.sly_data.get(SKIP_DESIGNER) is True and agent_network_name:
                return {
                    "messages": [
                        AIMessage(content=f"The network {agent_network_name} has been modified by the user.")
                    ],
                    "jump_to": "end",
                }
        return None

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """
        Inject the agent network definition into the system prompt before each model call.

        :param request: Model request containing messages and state
        :param handler: Handler to execute the model call
        :return: Model response from handler
        """
        if not self.network_def:
            return await handler(request)

        return await self._inject_into_request(self.network_def, request, handler)

    @override
    async def aafter_agent(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        """
        Flush any progress report that the throttle suppressed during this agent run.

        ProgressHandler's throttle drops (rather than delays) reports arriving within the
        throttle window, so without this hook a build whose final edit lands shortly after
        the previous sent report would leave the client's progress view permanently stale
        (issue #1257). Flushing here — when the agent loop exits normally, while the
        request and its journal are still alive — ensures the final network state goes
        out. (If the run aborts on an unhandled error, after-agent hooks are skipped and
        the throttled report stays dropped, matching pre-throttle behavior.)

        This matters most in the subnetworks (agent_network_editor and pals) and when those
        networks are used directly: their middleware has no progress_reporter by design
        (the client already receives the tools' own progress reports, so a middleware
        reporter would duplicate that stream). The flush therefore reuses the reporter
        stashed from the throttled tool call instead of needing one of its own.

        In the top-level designer this is effectively a no-op: its forced middleware
        reports (see _inject_into_request) clear the pending state on every model call.

        flush_pending contains its own error handling — this hook runs as a langgraph
        node, and an exception escaping it would replace the run's real final answer.

        :param state: Current agent state
        :param runtime: Runtime context
        :return: None to proceed normally
        """
        await ProgressHandler.flush_pending(self.sly_data)
        return None

    async def _resolve_network_def(self) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Resolve the agent network definition from sly_data, HOCON file, or S3 reservation.

        :return: Agent network definition, or None if not found
        """
        network_def: dict[str, Any] | list[dict[str, Any]] | None = self.sly_data.get(AGENT_NETWORK_DEFINITION)
        hocon_file: str | None = self.sly_data.get(AGENT_NETWORK_HOCON_FILE)
        agent_reservations: list[dict[str, Any]] | None = self.sly_data.get(AGENT_RESERVATIONS)

        # First, check to see if there is a generated agent network definition in sly_data.
        if network_def:
            # This log level is set to debug since this gets called before every model call and can be quite verbose.
            self.logger.debug(">>>>>>>>>>>>>>>>>>>Getting Agent Network Definition from Sly Data>>>>>>>>>>>>>>>>>>>")
            return network_def

        # Next, check to see if the user provides HOCON file via sly data
        if hocon_file:
            self.logger.info(
                ">>>>>>>>>>>>>>>>>>>Reading & Parsing from Agent Network HOCON File '%s'>>>>>>>>>>>>>>>>>>>",
                hocon_file,
            )
            network_def = await self._hocon_to_definition(hocon_file)
            # When loading from hocon, use the file name (without extension) as the agent network name.
            # This is because the agent network name is only created when using the CreateNetwork tool.
            if network_def:
                self.sly_data[AGENT_NETWORK_NAME] = Path(hocon_file).stem
                # Retain the resolved path too -- source-preserving persistence (see
                # AgentNetworkPersistenceMiddleware.preserve_source_hocon) writes back to this
                # exact file rather than reassembling one from AGENT_NETWORK_NAME.
                source_file: str | None = self._resolve_hocon_path(hocon_file)
                if source_file:
                    self.sly_data[AGENT_NETWORK_SOURCE_FILE] = source_file
            return network_def

        # Lastly, check the reservation ID in agent reservation field in sly data.
        if agent_reservations:
            return await self._resolve_network_def_from_s3(agent_reservations)

        return network_def

    def _extract_reservation_id(self, agent_reservations: list[dict[str, Any]] | None) -> str | None:
        """
        Validate agent_reservations and extract the reservation ID from the last entry.

        :param agent_reservations: A list of reservation structures. The last entry is expected to be a dict
                    with a 'reservation_id' key.
        :return: The reservation ID string, or None if the input is missing or malformed.
        """
        if not agent_reservations or not isinstance(agent_reservations, list):
            return None

        last_reservation: Any = agent_reservations[-1]
        if not isinstance(last_reservation, dict):
            self.logger.warning(
                "Warning: Last entry in '%s' is not a dict: %s (expected a dictionary)",
                AGENT_RESERVATIONS,
                type(last_reservation).__name__,
            )
            return None
        if RESERVATION_ID not in last_reservation:
            self.logger.warning(
                "Warning: No %s field in %s",
                RESERVATION_ID,
                last_reservation,
            )
            return None

        return last_reservation.get(RESERVATION_ID)

    async def _resolve_network_def_from_s3(
        self, agent_reservations: list[dict[str, Any]] | None
    ) -> dict[str, Any] | None:
        """
        Resolve the agent network definition from an S3 reservation.

        :param agent_reservations: A list of reservation structures describing the temporary agent networks that were
                    created by interacting with this agent. By convention, the last one in the list is a top-level
                    handle which may reference any others listed.
        :return: Agent network definition, or None if no reservation ID is provided
                    or if there are issues retrieving/parsing the reservation
        """
        reservation_id: str | None = self._extract_reservation_id(agent_reservations)
        if not reservation_id:
            return None

        error_message: str = "Error: Failed to load agent network definition from S3 reservation for unknown reasons."
        if not isinstance(reservation_id, str):
            error_message = (
                f"Error: Invalid '{RESERVATION_ID}' value: {type(reservation_id).__name__} "
                "(expected a non-empty string)."
            )
            self.logger.error(error_message)
            self.error_message = error_message
            return None

        # AWS credentials are picked up from the standard boto3 credential chain
        # (env vars, ~/.aws/credentials, IAM role on EC2/ECS/Lambda, etc.).
        # AGENT_RESERVATIONS_S3_BUCKET must point at the bucket holding the reservations.
        bucket: str = os.getenv("AGENT_RESERVATIONS_S3_BUCKET", "")
        if not bucket:
            error_message = (
                f"Error: AGENT_RESERVATIONS_S3_BUCKET is not set; cannot load reservation '{reservation_id}'."
            )
            self.logger.error(error_message)
            self.error_message = error_message
            return None

        config: dict[str, Any] | None = None
        try:
            config = await asyncio.to_thread(self.fetch_reservation_from_s3, bucket, reservation_id)
        except NoCredentialsError as creds_error:
            error_message = (
                f"Error: AWS credentials not found while loading reservation '{reservation_id}'. {creds_error}"
            )
            self.logger.error(error_message)
        except ClientError as client_error:
            error_code: str = client_error.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                error_message = f"Error: Reservation '{reservation_id}' not found in S3 bucket '{bucket}'."
            else:
                error_message = f"Error: Failed to retrieve reservation '{reservation_id}' from S3. {client_error}"
            self.logger.error(error_message)
        except JSONDecodeError as json_error:
            error_message = f"Error: Reservation '{reservation_id}' in S3 contains invalid JSON. {json_error}"
            self.logger.error(error_message)
        except ValueError as value_error:
            error_message = f"Error: Reservation '{reservation_id}' in S3 has unexpected shape. {value_error}"
            self.logger.error(error_message)

        if not config:
            self.error_message = error_message
            return None

        # When loading from s3, use extract the name from id and used as the agent network name.
        # This is because the agent network name is only created when using the CreateNetwork tool.
        self.sly_data[AGENT_NETWORK_NAME] = self._extract_name_from_reservation_id(reservation_id)
        self.logger.info(
            ">>>>>>>>>>>>>Reading & Parsing Agent Network Config from Reservation %s in %s S3 Bucket>>>>>>>>>>>>>>>>>",
            reservation_id,
            os.getenv("AGENT_RESERVATIONS_S3_BUCKET"),
        )
        return await self._config_to_network_def(config, reservation_id)

    @staticmethod
    def fetch_reservation_from_s3(bucket: str, reservation_id: str) -> dict[str, Any]:
        """
        Read a reservation JSON object from S3 and return the parsed config dict.

        Runs synchronously; call via ``asyncio.to_thread`` from async code so the boto3
        network I/O does not block the event loop. Mirrors the storage layout used by
        the neuro-san server (``reservations/<reservation_id>.json``) so the middleware
        can read what the server writes without depending on the server's internal
        storage classes.

        :param bucket: Target S3 bucket name
        :param reservation_id: Reservation ID whose JSON object should be fetched
        :return: Parsed JSON content as a dict (matches what ``AgentNetwork.get_config()``
                would return for the same reservation)
        """
        boto3_client = ResolverUtil.create_type("boto3.client", install_if_missing="boto3")
        s3 = boto3_client("s3")
        key: str = f"reservations/{reservation_id}.json"
        response: dict[str, Any] = s3.get_object(Bucket=bucket, Key=key)
        stream = response["Body"]
        try:
            body: bytes = stream.read()
        finally:
            stream.close()
        parsed: Any = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError(f"Reservation JSON must decode to an object, got {type(parsed).__name__}")
        return parsed

    def _normalize_network_def(self, network_def: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        """
        Ensure the network definition is in dict format, converting from connectivity list if needed,
        and cache it in sly_data.

        :param network_def: Network definition in dict or connectivity list format
        :return: Network definition as a dict
        """
        # The agent network definition can be provided in either:
        # - dict format (internal), used when creating or editing the network, or
        # - list format (connectivity), which is the native Neuro-San representation.
        # If the definition is in connectivity format, convert it to dict format before editing.
        if isinstance(network_def, list):
            connectivity_dict_converter = ConnectivityDictionaryConverter()
            network_def = connectivity_dict_converter.to_dict(network_def)
        # Cache the agent network definition as dict in sly_data for subsequent calls within the same session.
        self.sly_data[AGENT_NETWORK_DEFINITION] = network_def
        return network_def

    async def _inject_into_request(
        self,
        network_def: dict[str, Any],
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """
        Inject the network definition into the system prompt and invoke the handler.

        :param network_def: Agent network definition dict
        :param request: Model request containing messages and state
        :param handler: Handler to execute the model call
        :return: Model response from handler
        """
        self.logger.debug(
            ">>>>>>>>>>>>>>>>>>>Injecting Agent Network Definition into System Prompt>>>>>>>>>>>>>>>>>>>"
        )
        definition_prompt: str = self.format_definition_prompt(network_def)

        system_message: BaseMessage | None = request.system_message
        if system_message is not None:
            original_content: str = system_message.content if isinstance(system_message.content, str) else ""
            system_message = SystemMessage(content=f"{original_content}\n\n{definition_prompt}")
        else:
            system_message = SystemMessage(content=definition_prompt)

        if self.progress_reporter is not None:
            # Pass the real sly_data (the same dict instance the coded tools receive) so this
            # report shares the throttle bookkeeping with the tools and can look up the network
            # name. (The ToolboxFactory used for connectivity conversion is no longer kept on
            # sly_data — it is a process-wide cache on ConnectivityDictionaryConverter.)
            #
            # force=True keeps this report unthrottled: it fires at most once per model call of
            # the top-level designer (only the designer's middleware is configured with a
            # progress_reporter) — far less frequently than the editor tools in the subnetworks —
            # and it is what guarantees the client sees the fully merged network state, including
            # subnetwork edits whose own throttled reports may have been dropped, before each
            # designer model call.
            await ProgressHandler.report_progress(
                {"progress_reporter": self.progress_reporter},
                self.sly_data,
                network_def,
                self.sly_data.get(AGENT_NETWORK_NAME),
                force=True,
            )

        return await handler(request.override(system_message=system_message))

    def format_definition_prompt(self, network_def: dict[str, Any]) -> str:
        """
        Format the agent network definition as a system prompt section.

        :param network_def: The agent network definition dictionary
        :return: Formatted prompt string
        """
        # Prefer the full redacted HOCON view when a network was loaded from file: diagnosing a
        # failure needs llm_config/class/toolbox/max_iterations, which network_def drops. Falls
        # back to network_def for networks built in-session, which have no source config.
        diagnostic_context = self.sly_data.get(AGENT_NETWORK_DIAGNOSTIC_CONTEXT, network_def)
        definition_str: str = json.dumps(diagnostic_context, indent=2)
        return f"## Current Agent Network Diagnostic Context\n\n```json\n{definition_str}\n```"

    async def _hocon_to_definition(self, network_hocon_file: str | None) -> dict[str, Any] | None:
        """
        Convert hocon file path into agent network definition
        :param network_hocon_file: Agent network hocon file path

        :return: Agent network definition
        """
        config: dict[str, Any] | None = await self._hocon_to_config(network_hocon_file)
        if config is None:
            return None
        network_def = await self._config_to_network_def(config, network_hocon_file)
        if network_def is not None:
            await self._apply_unresolved_instructions(network_def, network_hocon_file)
            self.sly_data[AGENT_NETWORK_DIAGNOSTIC_CONTEXT] = self._build_diagnostic_context(config, network_def)
        return network_def

    async def _apply_unresolved_instructions(
        self, network_def: dict[str, Any], network_hocon_file: str | None
    ) -> None:
        """
        Replace each agent's instructions with the source literal, minus any ${...} substitution.

        The normal parse expands ${aaosa_instructions} / ${expertise_scoping_instructions} inline, so
        the editor reads shared boilerplate as if it were the agent's own text -- and then copies it
        forward into whatever it writes. In the source those substitutions are separate tokens
        concatenated OUTSIDE the quoted string, so re-parsing with resolve=False lets us keep the
        quoted part and drop them structurally: no text matching, and no dependence on which aaosa
        variant a given network happens to include (registries/aaosa.hocon and
        registries/aaosa_basic.hocon define different text under the same key).

        The hocon on disk is untouched -- nsflow and the runtime still load it resolved, with the
        variables expanded as usual. This only changes what the editor reads.

        Best-effort: if the unresolved parse fails, the already-resolved instructions are left in
        place rather than failing the load.

        :param network_def: Agent network definition to patch in place
        :param network_hocon_file: Agent network hocon file path
        """
        file_reference: str | None = self._resolve_hocon_path(network_hocon_file)
        if file_reference is None:
            return

        try:
            # basedir="." to match the cwd-relative include paths these registries use
            # (e.g. include "registries/aaosa.hocon"). With resolve=False an unreadable include
            # only warns, so the agent literals still come back intact either way.
            unresolved: ConfigTree = ConfigFactory.parse_string(
                Path(file_reference).read_text(encoding="utf-8"), basedir=".", resolve=False
            )
            agents: Any = unresolved.get("tools", [])
        except (OSError, ConfigException, ParseException) as error:
            self.logger.warning(
                "WARNING: Could not re-read '%s' unresolved; instructions may contain expanded "
                "substitutions. %s",
                file_reference,
                error,
            )
            return

        if not isinstance(agents, list):
            return

        for agent in agents:
            if not isinstance(agent, ConfigTree):
                continue
            agent_name: Any = agent.get("name", None)
            # Only touch agents that already came through with instructions -- function/toolbox
            # agents have none, and must keep not having the key.
            if not isinstance(agent_name, str) or agent_name not in network_def:
                continue
            if network_def[agent_name].get("instructions") is None:
                continue
            literal: str | None = self._literal_without_substitutions(agent.get("instructions", None))
            if literal:
                # Run through the same extractor the resolved path uses, so demo-mode text and the
                # legacy prefix still get removed and whitespace is normalized identically. Its
                # aaosa/expertise replacements simply find nothing left to strip.
                network_def[agent_name]["instructions"] = await self._extract_custom_instructions(literal)

    @staticmethod
    def _literal_without_substitutions(value: Any) -> str | None:
        """
        Return a value's source text with its ${...} substitution tokens dropped.

        A HOCON value built by concatenating strings and substitutions parses (unresolved) into a
        ConfigValues holding one token per part: plain str for triple-quoted blocks,
        ConfigQuotedString for "..." and ConfigSubstitution for each ${...}.

        :param value: An unresolved parsed value
        :return: The concatenated literal parts, or None if there are none
        """
        if isinstance(value, str):
            return value
        if not isinstance(value, ConfigValues):
            return None

        parts: list[str] = []
        for token in value.tokens:
            if isinstance(token, ConfigSubstitution):
                continue
            if isinstance(token, ConfigQuotedString):
                parts.append(token.value)
            elif isinstance(token, str):
                parts.append(token)

        return "".join(parts).strip() or None

    @classmethod
    def _build_diagnostic_context(cls, config: dict[str, Any], network_def: dict[str, Any]) -> dict[str, Any]:
        """Build a complete but redacted view of fields relevant to failure diagnosis."""
        omitted_root_keys = {
            "aaosa_call",
            "aaosa_command",
            "aaosa_instructions",
            "demo_mode",
            "instructions_prefix",
            "pii_patterns",
            "tools",
        }
        context = {key: deepcopy(value) for key, value in config.items() if key not in omitted_root_keys}
        diagnostic_agents = []
        for raw_agent in config.get("tools", []):
            if not isinstance(raw_agent, dict):
                diagnostic_agents.append(deepcopy(raw_agent))
                continue
            agent = deepcopy(raw_agent)
            agent_name = agent.get("name")
            # Use the boilerplate-stripped instructions, so the diagnostic view does not
            # re-introduce the aaosa/expertise text _extract_custom_instructions just removed.
            if agent_name in network_def and "instructions" in agent:
                agent["instructions"] = network_def[agent_name].get("instructions", "")
            diagnostic_agents.append(agent)
        context["tools"] = diagnostic_agents
        return cls._redact_sensitive_values(context)

    @classmethod
    def _redact_sensitive_values(cls, value: Any, key_name: str = "") -> Any:
        """Redact secret-bearing keys and common credential-shaped string values recursively."""
        if re.search(r"(?:api[_-]?key|authorization|credential|password|secret|token)", key_name, re.IGNORECASE):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {key: cls._redact_sensitive_values(item, str(key)) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._redact_sensitive_values(item, key_name) for item in value]
        if isinstance(value, str) and re.search(
            r"(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{35}|gh[pousr]_[A-Za-z0-9]+|"
            r"xox[baprs]-[A-Za-z0-9-]+|Bearer\s+[A-Za-z0-9._~+/=-]{20,})",
            value,
        ):
            return "[REDACTED]"
        return value

    def _resolve_hocon_path(self, network_hocon_file: str | None) -> str | None:
        """
        Validate and resolve a user-supplied HOCON file reference into a concrete path string.

        Resolution order:
          1. Absolute paths (POSIX-rooted, or Windows with drive/UNC anchor) are used as-is.
          2. Paths relative to cwd (typically the repo root) — if the input resolves to an
             existing file under cwd, it is used as-is. This covers paths copied from the
             repo tree such as "registries/generated/foo.hocon".
          3. Otherwise, paths are resolved against ``base_dir`` — the directory of the
             first entry in ``AGENT_MANIFEST_FILE`` (a whitespace-separated list of
             manifest files), or ``DEFAULT_REGISTRIES_DIR`` when the env var is empty or
             unset. This mirrors ``FileSystemAgentNetworkPersistor`` so loads and saves
             agree on file location.

        Backslashes in the input are normalized to forward slashes so Windows-style paths
        work on POSIX (and vice versa).

        On invalid input, sets ``self.error_message`` and returns None.

        :param network_hocon_file: Agent network hocon file path
        :return: The resolved file reference as a forward-slash path string, or None if invalid
        """
        if not isinstance(network_hocon_file, str) or not network_hocon_file.strip():
            error_message: str = (
                f"Error: Invalid network_hocon_file value: {type(network_hocon_file).__name__} "
                "(expected non-empty string)."
            )
            self.logger.error(error_message)
            self.error_message = error_message
            return None

        # Normalize backslashes so Windows-style input also works on POSIX.
        normalized: str = network_hocon_file.strip().replace("\\", "/")
        candidate: Path = Path(normalized)
        # Treat as absolute only if pathlib agrees AND, on Windows, the path has a drive
        # letter (e.g. "C:/...") or a UNC anchor (e.g. "//server/share/..."). On Windows
        # a bare "/foo" is "drive-rooted": Python 3.13+ reports is_absolute() == True for
        # it, but the path is ambiguous without a drive, so we fall through to the
        # relative branch where the leading slash is stripped — preventing the input
        # from bypassing base_dir.
        if candidate.is_absolute() and (os.name != "nt" or candidate.drive):
            return candidate.as_posix()

        # Strip leading separators so a user-supplied "/foo.hocon" cannot escape base_dir.
        # POSIX absolute paths are handled above; this catches the Windows drive-rooted
        # case where Path() would otherwise discard base_dir when joining with a rooted
        # right-hand side.
        trimmed_input: str = normalized.lstrip("/")

        # If the input resolves to an existing file relative to cwd (typically the repo
        # root when running the server from the project directory), use it as-is. This
        # covers any repo-root-relative path, including "registries/generated/foo.hocon"
        # or files outside the registries folder.
        if Path(trimmed_input).is_file():
            return trimmed_input

        # Derive the base registries directory from AGENT_MANIFEST_FILE (the dirname of the
        # first listed manifest), falling back to the default registries directory.
        agent_manifest_file: str = os.environ.get("AGENT_MANIFEST_FILE", "")
        manifest_parts: list[str] = agent_manifest_file.split()
        base_dir: str = os.path.dirname(manifest_parts[0]) if manifest_parts else DEFAULT_REGISTRIES_DIR
        return (Path(base_dir) / trimmed_input).as_posix()

    async def _hocon_to_config(self, network_hocon_file: str | None) -> dict[str, Any] | None:
        """
        Read and parse an agent network config file into a raw config dictionary.

        ``AbstractAsyncConfigRestorer`` accepts both ``.hocon`` and ``.json`` files, so JSON
        inputs work as well even though the surrounding API is named for HOCON.

        :param network_hocon_file: Agent network config file path (absolute or relative);
                see ``_resolve_hocon_path`` for resolution rules
        :return: Parsed config contents as a dict, or None if the file is invalid, has an
                unsupported extension, fails to parse, or cannot be read
        """
        file_reference: str | None = self._resolve_hocon_path(network_hocon_file)
        if file_reference is None:
            return None

        # Note we don't need to cache this because we only expect to read the file once.
        try:
            hocon = AbstractAsyncConfigRestorer(file_purpose="get_agent_network_definition", must_exist=True)
            return await hocon.async_restore(file_reference=file_reference)
        except FileNotFoundError:
            error_message: str = f"Error: Agent network config file not found: {file_reference}"
            self.logger.error(error_message)
            self.error_message = error_message
            return None
        except OSError as os_error:
            # Catches PermissionError, IsADirectoryError, and other OS-level read failures
            # whose specific subclasses differ across operating systems.
            error_message = f"Error: Failed to read agent network config file '{file_reference}'. {os_error}"
            self.logger.error(error_message)
            self.error_message = error_message
            return None
        except ValueError as value_error:
            # Raised by AbstractAsyncConfigRestorer when the file extension is not .hocon or .json.
            error_message = f"Error: Unsupported agent network config file '{file_reference}'. {value_error}"
            self.logger.error(error_message)
            self.error_message = error_message
            return None
        except ParseException as parse_error:
            # AbstractAsyncConfigRestorer wraps HOCON/JSON parse failures (ParseException,
            # ParseSyntaxException, JSONDecodeError, ConfigException) into ParseException.
            error_message = f"Error: Failed to parse agent network config file '{file_reference}'. {parse_error}"
            self.logger.error(error_message)
            self.error_message = error_message
            return None

    async def _config_to_network_def(self, config: dict[str, Any], source: str) -> dict[str, Any] | None:
        """
        Convert a parsed HOCON config dictionary into an agent network definition.

        :param config: Parsed HOCON config
        :param source: Identifier for the config source (hocon file path or reservation ID), used for error messages
        :return: Agent network definition, or None on failure
        """
        agents: list[dict[str, Any]] | None = config.get("tools")
        if not isinstance(agents, list):
            msg: str = "No field 'tools' found" if agents is None else "The 'tools' field is not a list"
            error_message: str = f"Error: {msg} in config from {source}."
            self.logger.error(error_message)
            self.error_message = error_message
            return None

        network_def: dict[str, Any] = {}
        for agent in agents:
            name, agent_def = await self._parse_agent(agent, source)
            if name is not None:
                network_def[name] = agent_def

        return network_def

    async def _parse_agent(self, agent: Any, source: str) -> tuple[str | None, dict[str, Any]]:
        """
        Parse a single agent entry from the hocon 'tools' list.

        :param agent: A single entry from the 'tools' list in the hocon file
        :param source: Identifier for the config source (hocon file path or reservation ID), used for warning messages

        :return: (agent_name, agent_def) where agent_name is None if the entry should be skipped
        """
        if not isinstance(agent, dict):
            self.logger.warning("WARNING: Skipping non-dict entry in 'tools' list in '%s': %r", source, agent)
            return None, {}

        agent_name: str | None = agent.get("name")
        if not isinstance(agent_name, str) or not agent_name:
            self.logger.warning("WARNING: Skipping agent with missing/invalid 'name' in '%s': %r", source, agent)
            return None, {}

        # Only extract agents info and only "instructions" and "tools" parts
        agent_def: dict[str, Any] = {}

        instructions: str | None = agent.get("instructions")
        if instructions is not None:
            if not isinstance(instructions, str):
                self.logger.warning(
                    "WARNING: Skipping agent %s due to non-string 'instructions' in '%s'",
                    agent_name,
                    source,
                )
                return None, {}
            if instructions.strip():
                # Extract only the unique instructions
                # (remove aaosa instructions, instructions prefix, and demo mode)
                agent_def["instructions"] = await self._extract_custom_instructions(instructions.strip())

            # Initialize description for non-function agents so the description setter
            # can distinguish them from function/toolbox agents (which have no description key).
            agent_def["description"] = ""

        function: dict[str, Any] = agent.get("function", {})
        description: str | None = function.get("description") if isinstance(function, dict) else None
        if description is not None:
            if not isinstance(description, str):
                self.logger.warning(
                    "WARNING: Skipping agent %s due to non-string 'description' in '%s'",
                    agent_name,
                    source,
                )
                return None, {}
            if description.strip():
                agent_def["description"] = description.strip()

        tools: list[str] | None = agent.get("tools")
        if tools:
            agent_def["tools"] = tools

        return agent_name, agent_def

    async def _extract_custom_instructions(self, instructions: str) -> str:
        """
        Extract the custom part of instructions, excluding aaosa instructions, expertise-scoping
        instructions, and demo mode.
        :param instructions: The full instructions of an agent.

        :return: The part of instructions that is unique to the agent.
        """

        # Legacy prefix some older networks still have baked in verbatim, from before
        # expertise_scoping_instructions.hocon was simplified to drop this lead-in sentence.
        legacy_prefix_pattern = r"You are part of a \w+ of assistants\.\s*"

        demo_mode = (
            "You are part of a demo system, so when queried, make up a realistic response as if "
            "you are actually grounded in real data or you are operating a real application API or microservice."
        )

        # Both substitution files define their text with real newlines, but custom_part gets its
        # whitespace collapsed to single spaces below -- normalize these the same way, or the
        # .replace() calls never match and the boilerplate silently survives into custom_part.
        aaosa_instructions: str = " ".join((await self._get_aaosa_instructions()).split())
        expertise_scoping_instructions: str = " ".join((await self._get_expertise_scoping_instructions()).split())

        # Clean and normalize the input
        custom_part: str = instructions.strip()
        custom_part = re.sub(r"\s+", " ", custom_part)  # Normalize whitespace

        # Remove the legacy prefix sentence, if present
        custom_part = re.sub(legacy_prefix_pattern, "", custom_part).strip()

        # Remove aaosa text and expertise-scoping text -- loaded from the same source files that
        # get substituted into the actual hocon, so this always matches what's really there instead
        # of a hardcoded guess that can drift out of sync and let boilerplate leak into custom_part.
        custom_part = custom_part.replace(aaosa_instructions, "").strip()
        custom_part = custom_part.replace(expertise_scoping_instructions, "").strip()

        # Remove demo mode text
        custom_part = custom_part.replace(demo_mode.strip(), "").strip()

        # Clean up any extra whitespace
        custom_part = " ".join(custom_part.split())

        return custom_part

    async def _get_aaosa_instructions(self) -> str:
        """
        Get aaosa instructions potentially from cache in sly_data

        :return: aaosa instructions
        """
        return await self._get_cached_substitution("aaosa_instructions", "registries/aaosa.hocon")

    async def _get_expertise_scoping_instructions(self) -> str:
        """
        Get expertise-scoping instructions potentially from cache in sly_data

        :return: expertise-scoping instructions
        """
        return await self._get_cached_substitution(
            "expertise_scoping_instructions", "registries/expertise_scoping_instructions.hocon"
        )

    async def _get_cached_substitution(self, key: str, hocon_file: str) -> str:
        """
        Load a single-key substitution hocon file (e.g. aaosa_instructions, expertise_scoping_instructions),
        caching the result in sly_data so it's only read from disk once per session.

        :param key: The top-level hocon key holding the substitution text.
        :param hocon_file: Path to the hocon file defining that key.
        :return: The substitution text, or "" if the file/key doesn't exist.
        """
        value: str = ""

        async with await SlyDataLock.get_lock(self.sly_data, f"{key}_lock"):
            value = self.sly_data.get(key)
            if value is not None:
                # Return early with cached value
                return value

            # Get from file
            try:
                hocon = AbstractAsyncConfigRestorer(
                    file_purpose="get_agent_network_definition - custom instructions", must_exist=True
                )
                config: dict[str, Any] = await hocon.async_restore(file_reference=hocon_file)
                value = config.get(key, "")
            except FileNotFoundError:
                value = ""

            # Cache the loaded value in sly_data for subsequent calls
            self.sly_data[key] = value

        return value

    def _extract_name_from_reservation_id(self, reservation_id: str) -> str:
        # re.search() scans through the string looking for the UUID pattern
        # The pattern explained:
        #   -           matches a literal hyphen (separator between name and UUID)
        #   [0-9a-f]    matches any hex character (digits 0-9 or letters a-f)
        #   {8}         exactly 8 hex characters  → "550e8400"
        #   -           literal hyphen
        #   [0-9a-f]{4} exactly 4 hex characters  → "e29b"
        #   -           literal hyphen
        #   [0-9a-f]{4} exactly 4 hex characters  → "41d4"
        #   -           literal hyphen
        #   [0-9a-f]{4} exactly 4 hex characters  → "a716"
        #   -           literal hyphen
        #   [0-9a-f]{12} exactly 12 hex characters → "446655440000"
        #   $           end of string (UUID must be at the very end)
        #
        # re.IGNORECASE makes it match both uppercase and lowercase hex (a-f or A-F)
        match: Match | None = re.search(
            r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", reservation_id, re.IGNORECASE
        )

        # match.start() gives the index where the UUID pattern begins in the string
        # reservation_id[:match.start()] slices the string from the beginning up to (not including) that index
        # if no UUID is found (match is None), we just return the original string unchanged
        return reservation_id[: match.start()] if match else reservation_id
