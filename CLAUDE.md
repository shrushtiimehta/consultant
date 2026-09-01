# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this repo is

- **neuro-san** (`pip` package, source at `/Users/978131/Projects/neuro-san`) — the multi-agent
  orchestration framework. Agent networks are declared in HOCON, agents talk via the AAOSA
  protocol, custom Python logic plugs in as "coded tools."
- **neuro-san-studio** — this repo. It's the playground/product built on top of neuro-san:
  ready-made agent networks (`registries/*.hocon`), the coded tools those networks call
  (`coded_tools/`, `neuro_san_studio/coded_tools/`), apps (`apps/`), and the nsflow UI (`nsflow/`).

Don't reimplement anything neuro-san already provides (registry loading, session/agent
orchestration, sly-data plumbing, logging/tracing) — import and use it. This repo's job is
tools, networks, and apps, not a second copy of the framework.

## OOP patterns to follow

### Coded tools inherit `CodedTool`

Every coded tool subclasses `neuro_san.interfaces.coded_tool.CodedTool` and implements
**`async_invoke`** (preferred) — only fall back to sync `invoke` for trivial/test tools, since
sync blocks the whole agent event loop.

```python
from neuro_san.interfaces.coded_tool import CodedTool

class MyTool(CodedTool):
    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        ...
```

See [coded_tools/basic/accountant.py](coded_tools/basic/accountant.py) for the minimal shape,
`neuro_san_studio/coded_tools/base_rag.py` for a heavier example: an `ABC` base class
(`BaseRag`) that concrete RAG tools (`pdf_rag.py`, `arxiv_rag.py`, `wikipedia_rag.py`,
`confluence_rag.py`, ...) subclass and override `abstractmethod`s for. **Before writing a new
tool, check if it's a RAG variant, a search tool, or similar to something already in
`neuro_san_studio/coded_tools/` or `coded_tools/` — subclass the existing base rather than
copy-pasting invoke logic.**

### Middleware classes for cross-cutting concerns

`middleware/agent_network_designer/` follows a definition/persistence/validation split
(`AgentNetworkDefinitionMiddleware`, `agent_network_persistence_middleware.py`,
`validation/`). If you're adding logic that loads, mutates, or persists a network HOCON, put it
behind one of these middleware classes instead of hand-rolling file I/O in a coded tool —
reuse `AgentNetworkDefinitionMiddleware`/`AgentNetworkPersistenceMiddleware` for anything that
reads or writes a network's source `.hocon`.

### Registries compose via HOCON `include`, not duplication

Networks in `registries/*.hocon` share structure via `include "registries/aaosa_basic.hocon"`
plus a `manifest.hocon` per directory. When adding a network, include the closest existing base
rather than pasting agent boilerplate.

## Reuse checklist — don't reinvent the wheel

Before writing new code, check in this order:

1. **neuro-san itself** — orchestration, sessions, sly-data, HOCON parsing, logging/tracing are
   all provided. Never re-implement framework plumbing here.
2. **An existing coded tool** — `coded_tools/basic/`, `coded_tools/industry/`,
   `neuro_san_studio/coded_tools/` likely already has something close (RAG variants, search
   tools, file tools in `neuro_san_studio/coded_tools/file_management/`). Subclass or extend,
   don't duplicate.
3. **An existing base class** — `BaseRag` (`neuro_san_studio/coded_tools/base_rag.py`),
   `CodedTool` (neuro-san), `BasePlugin` (`neuro_san_studio/interfaces/base_plugin.py`).
4. **An existing middleware class** — for network load/validate/persist, use
   `middleware/agent_network_designer/` and `middleware/persistent_memory/`, not ad hoc file
   handling.
5. **An existing registry to `include`** — check `registries/aaosa*.hocon` and sibling
   `manifest.hocon`s before writing a network from scratch.
6. **A dependency already in `requirements.txt`** — langchain, openai, anthropic clients, etc.
   are already wired; don't add a new library for what one of these already does.

Only after checking all of the above does new code get written, and it should live as one
tool/middleware/network, not a new abstraction layer.

## Conventions (from CONTRIBUTING.md)

- Python 3.12/3.13, Ruff for lint/format, line length 119.
- `snake_case` functions/vars, `PascalCase` classes, `UPPER_CASE` constants (Google style).
- Docstrings on public functions/classes/modules — match the `:param:`/`:return:` style already
  used in `coded_tools/basic/accountant.py`.
