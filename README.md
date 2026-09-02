# network-designer-toolkit

A fork of [`neuro-san-studio`](https://github.com/cognizant-ai-lab/neuro-san-studio) with the
agent-network designer/editor/consultant tooling vendored in. `neuro-san` and `nsflow` are still
PyPI dependencies (see `requirements.txt`); the studio itself is this checkout. The tooling:

- `registries/agent_network_designer.hocon` — creates/modifies agent networks from a use-case description.
- `registries/agent_network_editor.hocon`, `agent_network_instructions_editor.hocon`,
  `agent_network_query_generator.hocon` — the designer's own support networks.
- `registries/agent_network_test_generator.hocon` (ANTeGen) — generates data-driven test fixtures for a network.
- `registries/agent_network_consultant.hocon` — given a failing-test report for an existing network,
  fixes the responsible agent instructions, corrects bad test fixtures, delegates structural changes, and (when
  asked to reduce token usage) simplifies hierarchy/verbosity or suggests coded-tool conversions.
- `apps/network_consultant/` — iterative test-and-fix loop driven by `agent_network_consultant`.
  Generated fixtures land in `tests/fixtures/<network path>/`.

This is tooling, plus one example network to try it against:
`registries/basic/coffee_finder.hocon` — five AAOSA agents that answer "where can I get coffee
right now?" based on the time of day. Use it to see the loop work end to end before pointing
`--use-case` or `--hocon-file` at your own network.

## Setup

Do these in order. Steps 2 and 3 must come after step 1, or `uv sync` will reinstall the
PyPI copies over your forks.

### 1. This repo

This repo *is* the studio, so there is nothing to clone alongside it. Installing it pulls in
`neuro-san` and `nsflow` from PyPI:

```bash
git clone https://github.com/shrushtiimehta/consultant.git
cd consultant

make install
source venv/bin/activate
pip install neuro-san-studio
```

### 2. The forked `nsflow` (required)

Step 1 installed `nsflow` from PyPI, but this project needs a patched version. Clone it
somewhere outside this repo and install editable, so it takes precedence:

```bash
git clone -b network-consultant-compat https://github.com/shrushtiimehta/nsflow.git
pip install -e ./nsflow
```

It is a fork of the `cognizant-ai-lab` original. Editable (`-e`) means edits in that checkout
take effect here with no reinstall. 

### 3. Credentials and local registries

```bash
# API key(s) for whichever provider config/llm_config.hocon selects
echo 'OPENAI_API_KEY=sk-...' >> .env
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
```

## Usage

Two ways to drive it: the Network Consultant panel in the nsflow UI, or the CLI runner.
Start with the UI — it is the only one where you can watch the run and answer the
consultant's questions as they come up.

### UI

Start the server and the nsflow UI (`ns` comes from `neuro-san-studio`, inside the venv):

```bash
ns run
```

That serves neuro-san on **:8080** and nsflow on **:4173**. Open <http://localhost:4173>, then:

1. Pick the network to work on from the agent list.
2. Open the **Network Consultant** panel.
3. **Generate tests** for the network, then **Improve** to start the test-and-fix loop.
4. Watch progress in the LogsPanel — the fork mirrors the job's log into the session's channel.
5. If the run stops on a **NEEDS_CLARIFICATION** question, answer it inline and the job resumes.
   Anything reported as a **TOOL_ISSUE** is a broken coded tool needing a human code fix; an
   answer will not clear it.
6. **Stop** cancels a running job.

Backing routes, if you want to script against them:
`POST /api/v1/network_consultant/generate-tests`, `POST .../improve`,
`GET .../jobs/{job_id}`, `POST .../jobs/{job_id}/answer`, `POST .../jobs/{job_id}/stop`.

### CLI

Unattended runs. Note the CLI does **not** appear in the nsflow live view, even with
`--connection http` — nsflow only renders conversations started through its own websocket.
A run that hits a NEEDS_CLARIFICATION question has no way to answer it here; use the UI for that.

```bash
source venv/bin/activate
```

Build a brand-new network, generate tests, and iteratively fix it:

```bash
python -m apps.network_consultant.runner \
  --use-case "A pizza shop assistant that checks store hours, takes orders, and answers menu questions" \
  --test-level normal
```

Fix an existing network — the bundled example, to see the loop run end to end:

```bash
python -m apps.network_consultant.runner \
  --hocon-file basic/coffee_finder.hocon \
  --direction "Name the exact open venues for coffee or coffee-liquor requests, based on the time of day" \
  --test-level normal
```

The path is relative to `registries/`. For your own network, put the hocon under `registries/`
and add it to `registries/manifest.hocon` first — nothing is served until it is listed there,
and a request for an unregistered agent fails with `Agent named "..." not found in manifest file`.

Useful flags — full list via `--help`:

| Flag | Meaning |
| --- | --- |
| `--test-level {minimum,normal,max}` | How many fixtures ANTeGen generates. |
| `--max-iterations N` | Cap on test-and-fix rounds. |
| `--success-ratio N/M` | Ratio a fixture is bumped to once the fix is judged confident (default `3/3`). |
| `--connection {direct,http}` | `direct` (default) runs in-process, no server. `http` talks to a running `ns run`. |
| `--git-versions` | Commit each tested version of the network hocon to a `consultant-versions/<network>/<run-id>` branch and push to `origin`. Off by default — it pushes repeatedly during the run. |

## Notes

- Don't run more than one of these processes concurrently against this repo -- they share
  `registries/generated/manifest.hocon`, and concurrent writes can race.
- `registries/generated/` is where the designer persists new/modified networks. Nothing under it is tracked,
  so create it (with a `manifest.hocon` containing `{}`) on a fresh clone before running the designer.
- `registries/manifest.hocon` is tracked listing **only** the networks this repo ships. Any network you add
  locally (`basic/`, `industry/`, your own) must be registered there to be served — but don't commit those
  entries, or a fresh clone will point at files it doesn't have. To stop the local edits showing up in
  `git status`: `git update-index --skip-worktree registries/manifest.hocon`.
