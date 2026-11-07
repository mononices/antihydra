# AGENTS.md

Instructions for agents (and humans) working in this repository.

## Project

Independent, clean-room LLM CTF solver for CSAW/NYU-style challenges.

The code here is not derived from or forked off the NYU CTF Bench
repositories. It is compatible with their data format (`challenge.json`,
`box`/`internal_port` fields, `docker-compose.yml` challenge services on the
`ctfnet` docker network) so the same challenge corpora work.

## Architecture

solve.py (host, stdlib only)
├── challenge.py ......... challenge.json metadata + compose service (ctfnet)
├── sandbox.py ........... sandbox container (ctfenv:antihydra, linux/amd64)
├── llm.py ............... OpenRouter chat/tool calls
└── Agent (main) ......... conversation loop
    ├── tools.py ......... docker exec into the sandbox
    ├── blackboard ....... notes.md inside the sandbox
    ├── CostLedger ....... budget shared by main + workers
    └── spawn_subtask .... worker Agent (fresh conversation, same
                           sandbox/board/ledger, ends with finish_task)

### Runtime model

- One `Agent` instance is one conversation loop with its own message list.
  The main agent and every worker are the same class; workers differ by tool
  profile, prompts, and caps.
- Shared per run: sandbox container, findings board, cost ledger, challenge.
  Not shared: conversation, round counter, stall tracker, compaction state.
- Model calls happen on the host through `llm.py`. The sandbox never sees an
  API key and only receives `docker exec` calls.
- Challenge services and sandboxes join `ctfnet`. Sandboxes have no docker
  socket, no host mounts, and run as non-root `ctfplayer` with sudo only
  inside the container.
- The flag lives host-side in the challenge metadata and inside the challenge
  service container; it never enters the sandbox.

### Agent round

```
round start
  ├ check caps (rounds, cost, stalled, solved, gave_up, finished)
  ├ compact history if the context is filling
  ├ llm.chat(messages, tool schemas)
  ├ run tool calls in the sandbox, append observations
  ├ scan every output for the flag (submit_flag verifies host-side)
  └ stall check on tool-call fingerprints
```

### Context and state

- Observation trimming keeps only the last few tool outputs at full length.
- Compaction triggers at ~70% of the model context (100k estimated tokens when
  the size is unknown), cuts only at assistant-message boundaries, keeps the
  last 12 messages, and replaces the dropped range with an
  `[Earlier context summary]` message. Dropped messages and summaries are
  archived in the transcript.
- Stall detection fingerprints identical tool calls (whitespace-normalized):
  the same call 3× gets a nudge, the second event a harder nudge, the third
  ends the run with reason `stalled`.
- Transcripts land in `logs/<timestamp>__<category>_<challenge>_<agent>_<model>.json`
  with `messages`, `events`, `notes`, `summaries`, and `compacted_messages`.

### Future extension points

- Retrieval/RAG: hint injection at the `spawn_subtask` boundary, using the
  findings board as the durable channel.
- Racing: `race.py` runs `Lane` objects in-process via `RaceCoordinator` with a
  shared hint bus and an overseer; the retrieval (RAG) service remains future
  work.
- Category playbooks: static prompt modules per challenge category.

## Working conventions

- Clean-room: do not copy code or prompt text from the reference repositories
  (NYU CTF Bench / nyuctf_agents, CRAKEN / ctfrag, verialabs/ctf-agent). They
  are design references; compatibility is data-level only.
- Style: no banner or separator comments; short docstrings; comments only where
  behavior is non-obvious.
- Secrets: never read, print, or commit `.env` values — refer to key names only.
- Docker and model calls are manual: hand over the command instead of running
  container builds, `solve.py` runs, or OpenRouter requests. Local edits,
  compile checks, and the fake-sandbox tests in `tests/` are fine to run.
- Flags: only the `files` list from `challenge.json` is copied into the sandbox;
  never copy flag files. `scripts/audit_dataset.py` verifies this.
- `competition/` is read-only: never edit, move, or delete dataset files;
  corrections belong in `overrides.toml`, applied at load time.
- Lanes and workers solve from the challenge description and files alone.
  Knowledge hints (RAG/overseer, when built) must stay technique-level and must
  never contain flags or direct solutions.

## Layout

```
antihydra/                  # solver package (host side uses the stdlib only)
  challenge.py           # discovery, metadata, compose service lifecycle
  sandbox.py             # docker sandbox lifecycle (docker CLI)
  tools.py               # tool registry, cost ledger, main/worker tool profiles
  blackboard.py          # shared markdown findings board (notes.md in sandbox)
  llm.py                 # OpenRouter chat client + model price catalog
  runtime.py             # shared model clients + catalog cache
  compaction.py          # summarize old history when the context fills
  stalls.py              # repeated-call stall detection
  agent.py               # agent loop: main agent + fresh-context subtask workers
  racing.py              # Lane / RaceCoordinator classes + roster parsing
  hints.py               # in-process hint bus between racing lanes
  overseer.py            # trigger-driven supervisor that pushes curated hints
  evaluation.py          # eval harness: run a challenge set, aggregate results
  prompts.py             # system/initial/worker prompts
  transcript.py          # JSON run transcripts
solve.py                 # CLI entry point for one solve
race.py                  # CLI entry point for parallel racing
eval.py                  # CLI entry point for the evaluation harness
lanes.toml               # modifiable lane roster used by race.py
docker/sandbox/          # sandbox image (Dockerfile, tools.txt, Ghidra script)
docker/build_sandbox.sh  # build image + create the ctfnet network
scripts/audit_dataset.py # flag-leak audit for challenge data
scripts/show_race.py     # race log viewer
tests/test_flow.py       # fake-sandbox unit tests (no docker, no network)
competition/2025/        # CSAW 2025 challenge set (crypto, misc, pwn, rev, web)
overrides.toml           # loader-side corrections for dataset defects
```

## Requirements

- Python 3.10+ (no third-party packages needed on the host)
- Docker (Docker Desktop on Windows/macOS, or Docker Engine on Linux)
- An OpenRouter API key

## Quick start

```bash
# 1. Configure credentials
cp .env.example .env
#    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_COMPACT_MODEL
#    Windows PowerShell: Copy-Item .env.example .env

# 2. Build the sandbox image (takes a while: Ghidra, SageMath, angr, ...)
bash docker/build_sandbox.sh
#    lightweight build without cado-nfs:
#    EXTRA_BUILD_ARGS="--build-arg WITH_HEAVY=0" bash docker/build_sandbox.sh
#    Windows PowerShell (no bash):
#    docker build -t ctfenv:antihydra docker/sandbox
#    docker network create ctfnet    # "already exists" -> harmless

# 3. Explore
python solve.py --list
python solve.py --list-models --grep claude
python solve.py --show --challenge obligatory-rsa

# 4. Smoke-test the sandbox plumbing (no model calls)
python solve.py --dry-run --challenge obligatory-rsa

# 5. Solve
python solve.py --challenge obligatory-rsa --model openrouter/anthropic/claude-sonnet-5.5

# 6. Housekeeping and dataset audit
python solve.py --cleanup
python scripts/audit_dataset.py
```

Exit codes: `0` solved, `1` not solved, `2` usage error, `3` runtime error.

## CLI reference

| Flag | Meaning |
|---|---|
| `--challenge NAME\|DIR` | challenge to solve |
| `--list` / `--show` | list challenges / print details + prompts |
| `--dry-run` | start sandbox + copy files, no model calls |
| `--list-models` | OpenRouter models + prices (`--grep`, `--limit`, `--refresh-catalog`) |
| `--cleanup` | remove orphan sandbox containers |
| `--model` / `--api-key` | defaults from `OPENROUTER_MODEL` / `OPENROUTER_API_KEY` |
| `--image` / `--platform` / `--network` / `--memory` | sandbox settings (defaults: `ctfenv:antihydra`, `linux/amd64`, `ctfnet`, `8g`) |
| `--max-rounds` / `--max-cost` / `--temperature` / `--max-tokens` | model loop limits |
| `--timeout` | default `run_command` timeout (seconds) |
| `--no-service` | never start the challenge `docker-compose` service |
| `--no-subtasks` / `--max-subtasks` | disable / cap fresh-context helper agents |
| `--subtask-rounds` / `--subtask-cost` | per-worker round and USD caps |
| `--no-compaction` / `--compact-model` | disable / change model for context compaction |
| `--no-stall-detection` / `--stall-repeat` | disable / tune repeated-call detection |
| `--log-dir` / `--quiet` | transcript directory / console verbosity |

## How a run works

1. The challenge is loaded from `challenge.json`; server-backed challenges have
   their compose stack started on `ctfnet`.
2. A sandbox container (`ctfenv:antihydra`) starts on `ctfnet`; challenge files are
   copied to `/home/ctfplayer/ctf_files/`.
3. The agent loop calls the model (OpenRouter) with the tool schemas; tool calls
   execute inside the sandbox (`run_command`, `create_file`, `read_file`,
   `submit_flag`, `giveup`).
4. Durable knowledge lives on a shared markdown notes board
   (`/home/ctfplayer/notes.md`) via `record_finding` / `get_findings`; it
   survives context trimming and is greppable with `cat` / `grep`.
5. For context-heavy work the main agent can call `spawn_subtask`: a worker with
   a fresh conversation, the same sandbox and notes board, bounded rounds/cost,
   ending with `finish_task(summary)`. Subtask summaries go to the board and the
   transcript.
6. Outputs are scanned for the flag; `submit_flag` verifies it against the
   challenge metadata. Long runs compact old history into a summary when the
   context window fills, and a stall detector nudges (then stops) agents that
   repeat identical calls. Transcripts (messages + events + notes + summaries +
   cost) are written to `logs/`.

## Racing

Multiple attempts are allowed, so the solver can race several lanes against one
challenge; the first flag wins and the rest are stopped.

```bash
# edit lanes.toml (models, temperature, caps per lane), then:
python race.py --challenge arm-strong --max-minutes 30

# or pass models inline for a quick race
python race.py --challenge arm-strong \
  --lanes openrouter/anthropic/claude-sonnet-5.5 \
          openrouter/openai/gpt-5.6-sol \
  --lane-max-cost 4 --max-minutes 30
```

- Lanes are `Lane` objects inside the coordinator process: each owns its own
  sandbox container, findings board, cost ledger, and transcript. The challenge
  service is started once by the coordinator and shared.
- First lane to report `solved` wins; the others abort at their next round
  boundary (stop event) and their sandbox containers are removed by label.
- `Ctrl+C` aborts all lanes, stops the challenge service, removes their
  containers, and still writes `race-summary.json` (marked `interrupted`).
- Lanes share an in-process hint bus (`publish_insight` / `check_hints`, no LLM
  relay yet); subtask summaries are published automatically and the summary
  records hint counts.
- An overseer watches lane state on a timer and on triggers (stall, no remote
  contact, failed submits, bus starvation), then pushes one curated hint into a
  lane's inbox after optionally requesting a targeted trace slice. Default model
  `z-ai/glm-5.3-flash` at `max` reasoning, with its own call and cost caps.
- Lane output streams live with `[lane-N model]` prefixes; `--no-follow`
  silences the stream. Full output is always in `lane-N/run.log`.
- Artifacts: `logs/races/<ts>__<challenge>/lane-N/` (`run.log`, `lane.json`,
  transcript) plus `race-summary.json` (per-lane status, cost, rounds,
  `pass_at_k`, `solved_by`, `flag`) and `flag.txt` when a lane finds the flag.
- Rosters: `lanes.toml` (shipped template; used automatically when no
  `--lanes`/`--lanes-file` is given) or a JSON list. Precedence per field:
  lane entry > CLI flag > `[defaults]` > built-in fallback. Fields: `model`,
  `temperature`, `reasoning_effort` (default `max`), `max_cost`, `max_rounds`,
  `no_subtasks`.
- Aborted lanes report their spend from their own ledger (they stop at a round
  boundary); the coordinator force-stops only stragglers.
- Race sandboxes use per-lane labels (`antihydra.lane.<race>.N`); `solve.py
  --cleanup` only removes the default label, the coordinator cleans its own.

## Evaluation

Run the racing solver across a challenge set and aggregate the results:

```bash
# whole local set, resumable, 30 min cap per challenge
python eval.py --max-minutes 30 --max-total-cost 25

# one category with an inline roster
python eval.py --category pwn --lanes openrouter/anthropic/claude-sonnet-5.5 --max-minutes 45
```

- Per challenge: `<eval-dir>/<NN>__<challenge>/` holds the race artifacts
  (`race-summary.json`, `flag.txt` when solved, `lane-N/` logs and transcripts).
- Aggregates in the eval root: `eval-summary.json` (per-challenge results,
  per-category breakdown, cost, `pass_at_k` where k = lanes, `lane_solve_rate`),
  `summary.json` (leaderboard shape: metadata + name to bool), and
  `traces.jsonl` (one line per lane transcript).
- `--concurrency N` (default 3) runs N challenges at once; challenges whose
  compose publishes the same host port (or uses a fixed `container_name`) are
  serialized automatically. Sandbox memory scales with concurrency x lanes.
- `--skip-existing` reuses completed race summaries; `--max-total-cost` stops the
  eval between challenges.

## Status / roadmap

- [x] challenge discovery, metadata, compose services
- [x] docker sandbox + basic tool set
- [x] OpenRouter client with price catalog, retries, and cost tracking
- [x] single-agent solve loop + JSON transcripts
- [x] agent-as-tool subtasks (`spawn_subtask` → fresh-context workers)
- [x] shared markdown findings board (`record_finding` / `get_findings`)
- [x] context compaction (old history summarized when the context fills)
- [x] stall detection (repeated identical calls are nudged, then abort the run)
- [ ] retrieval service (CTF writeup RAG, hints injected at subtask boundary)
- [x] racing coordinator (`race.py`: in-process lanes, first flag wins,
      pass@k summary, `flag.txt` artifact)
- [x] evaluation harness (`eval.py`: per-challenge races, `eval-summary.json`,
      leaderboard-shaped `summary.json`, `traces.jsonl`, `--skip-existing`)
- [x] cross-lane hint bus (`publish_insight` / `check_hints`, in-process; no LLM
      relay yet)
- [x] per-lane reasoning effort (`--reasoning-effort`, `lanes.toml` field,
      default `max`; compaction override still pending)
- [x] overseer (trigger reviews, inbox push, one-step trace inspection; RAG
      hint injection still pending)
- [x] leaderboard packaging (leaderboard-shaped `summary.json` + `traces.jsonl`
      from `eval.py`; NYU canonical challenge names still manual)

## Notes

- The sandbox image is amd64 so x86/x86-64 challenge binaries run locally;
  ARM binaries run via prebaked `qemu-aarch64-static`, with `gdb-multiarch` and
  `aarch64-linux-gnu-*` binutils available.
- Findings live in `/home/ctfplayer/notes.md` inside the sandbox; racing lanes
  will each get their own board file once the coordinator lands.
- `docker/sandbox/tools.txt` ships inside the image at `/tools.txt` as an
  agent-facing tool cheat sheet.
- Keep `.env` out of git; it is already in `.gitignore`.
- The sandbox currently has internet egress; use an `--internal` `ctfnet` if a
  competition requires offline agents.
