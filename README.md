# Antihydra
#### (C) Team AUSEC

Autonomous racing-based LLM CTF solver: several models attack the same challenge in parallel lanes. The first lane to produce a *verified* flag wins and the rest are stopped. ***No** human is involved*.
## Results

| Challenge | Solved | Wall clock | Evidence |
|:-:|:-:|:-:|:-:|
| CSAW26 qualifiers | 6 / 16  (37.5%) | 48 min | [Logs](logs/evals/20261106-232811__eval/)  ·  [Writeup](SUBMISSION.md) |

## How it works

```
   solve.py        race.py         eval.py
   (1 agent)     (N lanes)       (a corpus)
        └──────────────┬─────────────┘
                       ▼
            ┌────────────────────┐
            │  RaceCoordinator   │
            └──┬──────┬──────┬───┘
               ▼      ▼      ▼
          ┌────────┐┌────────┐┌────────┐      each lane owns
          │ lane 1 ││ lane 2 ││ lane 3 │      a sandbox · findings
          └───┬────┘└───┬────┘└───┬────┘      board · ledger ·
              └────┬────┴────┬────┘           transcript
                   ▼         ▼
      first verified flag    the others abort at the
      ends the race  ────────► next round boundary

      overseer ──── triggers (stall · no remote contact · failed
                    submits) → one curated hint per review
```

- Lanes are independent agent loops; the first verified flag wins and the others abort
  at their next round boundary.
- A hint bus lets lanes share findings with each other, and an **overseer**
  (`glm-5.3`, capped budget) pushes one curated hint into a lane that stalls or
  loses contact.
- Tools run inside the sandbox: `run_command`, `create_file`, `read_file`,
  `submit_flag`, `giveup`, `record_finding`, `get_findings`, `spawn_subtask`, `publish_insight`, `check_hints`.
- Flags are scanned out of tool output and verified host-side against `challenge.json`,
  so a lane cannot self-report success.
- Budget and robustness: price catalog, per-lane cost/round caps, reasoning effort,
  `--max-total-cost`, context compaction, stall detection, full JSON transcripts.

## Quick start

| Platform | Set up |
|---|---|
| Linux, macOS, WSL, Git Bash | `cp .env.example .env`<br>`bash docker/build_sandbox.sh` |
| Windows (PowerShell) | `cp .env.example .env`<br>`docker build -t ctfenv:antihydra docker/sandbox`<br>`docker network create ctfnet` |

Put an `OPENROUTER_API_KEY` in `.env`, then:

```bash
python solve.py --list                            # browse the challenge set
python race.py --challenge arm-strong             # one challenge, N lanes
python eval.py --max-minutes 15 --concurrency 3   # a whole corpus
```

Exit codes: `0` solved · `1` unsolved · `2` usage error · `3` runtime error.

## Requirements

- Python 3.10+ (no dependencies)
- Docker 
- OpenRouter API key


## Layout

| Path | What |
|---|---|
| `antihydra/` | Solver package: challenge, sandbox, tools, agent, racing, hints, overseer, evaluation |
| `solve.py`, `race.py`, `eval.py` | One agent, one challenge with N lanes, a corpus with concurrency |
| `lanes.toml`, `overrides.toml` | Lane roster, loader-side fixes for defective dataset manifests |
| `docker/sandbox/` | Sandbox image: pwntools, Ghidra, SageMath, angr, gdb, qemu-user |
| `scripts/`, `tests/` | Flag-leak audit and race viewer, 54 unit tests |
| `competition/2025/`, [`logs/evals/`](logs/evals/20261106-232811__eval/) | Challenge set (read-only), committed run evidence |

## Dataset policy

`competition/` folder is never edited. Manifest defects are corrected at load time via
`overrides.toml`, and a listed file that cannot be delivered aborts the run instead of
silently handing the agent an empty challenge. Only the `files` list enters a sandbox and
never flags. `scripts/audit_dataset.py` audits this.
