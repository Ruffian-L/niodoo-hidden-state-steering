# CLAUDE.md

Orientation for any AI working in this repo. **Read `STANDARDS.md` — it is the law for how we record evidence.**

## What this repo is
`niodoo-hidden-state-steering` — the **public build surface** and source of truth for runnable **Niodoo** evals. The bridge-correction claim, its evidence, and the reproduce harness live here: `claim_card.md` (the worked example of a claim card), `RUNBOOK.md`, `WHITEPAPER.md`, `evidence/`, `harness/`, `reproduce.sh`.

## The standard (non-negotiable — full text in `STANDARDS.md`)
- Evidence is **plain-text, back-and-forth, human-readable**. Never buried in logs or a 6 GB JSONL.
- Every claim gets a **run card** (`run_card_template.md`): who & when, what we asked, what we ran, how, expected vs actual, the climb scoreboard, the math in plain words, decision note, sign-off.
- The **scoreboard always shows the recent attempt history** so a win feels earned, not magic. It's a scoreboard, not a muzzle — progress, not blame.
- **Provenance is decisions, not just dates** — record what we turned and *why*, for the day the runtime reads its own trail.
- When the work is irreducibly math, **explain it in plain words next to the real numbers**. Surface it; never hide it.
- `claim_card.md` already lives this: the correct answer is printed next to every result, with a "trust the bytes" provenance table and an honest boundary. Keep doing that.

## Build / run
See `RUNBOOK.md`. Build: `cd niodoo && cargo build --release --bin niodoo --features niodv4_bridge`, then `./reproduce.sh` from the repo root (the basin registry loads from a hardcoded relative path — run from anywhere else and you get the "enabled but empty" false regression).

## Cross-repo map
- **team_build** — private working tree; lib builds, binary broken (`BridgeForceTrajectorySchedule` / `TokenPhysics`).
- **niodoo-hidden-state-steering** (here) — public build surface, source of truth for runnable evals.
- **niodv4** — Python research lineage; source of the ported assets (RAVE codec, codebooks, basins, bridge JSON).
