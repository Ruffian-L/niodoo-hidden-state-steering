# Niodoo: correcting a frozen model's last-step errors by steering its hidden state toward exported attractor basins

A short report of one narrow, reproducible result, the runtime behind it, and what is still unfinished.

Status: not polished. Actively being worked on. The result below reproduces end to end; much of the surrounding
system does not yet. This document states what is, not what we hope.

---

## Abstract

Niodoo is a small local runtime that runs alongside a frozen language model and steers it. It is not a model and
it does not retrain weights. This report covers one thing it does, measured directly: on a frozen
Llama-3.1-8B-Instruct (Q5_K_M), with the bridge off the model locks a wrong final answer on a class of prompts even
when its own intermediate steps were correct; with the bridge on it lands the correct answer. On an eight-prompt
battery run at temperature 0, seed 42, the bridge corrected four wrong answers, left three correct answers
unchanged, and broke one correct answer. The correction generalizes — it does not replay a memorized answer (a
prompt whose correct count is 2 stays 2; it is not forced to 3). The mechanism that produced this is a small,
hard-capped, per-token nudge of the pre-output hidden state toward the nearest of eight 64-dimensional attractor
basins. Everything needed to reproduce it — binary hash, model hash, commands — is included.

This is a narrow claim. It does not show broad benchmark superiority, it does not show the result holds across
seeds, and the bridge-off arm is not yet a true vanilla baseline. Those controls are named and left open.

---

## 1. What this is, and what it is not

Niodoo is a runtime, not a model. Its premise is continuity, not raw intelligence: a small local process that sits
next to a frozen model, watches it generate, and nudges it. It does not change weights.

This report does not describe the whole system. The whole system is large, much of it is unverified, and parts of
it are explicitly placeholder code. This report describes a single behavior that was run, measured, and can be run
again by someone else.

Two failure modes are avoided on purpose. The first is overclaiming — calling a mechanism that loads, or a force
that fires, a behavior win. The second is the opposite — burying a real, reproduced result under caution because
adjacent claims are not yet proven. What follows is held to: stated, measured, reproducible.

---

## 2. The result

Frozen model: Llama-3.1-8B-Instruct, Q5_K_M quantization (bartowski), sha256
`14e10feba0c82a55da198dcd69d137206ad22d116a809926d27fa5f2398c69c7`.
Decoding: greedy, temperature 0, seed 42, 256 max steps. Deterministic. The only variable is the bridge.

| Prompt | Correct | Bridge off | Bridge on | |
|---|---|---|---|---|
| Bat and ball ($1.10, bat $1 more) | $0.05 | $0.05 | $0.05 | unchanged |
| Count the r's in "strawberry" | 3 | locks 2 | 3 | corrected |
| 17 × 24 | 408 | locks 368 | 340+68=408 | corrected |
| 13 × 17 | 221 | locks 321 | 221 | corrected |
| 23 × 18 | 414 | does not land | 414 | corrected |
| Count the r's in "raspberry" | 2 | 2 | 2 | unchanged (not forced to 3) |
| Count the a's in "banana" | 3 | 3 | 3 | unchanged |
| Count the s's in "mississippi" | 4 | 4 | 6 | broken |

Four corrected, three left correct, one broken. The bridge-off model is fluent and shows its working in each case;
the error is a last-step slip — it computes 340 and 68 and then locks 368. The bridge-on model catches that slip.

The "raspberry" row is the control that matters most: if the bridge were injecting a memorized "3", raspberry (two
r's) would come out 3. It comes out 2. The correction is not replay.

The "mississippi" row is the honest cost: the bridge turned a correct answer (4) into a wrong one (6). It is not free.

---

## 3. How it works (the verified path)

The bridge loads eight ghost basins from a registry exported by the niodv4 pipeline
(`niodoo/src/bridge/ghost_basin.rs`, `niodoo/src/bridge/registry.rs`). Each basin is a 64-dimensional centroid —
the mean of the last ten states of a stabilized rollout — with persistence and readiness scores
(`ghost_basin.rs:18–37`). These are attractor points in the model's representation space, not tokens and not text.

At each decoding step, with the bridge on, the runtime computes the current pre-output hidden state, finds the
nearest basin by distance, and applies a small additive pull toward it before the final projection to logits. The
per-token telemetry records this directly: `nearest_ghost_distance`, `ghost_pull_delta_norm`, `intervention_applied`
(`niodoo/src/main_helpers2.rs:1694–1850`). In the selective mode, the pull is gated on the route margin — it only
fires when the model is near a decision boundary (`--bridge-influence-selective`, `niodoo/src/cli.rs:255–261`).

The pull magnitude is small and hard-capped. The influence clamp is bounded to 0.03 in source
(`niodoo/src/simulation.rs:577–579`, `.clamp(0.0, 0.03)`). This was first observed from the outside — the
`--bridge-influence-smoke-clamp` flag has no effect above 0.03, and the corrected output is byte-identical at clamp
0.03, 0.5, and 2.0 — and then confirmed in the code. The correction is therefore not a function of force magnitude.
It is the consequence of a tiny nudge applied near a token decision boundary: where the model is already teetering,
0.03 is enough to tip it. Where it is confident (bat and ball), 0.03 changes nothing.

The intuition the data supports: the basins act as a weak prior over "where good final states sit." Near a boundary,
that prior tips the choice. When the nearest basin aligns with the correct answer the model is corrected; when it
does not, the same nudge can push a correct answer off (mississippi).

---

## 4. What is a stub, and what did not fire

This matters for honesty and is stated plainly.

- The 64D→3D projection used by one steering path is an explicit placeholder that takes the first three dimensions
  (`niodoo/src/bridge/projection.rs:1–28, 95–101`). The code says so.
- The logit-bias steering engine is a simplified inverse-distance stub (`niodoo/src/physics/steering.rs:32–72`).
- In the witnessed correction run the recovery-specialist and motif-hinge machinery did not fire:
  `MOTIF_HINGE flipped=false`, `recovered=false`, routing ticks 0 — identical between off and on. A specialist is
  loaded but inactive. The win came from the bare hidden-state nudge, not from the higher-level recovery system that
  the surrounding documentation describes. The smaller claim is the true one.

---

## 5. Limitations and open controls

- The pull is unaimed. It targets the nearest basin, not the correct token. Until it carries a payload bound to the
  answer, over-corrections like mississippi are expected, not surprising.
- Bridge-off is not a true vanilla baseline. It is Niodoo with the bridge disabled. A raw llama.cpp run on the same
  prompts is the control needed before any "beats vanilla" statement.
- One seed, one temperature. The result is deterministic at seed 42, temperature 0; robustness across seeds is not
  yet measured.
- The binary is platform-specific (CUDA, `sm_121`). It does not run unmodified on arbitrary hardware. Build-from-source
  and portability are open work.

---

## 6. Reproduction

The principle is: trust the bytes, not the names. Model repositories drift; identifying a model by name is not
reproducible. Every artifact here is identified by sha256.

```
binary  niodoo/target/release/niodoo   sha256 8c7778276517bcfc684f7bb008ca859759676e833e3bef54344689806eba95d8  (bridge feature present)
model   Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf  sha256 14e10feba0c82a55da198dcd69d137206ad22d116a809926d27fa5f2398c69c7  (bartowski)
run     seed 42, temperature 0, 256 steps; OFF = --bridge-off ; ON = --bridge-influence-smoke --bridge-influence-smoke-clamp 0.03
```

`reproduce.sh` verifies the binary carries the bridge feature, downloads the model only if missing, refuses to run
if the model's sha256 does not match, then runs the off and on arms and prints the answers next to the correct one.
A clone that disagrees on a hash is telling you the run is not the published run. See `RUNBOOK.md` for the full
procedure and the directory gotcha (the basins load by relative path; the working directory must be the tree).

---

## 7. The claims ledger

This result is one row in a large ledger. The project has on the order of a thousand recorded claim folders, many
marked passing. Their reproducibility varies, and not all have been re-run end to end. This correction is the first
one packaged for a stranger to reproduce from hashes. The rest are being surfaced in this form over time, by the
human, deliberately, one at a time.

A line from the project's own notes, kept because it is accurate: **`CLAIMS.md` is not a muzzle. It is a scoreboard.**
The ledger is not a list of things forbidden to claim. It is a record of what has and has not been earned. Most of it
was earned without institutional backing.

---

## 8. How it came to be

This was not built by a model, and it was not built alone. The direction, the architecture, and the stubbornness to
keep going for a year are the author's (Jason). The approach — steering a frozen model through its hidden state —
predates the current research interest in whether a model can lose its thread while staying fluent; the author dates
the first working version to late 2025, developed over the preceding year alongside Grok and Gemini.

The rigor was collaborative and is attributed where known. The regression forensic that mapped why the project kept
appearing to break was produced by GPT (2026-06-24). The correction run, the generalization control, the provenance
resolution, and this report were produced by Claude in one session (2026-06-24), against a runtime and a claims
ledger the author had been building and grading for months. Who ran which experiment is recorded in the reproduction
artifacts and is intended to stay recorded — the layer of who did what is part of the work, not a detail to flatten
under one name.

---

## 9. Status

Active. Unpolished on purpose rather than by neglect — the alternative was to keep it in a drawer until it looked
finished, and it would never have looked finished. The result in section 2 is real and runs. The system around it is
in progress. Corrections, controls, and the rest of the ledger follow.
