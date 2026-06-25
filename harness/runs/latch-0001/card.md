# Run card — latch-0001: can we even measure a latch today?

**Ran by:** Claude (Opus 4.8)   ·   **Date:** 2026-06-24   ·   **Tree:** niodoo-hidden-state-steering @ 10232f3 (feature/correction-to-trajectory-fixes)
**Verdict:** MIXED — the answer was correct, but **no latch was measurable**. Baseline floor established.

## 1. What we asked
With the runtime as it runs today (bridge-on smoke config), can we measure a correction *latching* onto a target — and what is the floor?

## 2. What we ran
The "strawberry" letter-count trap, bridge **ON** (`--bridge-influence-smoke`, clamp 0.03), seed 143, 96 steps, **full** per-token telemetry.

## 3. How we ran it (copy-paste reproducible)
```
NIODOO_MODEL_PATH=/home/ruffianl/projects/team_build/model/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf   # sha256 14e10feba0…c69c7
niodoo/target/release/niodoo --model-path "$NIODOO_MODEL_PATH" --model-size 8b \
  --runtime-speed-profile eval-fast --stdout-profile telemetry --telemetry-profile full \
  --seed 143 --temperature 0.0 --max-steps 96 \
  --prompt "How many times does the letter r appear in the word strawberry? Give the final number only." \
  --telemetry-out harness/runs/latch-0001/strawberry_on_seed143.jsonl \
  --bridge-influence-smoke --bridge-influence-smoke-clamp 0.03
# binary sha256 ee79e5c29799b395…  · GPU NVIDIA GB10 · run from repo root
```

## 4. What we expected
A latch signal — the probe pulled toward a target and *held* — that we could baseline now and make "louder" later.

## 5. What actually happened
- The model answered **3**. Correct answer: **3** (strawberry has three r's). ✓ — *answer matches.*
- But **no latch mechanism engaged.** Correction packets fired on **0 of 97** tokens; gate34 never latched (**0 of 97**). There was nothing to measure.
- The force was almost entirely the ghost pull: `applied_ghost_force = 10.0` out of `total_force ≈ 10.06`, every token.

## 6. The scoreboard — the climb
No prior attempts — **this is rung one.**

Frame each rung as *what-we-tried → result*, never as fault — a failure is a rung, not blame.

## 7. The math, in plain words
- **Ghost share of total force: median 99.8%** (range 98.8–100%). Plain: the ghost pull — a fixed strength of 10 — is essentially the *only* force acting. Gravity (0.05), motif (0.013), and everything else are rounding error beside it.
- **Correction-packet fires: 0/97 tokens. gate34 latched: 0/97 tokens.** Plain: the two mechanisms that could "latch" never switched on in this config.

**Raw data:** `harness/runs/latch-0001/strawberry_on_seed143.jsonl` (8.4 MB) + `…_seed143.txt` — kept, not buried. The numbers above are the lens, not the receipt.

## 8. Decision note (provenance)
**Decided by / on:** Claude (Opus 4.8) · 2026-06-24
This is the RC1 confound, quantified: you cannot measure a correction's own effect while a constant ghost-force of 10 is **99.8%** of the push. Next: (a) find and turn **down** the ghost gain so the correction isn't drowned, and (b) enable the correction-packet path (codebook + rave codec + store + live-mint) so a latch can actually form — then re-run this exact trap and measure latch strength + hold. *Expected to move:* ghost share well below 100%, and a non-zero correction-packet fire count, so a latch reading becomes possible.

## 9. Human verification / sign-off
- [x] The prediction (§4) was recorded before the run
- [x] Reviewed and accepted the finding (ghost force 99.8% of total → latch not yet measurable) — **jp / 2026-06-24**
- [x] The numbers match the raw data — jp
- [ ] Independently re-ran the command — _pending_
- Notes: Accepted as rung one. Proceed to rung two — turn the ghost gain down + enable the correction-packet path so a latch can be measured.
