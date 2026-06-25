# Run card — latch-0003: is the weak latch a magnitude problem? (No — both ways)

**Ran by:** Claude (Opus 4.8)   ·   **Date:** 2026-06-24   ·   **Tree:** niodoo-hidden-state-steering @ 10232f3 (feature/correction-to-trajectory-fixes)
**Verdict:** PASS — clean 2-seed floor established, and **magnitude is ruled out as the lever.**

## 1. What we asked
The latch is weak and slips (latch-0002). Is that a *magnitude* problem — is the ghost still drowning it, or is the correction just not strong enough? Or is it something else (aim / coherence)?

## 2. What we ran
Same 4-trap sweep (mississippi-s, strawberry-r, 17×24, 13×17), prebuilt 160-packet store, **seeds 143 and 377**, full telemetry. Two experiments:
- **A — isolate the ghost:** `--ghost-gravity 0.1` (near-off), default correction clamp.
- **B — louder correction:** `--ghost-gravity 0.1` + `--correction-packet-clamp 5.0` (default 0.03 → correction force ~80× louder).

## 3. How we ran it (copy-paste reproducible)
```
# A (isolate ghost): same as latch-0002 but --ghost-gravity 0.1, for --seed 143 and 377
# B (louder):        add --correction-packet-clamp 5.0
niodoo/target/release/niodoo --model-path "$MODEL" --model-size 8b \
  --runtime-speed-profile eval-fast --stdout-profile telemetry --telemetry-profile full \
  --seed <143|377> --temperature 0.0 --max-steps 110 \
  --session-script harness/traps/sweep_lettercount_arith.txt --reset-kv-cache-per-turn \
  --ghost-gravity 0.1 [--correction-packet-clamp 5.0] \
  --codebook-path "$CODEBOOK" --rave-codec-path "$RAVE" --runtime-bridge-path "$BRIDGE" \
  --correction-packets-path "$STORE" --bridge-influence-smoke --bridge-influence-smoke-clamp 0.03 \
  --telemetry-out harness/runs/latch-0003/<file>.jsonl
# assets/model hashes as in latch-0002 · GPU NVIDIA GB10 · run from repo root
```

## 4. What we expected
If it's a magnitude problem, isolating the ghost (A) or pushing the correction harder (B) should make the latch hold better.

## 5. What actually happened
- **A (isolate ghost):** latch still weak — seed 143: latch 0.26 / hold 0.53 / held **1 of 4**; seed 377: latch 0.11 / hold 0.43 / held **2 of 4**. Dropping ghost 10→1→0.1 barely moved it.
- **B (louder, force ~139 vs ~1.74):** **WORSE** — seed 143: latch 0.08 / hold 0.08 / held **0 of 4**; seed 377: latch 0.13 / hold 0.34 / held **1 of 4**. The 80× stronger pull *overshot*.
- Note: the "ghost share of force" reading stayed high even at ghost 0.1 — that metric compares physics-space ghost force to 4096D correction force (apples-to-oranges), so we trust the **distance trajectory**, not the share.

## 6. The scoreboard — the climb
| Rung | What we tried | Result |
|---|---|---|
| latch-0001 | measure a latch in smoke config | 0 fires; ghost 99.8% of force — nothing to measure |
| latch-0002 | load 160-packet store, ghost 1.0 | first measurable latch: ~0.17, held 1/4 — grabs then slips |
| latch-0003 A | isolate ghost (0.1), 2 seeds | still weak: latch 0.11–0.26, held 1–2/4 — **ghost is not the lever** |
| **latch-0003 B** | **louder correction (force ~139)** | **WORSE: held 0–1/4 — overshoot. Magnitude is not the lever.** |

> The climb: we chased magnitude in both directions — quieter ghost, louder correction — and neither made the latch hold. That's the finding.

## 7. The math, in plain words
- **latch_strength** (0 = no closer, 1 = onto target) stayed **0.08–0.26** across every config. Plain: the pull never gets the probe much closer to the target.
- **hold** (fraction of post-grab tokens that stayed at least as close as the grab): isolating ghost ≈ 0.43–0.53; louder ≈ 0.08–0.34. Plain: louder didn't help it hold — it **hurt**.
- **force_norm: 1.74 → 139** when we raised the clamp, yet **held dropped from 1–2/4 to 0–1/4.** A harder pull overshoots into a different basin (consistent with the known force band: above ~0.5 over-pulls into a different wrong answer).
- **Raw data:** `harness/runs/latch-0003/latch{3,4}_{143,377}.jsonl.gz` (~68 MB total, compressed) — kept, not buried.

## 8. Decision note (provenance)
**Decided by / on:** Claude (Opus 4.8) · 2026-06-24
We have now ruled out magnitude in **both** directions: a quieter ghost didn't help, a louder correction hurt. So the weak, slipping latch is a problem of **aim / coherence**, not force — the pull isn't *consistent in direction*, so more of it just overshoots. This is the empirical green light for the **basin-coherence** change: weight each correction packet by agreement with the consensus pull (mute the outliers), and let the budget grow only when many neighbors agree — coherence, not magnitude. *Expected to move:* hold-rate up from ~1–2/4 without raising force, re-running this exact sweep. **Next edit touches `principia.rs` — coordinate with the other instance's six-fix files first.**

## 9. Human verification / sign-off
- [x] The prediction (§4) was recorded before the run
- [x] Reviewed and accepted — magnitude ruled out both ways (quieter ghost no help; louder = worse/overshoot). It's aim/coherence. — **jp / 2026-06-25**
- [x] The numbers match the raw data — jp
- [ ] Independently re-ran — _pending_
- Notes: "Squeezed the knobs dry — good negative signal." Proceed to the basin-coherence code change once the other instance is clear of principia.rs.
