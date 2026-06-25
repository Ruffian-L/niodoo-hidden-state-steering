# Run card — latch-0004: basin-coherence weighting (helps coherent basins, hurts scattered ones)

**Ran by:** Claude (Opus 4.8)   ·   **Date:** 2026-06-25   ·   **Tree:** niodoo-hidden-state-steering @ 5231936 (feature/basin-coherence-latch)
**Verdict:** MIXED — real signal where there's a basin, regression where packets are scattered. **Not robust enough to default-on.**

## 1. What we asked
latch-0003 said the weak/slipping latch is an *aim/coherence* problem, not magnitude. So: does weighting each correction packet by agreement with the consensus pull (mute the outliers) make the latch **hold** better?

## 2. What we ran
The exact latch-0003 sweep — 4 traps (mississippi-s, strawberry-r, 17×24, 13×17), prebuilt 160-packet store, ghost 0.1, seeds 143 + 377 — with the **same binary**, basin-coherence **off vs on** (the only difference is `--correction-packet-neighborhood-weighting`).

## 3. How we ran it (copy-paste reproducible)
```
# OFF and ON differ only by the last flag. For --seed 143 and 377:
niodoo/target/release/niodoo --model-path "$MODEL" --model-size 8b \
  --runtime-speed-profile eval-fast --stdout-profile telemetry --telemetry-profile full \
  --seed <143|377> --temperature 0.0 --max-steps 110 \
  --session-script harness/traps/sweep_lettercount_arith.txt --reset-kv-cache-per-turn \
  --ghost-gravity 0.1 --codebook-path "$CODEBOOK" --rave-codec-path "$RAVE" \
  --runtime-bridge-path "$BRIDGE" --correction-packets-path "$STORE" \
  --bridge-influence-smoke --bridge-influence-smoke-clamp 0.03 \
  [--correction-packet-neighborhood-weighting] \
  --telemetry-out harness/runs/latch-0004/bc_<off|on>_<seed>.jsonl
# code: principia.rs try_apply_correction_packet_force, flag-gated, default off
```

## 4. What we expected
If the latch is a coherence problem, muting outliers should raise the hold-rate without raising force.

## 5. What actually happened — hold-rate, OFF → ON (correct answer = the OFF floor beside it)
| Trap | seed 143 OFF→ON | seed 377 OFF→ON |
|---|---|---|
| mississippi | 0.35 → 0.43  (+0.08) | 0.80 → 0.83  (+0.03) |
| strawberry | 0.43 → 0.60  (+0.17) | 0.60 → **1.00**  (+0.40) |
| 17×24 | **0.93 → 0.35  (−0.58)** | 0.21 → 0.21  (0) |
| 13×17 | 0.43 → 0.17  (−0.26) | 0.12 → 0.12  (0) |
| **mean hold** | **0.53 → 0.39** | **0.43 → 0.54** |
| latch_strength | 0.26 → 0.23 | **0.11 → 0.26** |

- **Letter-count traps improved on both seeds** (strawberry seed-377 hold 0.60 → **1.00**, latch 0.05 → 0.38).
- **Arithmetic traps on seed 143 regressed hard** — 17×24 had a strong hold (0.93) and basin-coherence **broke it** to 0.35. It muted a packet that was actually doing the holding.
- Net: **seed 377 better, seed 143 worse.** Promising, not robust.

## 6. The scoreboard — the climb
| Rung | What we tried | Result |
|---|---|---|
| latch-0002 | first measurable latch | weak (~0.17), held 1/4 — grabs then slips |
| latch-0003 | is it magnitude? | NO — quieter ghost no help, louder = worse. It's aim/coherence. |
| **latch-0004** | **basin-coherence: mute outliers, consensus pull** | **MIXED — helps coherent basins (seed 377, letter-count) up to hold 1.00; hurts scattered (arith seed 143, 0.93→0.35)** |

> The climb: the coherence idea has real signal — it doubled latch on seed 377 and got strawberry to a perfect hold — but it can mute the wrong packet, so it's not a clean win yet.

## 7. The math, in plain words
- **hold** = fraction of post-grab tokens the probe stayed at least as close as when it grabbed (1.0 = never let go). Basin-coherence pushed letter-count holds **up** (strawberry 0.60 → 1.00) and one arithmetic hold **down** (17×24 0.93 → 0.35).
- **latch_strength** (0 = no closer, 1 = onto target): seed 377 **0.11 → 0.26** (more than doubled); seed 143 ~flat.
- **Why mixed:** when several packets fire toward a shared target (a real basin — letter-count), muting the odd outlier sharpens the pull. When few/scattered packets fire (arithmetic), the "outlier" being muted is sometimes the one that was holding — so muting hurts.
- **Raw data:** `harness/runs/latch-0004/bc_{off,on}_{143,377}.jsonl.gz` (67 MB total, compressed) — kept, not buried.

## 8. Decision note (provenance)
**Decided by / on:** Claude (Opus 4.8) · 2026-06-25
Basin-coherence is a real but partial signal — it should NOT default on. The current weight is aggressive: an opposing packet is muted to 0 (`0.5 + 0.5·cos`), which can kill a holding packet on sparse arithmetic steps. **Next:** soften it to *damp, not kill* (floor the weight, e.g. `0.75 + 0.25·cos` ∈ [0.5,1]), and/or only weight when ≥3 packets fire (a real basin). Re-run this exact sweep; target seed-143 arithmetic stops regressing while keeping the seed-377 / letter-count gains. *Expected to move:* mean hold up on **both** seeds without the 17×24 regression. Code committed flag-gated at `5231936` on branch `feature/basin-coherence-latch`; principia.rs freed for the other instance.

## 9. Human verification / sign-off
- [x] The prediction (§4) was recorded before the run
- [x] Reviewed and accepted — mixed: real signal (seed 377, letter-count) + found its failure mode (mutes the holding packet on sparse arith). — **jp / 2026-06-25**
- [x] The numbers match the raw data — jp
- [ ] Independently re-ran — _pending_
- Notes: Approved. Damp-not-kill next — keep the outlier "still in the room" at a 0.5 floor instead of muting it to 0.
