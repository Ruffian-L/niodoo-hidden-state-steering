# Run card — latch-0005: damp-not-kill (it helped 377, hurt 143; the latch metric is noisy)

**Ran by:** Claude (Opus 4.8)   ·   **Date:** 2026-06-25   ·   **Tree:** niodoo-hidden-state-steering @ 338e414 (feature/basin-coherence-latch)
**Verdict:** NEGATIVE/INFORMATIVE — softening the outlier (floor 0.5) did **not** fix seed 143; it made it worse, while helping seed 377. Basin-coherence is **not a robust win**, and single-trap latch readings are **noisy**.

## 1. What we asked
latch-0004's basin-coherence muted opposing packets to 0, which silenced a packet that was holding (17×24 seed 143, 0.93 → 0.35). Fix: *damp, don't kill* — floor the outlier at 0.5 ("still in the room, helps a little"). Does that stop the regression while keeping the seed-377 gains?

## 2. What we ran
The exact sweep again (4 traps, prebuilt store, ghost 0.1, seeds 143 + 377), same binary, basin-coherence off vs on — with the weight mapping changed from `0.5 + 0.5·cos` ∈ [0,1] to `0.75 + 0.25·cos` ∈ [0.5,1].

## 3. How we ran it (copy-paste reproducible)
```
# identical to latch-0004; only the in-code weight floor changed (0 -> 0.5). Off vs on, seeds 143/377:
niodoo/target/release/niodoo … --ghost-gravity 0.1 --correction-packets-path "$STORE" \
  [--correction-packet-neighborhood-weighting] --telemetry-out harness/runs/latch-0005/bc2_<off|on>_<seed>.jsonl
# code: principia.rs, weight = (0.75 + 0.25*cos).clamp(0.5, 1.0); committed 338e414, default off
```

## 4. What we expected
With the outlier kept at half weight, seed-143 arithmetic should stop regressing while seed 377 keeps its gains — a net win on both.

## 5. What actually happened
| Trap | seed 143 OFF→ON | seed 377 OFF→ON |
|---|---|---|
| mississippi | 0.35 → 0.35 (0) | 0.80 → 0.83 (+0.03) |
| strawberry | 0.43 → 0.38 (−0.05) | 0.60 → 0.75 (+0.15) |
| 17×24 | **0.93 → 0.06 (−0.87)** | 0.21 → 0.38 (+0.18) |
| 13×17 | 0.43 → 0.08 (−0.35) | 0.12 → 0.25 (+0.12) |
| **mean hold** | **0.53 → 0.22 (held 0/4)** | **0.43 → 0.55 (held 2/4)** |

- **Seed 377: a clean win** — all four traps improved, latch 0.11 → 0.22.
- **Seed 143: worse than latch-0004** — 17×24 collapsed to 0.06. A *gentler* setting performing *worse* than the aggressive one is not a smooth effect → the single-trap reading is **noisy**.

## 6. The scoreboard — the climb
| Rung | What we tried | Result |
|---|---|---|
| latch-0003 | is it magnitude? | no — it's aim/coherence |
| latch-0004 | basin-coherence (mute outliers to 0) | mixed: +377, −143 (mutes the holding packet) |
| **latch-0005** | damp-not-kill (outlier floor 0.5) | **+377 (all 4 up), −143 worse (17×24→0.06). Not robust; metric is noisy.** |

> The climb: we predicted softening the mute would fix seed 143. It didn't — it helped 377 and hurt 143 more. The honest lesson is about the *measurement*, not just the method.

## 7. The math, in plain words
- **Aggregate hold** (both seeds): OFF ~**0.48**, kill-to-0 ~**0.47**, damp-0.5 ~**0.39**. Plain: across everything, basin-coherence is roughly neutral-to-slightly-negative — the seed-377 gains and seed-143 losses cancel (or worse).
- **The seed split is real and consistent** — 377 improved under *both* weightings, 143 regressed under *both*. So it's a genuine seed-dependent effect, not pure noise.
- **But single-trap deltas are noisy** — a gentler weight (0.5 floor) made 17×24 seed-143 *worse* (0.35 → 0.06) than the harsh weight. That ordering can't be smooth, so per-trap geometry on one seed isn't trustworthy on its own.
- **Raw data:** `harness/runs/latch-0005/bc2_{off,on}_{143,377}.jsonl.gz` (67 MB) — kept, not buried.

## 8. Decision note (provenance)
**Decided by / on:** Claude (Opus 4.8) · 2026-06-25
Two honest conclusions: (1) basin-coherence is **not** a robust win as built — it trades seed 377 for seed 143, both forms net neutral-to-negative; (2) the **geometric latch (min target distance) is a noisy proxy** on a single seed/trap, so we've been partly chasing noise. **Next, change the measurement before more method-tuning:** (a) add more seeds (e.g. 42, 211) so per-trap noise averages out and the seed split is characterized, and (b) score the thing we actually care about — **answer correctness** off vs on — instead of only the geometric pull. If basin-coherence doesn't move *answers*, it isn't worth defaulting on regardless of the distance trace. Code stays flag-gated, default OFF (338e414).

## 9. Human verification / sign-off
- [ ] The prediction (§4) was recorded before the run
- [ ] I re-ran and saw 377 up / 143 down + the noisy 17×24 ordering — _initials / date_
- [ ] The numbers match the raw data — _initials_
- [ ] Notes:
