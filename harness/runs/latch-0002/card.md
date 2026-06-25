# Run card — latch-0002: first measurable latch (it grabs, then slips)

**Ran by:** Claude (Opus 4.8)   ·   **Date:** 2026-06-24   ·   **Tree:** niodoo-hidden-state-steering @ 10232f3 (feature/correction-to-trajectory-fixes)
**Verdict:** PASS — we can now measure a latch. And the latch is **weak: it grabs, then slips.**

## 1. What we asked
With the ghost force turned down and the correction-packet path actually firing, can we measure a latch — and does it hold?

## 2. What we ran
Loaded a prebuilt 160-packet correction store (letter-count + arithmetic), turned the ghost force down (`--ghost-gravity 1.0`), and swept 4 traps — mississippi-s, strawberry-r, 17×24, 13×17 — at seed 143, full per-token telemetry.

## 3. How we ran it (copy-paste reproducible)
```
MODEL=/home/ruffianl/projects/team_build/model/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf   # sha256 14e10feba0…c69c7
RAVE=/home/ruffianl/projects/team_build/niodoo/runtime_assets/rave_codec.safetensors
CODEBOOK=/home/ruffianl/projects/team_build/niodv4/experiments/encode_decode/niodv4/results/codebook_256.json
BRIDGE=niodoo/memory/runtime_bridge/niodoo_runtime_bridge.json
STORE=/home/ruffianl/projects/team_build/artifacts/correction_packets_letter_count_plus_arith_correct_20260503.jsonl
niodoo/target/release/niodoo --model-path "$MODEL" --model-size 8b \
  --runtime-speed-profile eval-fast --stdout-profile telemetry --telemetry-profile full \
  --seed 143 --temperature 0.0 --max-steps 110 \
  --session-script harness/traps/sweep_lettercount_arith.txt --reset-kv-cache-per-turn \
  --ghost-gravity 1.0 \
  --codebook-path "$CODEBOOK" --rave-codec-path "$RAVE" --runtime-bridge-path "$BRIDGE" \
  --correction-packets-path "$STORE" \
  --bridge-influence-smoke --bridge-influence-smoke-clamp 0.03 \
  --telemetry-out harness/runs/latch-0002/sweep_store_loaded_seed143.jsonl
# run from repo root · GPU NVIDIA GB10
```

## 4. What we expected
Packets fire and pull the probe toward the target; measure whether the pull **holds** or snaps back.

## 5. What actually happened
- **Correction packets fired** — 102–106 of 110 tokens per trap (was **0** in latch-0001), `force_norm` **1.74** (was 0). The path is finally engaged.
- The probe **got pulled toward the target in all 4 traps** — the distance dipped below where it grabbed.
- **But it didn't hold.** 3 of 4 ended *farther* than they grabbed. The latch forms, then slips.
- Answer-correctness scoring is not yet cleanly parsed (a harness TODO). The latch *geometry* is the measured result here.

## 6. The scoreboard — the climb
| Rung | What we tried | Result |
|---|---|---|
| latch-0001 | measure a latch in the bridge-smoke config | 0 fires; ghost is 99.8% of all force — nothing to measure |
| rung | turned ghost-gravity **10 → 1** | ghost share 99.8% → ~90%; still 0 fires (empty store) |
| rung | added the live-mint **writer** (`--correction-packets-out`) | 1 packet minted, wrong bucket/schema — still 0 fires |
| **latch-0002** | **loaded the 160-packet store** | **fires 102–106/110; latch forms (strength ~0.17) but slips (held 1/4)** |

> The climb: three tries got nothing to measure; this one made the latch visible — and it confirmed the read that it's *not loud enough to hold*.

## 7. The math, in plain words
- **latch_strength** = how much closer the probe got to the target at its best, where *0 = no closer, 1 = collapsed onto it*. Per trap: mississippi **0.16**, strawberry **0.18**, 17×24 **0.27**, 13×17 **0.07** (avg ~**0.17**) → a **weak grab**.
- **hold** = did it end at least as close as when it grabbed? **1 of 4 (only 17×24).** Plain: it grabs, then mostly lets go.
- **fires ~104/110 tokens; correction force 1.74 vs ghost 1.0** — the correction is now a real, measurable fraction of the push. In latch-0001 it was rounding error beside ghost-10.
- **Raw data:** `harness/runs/latch-0002/sweep_store_loaded_seed143.jsonl.gz` (18 MB) + `…seed143.txt` — kept, compressed, not buried. The numbers above are the lens.

## 8. Decision note (provenance)
**Decided by / on:** Claude (Opus 4.8) · 2026-06-24
This is the measurable **floor** for the "louder / neighborhood" work: latch_strength ~0.17, holds 1/4. The job now is to make the latch **hold** — the distance should shrink *and stay shrunk*, not drift back. The basin-coherence change (weight packets by agreement with the consensus pull; let the budget grow when many neighbors agree) targets exactly this slip. *Expected to move:* hold-rate up from 1/4, and latch_strength up from ~0.17, measured by re-running this exact sweep.

## 9. Human verification / sign-off
- [x] The prediction (§4) was recorded before the run
- [x] Reviewed and accepted — first measurable latch; it grabs but slips (strength ~0.17, held 1/4) — **jp / 2026-06-24**
- [x] The numbers match the raw data — jp
- [ ] Independently re-ran — _pending_
- Notes: Accepted as the floor. Keep climbing for a cleaner/better signal (isolate ghost, add seed 377) before the basin-coherence change.
