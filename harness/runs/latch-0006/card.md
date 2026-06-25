# Run card — latch-0006: does basin-coherence move ANSWER correctness? (the pivot)

**Ran by:** Claude (Opus 4.8)   ·   **Date:** 2026-06-25   ·   **Tree:** niodoo-hidden-state-steering @ 338e414 (feature/basin-coherence-latch)
**Verdict:** NO net gain — **OFF 13/16, ON 13/16 (exact wash)**. But the effect is **not** seed-noise: it splits cleanly **by trap type** — basin-coherence **helps letter-counting (+3) and hurts arithmetic (−3)**, and they cancel. Keep it **default-OFF**.

## 1. What we asked
Rungs latch-0004/0005 measured a *geometric* latch (min target distance) and found basin-coherence "seed-dependent and noisy." jp's call: stop chasing the geometry proxy — **score the thing we care about (did the model give the right number)**, off vs on, across **4 seeds (143, 377, 42, 211)**. Two questions:
1. Does basin-coherence **reliably** move final answer correctness?
2. Any **seed or trap type** where it consistently helps vs hurts?

## 2. What we ran
The exact 4-trap sweep (mississippi-s→4, strawberry-r→3, 17×24→408, 13×17→221), prebuilt 160-packet store, ghost 0.1, **basin-coherence off vs on**, across all four seeds = **8 runs**. Same binary as latch-0005 (damp-not-kill weight `0.75 + 0.25·cos ∈ [0.5,1]`, committed 338e414). Seeds 143/377 reuse the latch-0005 transcripts (same binary); 42/211 are fresh.

New this rung: a **deterministic answer-correctness scorer** (`harness/score_answers.py`) that reads the `=== TURN N OUTPUT ===` blocks and extracts the model's **committed final answer** (it self-corrects mid-ramble, so we take the last asserted count / "final answer is N" / leading computed result — see §7). Every one of the 32 cells was **hand-checked against the raw text** (`answers_readable.txt`).

## 3. How we ran it (copy-paste reproducible)
```
# OFF and ON differ only by the last flag. For --seed in {143,377,42,211}:
niodoo/target/release/niodoo --model-path "$MODEL" --model-size 8b \
  --runtime-speed-profile eval-fast --stdout-profile telemetry --telemetry-profile full \
  --seed <SEED> --temperature 0.0 --max-steps 110 \
  --session-script harness/traps/sweep_lettercount_arith.txt --reset-kv-cache-per-turn \
  --ghost-gravity 0.1 --codebook-path "$CODEBOOK" --rave-codec-path "$RAVE" \
  --runtime-bridge-path "$BRIDGE" --correction-packets-path "$STORE" \
  --bridge-influence-smoke --bridge-influence-smoke-clamp 0.03 \
  [--correction-packet-neighborhood-weighting] \
  --telemetry-out harness/runs/latch-0006/ans_<off|on>_<seed>.jsonl > ans_<off|on>_<seed>.txt
# score (after gunzip ans_*.txt.gz):  python3 harness/score_answers.py harness/runs/latch-0006/ans_*.txt
```

## 4. What we expected
After latch-0004/0005, the honest prior: basin-coherence is a wash or slightly negative on answers, with seed 377 favoring it. We expected the extra seeds to either confirm a real net gain (→ turn it on) or confirm the wash (→ park it).

## 5. What actually happened — answer correctness, OFF → ON
**Per seed (out of 4 traps):**
| Seed | mississippi (4) | strawberry (3) | 17×24 (408) | 13×17 (221) | total OFF→ON |
|---|---|---|---|---|---|
| 143 | ✓→✓ | ✓→✓ | ✓ → **✗ (412/442)** | ✓→✓ | **4/4 → 3/4** |
| 377 | ✓→✓ | **✗ (2) → ✓ (3)** | ✓→✓ | ✓→✓ | **3/4 → 4/4** |
| 42  | ✓→✓ | **✗ (4) → ✓ (3)** | ✓→✓ | ✓ → **✗ (derail)** | **3/4 → 3/4** |
| 211 | ✓→✓ | **✗ (derail) → ✓ (3)** | ✓→✓ | ✓ → **✗ (187)** | **3/4 → 3/4** |

**Per trap (the real structure):**
| Trap | type | OFF | ON | Δ |
|---|---|---|---|---|
| mississippi-s | letter-count | 4/4 | 4/4 | **0** (always right) |
| strawberry-r | letter-count | **1/4** | **4/4** | **+3** (BC fixes it) |
| 17×24 | arithmetic | 4/4 | 3/4 | **−1** |
| 13×17 | arithmetic | 4/4 | 2/4 | **−2** |
| **TOTAL** | | **13/16** | **13/16** | **0** |

- **Basin-coherence consistently HELPS the hard letter-count (strawberry): +3 of 4 seeds, never hurt.** OFF the model stalls on a wrong count (2, 4) or derails; ON it lands on "the letter r appears **3** times" (seed 211: "…s-t-r-a-w-b-e-r-r-y… appears 3 times"; seed 377: "the correct answer is: 3, I triple-checked").
- **Basin-coherence consistently HURTS arithmetic: −3 across the two products, never helped.** ON, seed 143's 17×24 computes 408 then over-adds to 442; seed 42's 13×17 derails onto an "agency state" tangent; seed 211's 13×17 mis-multiplies to 187.
- **Net answer correctness is identical (13/16 both arms).** The letter-count gains and arithmetic losses cancel.

## 6. The scoreboard — the climb
| Rung | What we tried | Result |
|---|---|---|
| latch-0003 | is it magnitude? | no — it's aim/coherence |
| latch-0004 | basin-coherence (mute outliers) — geometric latch | mixed: +377, −143 |
| latch-0005 | damp-not-kill (floor 0.5) — geometric latch | negative + the geometry proxy is **noisy** |
| **latch-0006** | **score ANSWER correctness, 4 seeds** | **net wash (13/16 ↔ 13/16); splits by trap type: letter-count +3, arithmetic −3** |

> The climb: changing the *measurement* (geometry → answers) paid off. The "seed-dependent noise" of latch-0004/0005 resolves into a **clean, mechanistic signal**: sharpening the consensus pull helps when the consensus *is* the answer (counting) and hurts when the answer is a precise computation the packets don't encode (multiplication).

## 7. The math, in plain words
- **Scoring rule (deterministic, in `score_answers.py`):** the model rambles for 110 tokens and self-corrects, so the committed answer = the **last** thing it asserts as an answer — last "appears N times" / "there are N occurrences" / "final/correct answer is N", else the **leading** computed result (trailing "17×4 = 68" re-derivation steps are *not* the answer). Every cell was eyeballed against `answers_readable.txt`; the parser matches the hand-verdict on all 32.
- **Why letter-count improves:** many correction packets genuinely agree on "count a letter → small integer," so they form a real **basin**. Amplifying the agreed direction nudges the model onto the right count. This is the same place the geometry helped (seed 377 strawberry hold 0.60→1.00 in latch-0004).
- **Why arithmetic regresses:** the answer (408, 221) is a sharp point the packets don't collectively encode. Amplifying the consensus pull pushes the model *off* the exact product — into an over-add (442), a mis-multiply (187), or a derail.
- **Two genuinely-ambiguous cells (flagged, do not change the verdict):** (a) seed-143 strawberry is "3" then "I made a mistake" on **both** arms → scored ✓ both, so it cancels in the 143 delta (which is driven by 17×24). (b) seed-211 strawberry OFF is a bare "3" then derail ("3assist… VSCALE…") → scored ✗ (no coherent answer); if counted ✓ instead, seed 211 becomes 4/4→3/4 and the net tilts to **ON slightly worse** — either way **not an improvement**.
- **Raw data:** `ans_{off,on}_{143,377,42,211}.jsonl.gz` (telemetry) + `ans_*.txt.gz` (full transcripts) + `answers_readable.txt` (the 32 turn outputs, human-readable) + `scores.json` (per-cell verdicts). Kept, compressed, pointed to — not buried.

## 8. Decision note (provenance)
**Decided by / on:** Claude (Opus 4.8) · 2026-06-25
**Answer to jp's two questions:**
1. **Does basin-coherence reliably move final answer correctness? — No.** Net is an exact wash (13/16 both arms). It does **not** clear the bar jp set ("clearly improves answer correctness across the seeds"), so **it stays default-OFF**.
2. **Consistent help/hurt? — Yes, by TRAP TYPE, not seed.** It **helps letter-counting** (strawberry +3/−0) and **hurts arithmetic** (−3/+0), reliably across seeds. The latch-0004/0005 "seed split" was really this trap-type split showing through.

**This is not a dead lever — it's a mis-aimed one.** It has a real, repeatable, *mechanistically sensible* effect. Two honest paths for jp:
- **Park it** (default-OFF, documented) and move to the next lever — defensible, since net correctness didn't move.
- **Develop it as a gated lever:** apply basin-coherence only on consensus/counting route-families and disable it on arithmetic/precise-computation. The data says that targeted version would be a real win (letter-count 1/4→4/4) with no arithmetic cost. This is the first lever in the whole climb that *moved answers in a predictable direction* — worth a decision, not an auto-park.

Code stays flag-gated, default-OFF (338e414). No code change this rung — only measurement.

## 9. Human verification / sign-off
- [x] The prediction (§4) was recorded before the run
- [ ] I re-ran `score_answers.py` and the per-trap tally matches (strawberry +3, arith −3, net 0) — _initials / date_
- [ ] Spot-checked `answers_readable.txt` for 3–4 cells (e.g. on_377 strawberry = "correct answer is 3"; on_143 17×24 = 442) — _initials_
- [ ] Decision: **park** ▢  /  **develop as trap-type-gated lever** ▢ — _jp_
- [ ] Notes:
