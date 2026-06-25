# Scoreboard

The rolling ledger of the climb — every eval's rung, newest at the bottom. Each run card restates its own recent rungs inline; this is the full ladder. Progress, not blame (STANDARDS.md §3).

| Date | Ran by | What we tried | Result | Card |
|---|---|---|---|---|
| 2026-06-24 | prior run | bridge on vs off, 8-prompt trap battery, seed 42, temp 0 | 4 corrected, 3 held correct, 1 broken (mississippi) | `claim_card.md` |
| 2026-06-24 | Claude (Opus 4.8) | latch-0001: can we measure a latch? strawberry, bridge-on, seed 143 | answer correct (3), but no latch engaged — ghost force is 99.8% of total; 0/97 packet fires. Confound quantified. | `harness/runs/latch-0001/card.md` |
| 2026-06-24 | Claude (Opus 4.8) | latch-0002: ghost down (10→1) + load 160-packet store, sweep 4 traps, seed 143 | FIRST measurable latch — packets fire 102–106/110, but latch is weak (strength ~0.17) and slips (held 1/4). Floor set for the "louder" work. | `harness/runs/latch-0002/card.md` |
| 2026-06-24 | Claude (Opus 4.8) | latch-0003: is it magnitude? isolate ghost (0.1) + louder correction (force ~139), seeds 143+377 | NO — quieter ghost didn't help, louder correction made it WORSE (overshoot, held 0/4). Magnitude ruled out → it's aim/coherence. | `harness/runs/latch-0003/card.md` |
| 2026-06-25 | Claude (Opus 4.8) | latch-0004: basin-coherence (mute outliers, consensus pull), flag-gated, on vs off, seeds 143+377 | MIXED — helps coherent basins (seed 377 latch 0.11→0.26; strawberry hold 0.60→1.00) but hurts scattered arith (seed 143 17×24 hold 0.93→0.35). Promising, not robust. Next: damp-not-kill. | `harness/runs/latch-0004/card.md` |
| 2026-06-25 | Claude (Opus 4.8) | latch-0005: damp-not-kill (outlier floor 0.5), on vs off, seeds 143+377 | NEGATIVE — helped seed 377 (all 4 traps) but hurt seed 143 MORE (17×24 hold 0.93→0.06). Aggregate neutral-to-worse. Single-trap latch is noisy. Next: more seeds + score ANSWER correctness, not just geometry. | `harness/runs/latch-0005/card.md` |
| 2026-06-25 | Claude (Opus 4.8) | latch-0006 (PIVOT): score ANSWER correctness off vs on, 4 seeds (143/377/42/211) | WASH on net (OFF 13/16, ON 13/16) — but clean split BY TRAP TYPE: basin-coherence HELPS letter-count (strawberry 1/4→4/4, +3) and HURTS arithmetic (17×24 −1, 13×17 −2). Not seed-noise — mechanistic. Stays default-OFF; candidate for a trap-type-gated lever. | `harness/runs/latch-0006/card.md` |
