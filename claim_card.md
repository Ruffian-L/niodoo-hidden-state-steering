# Claim card: bridge-on correction

One claim, stated so it can be checked. The correct answer is printed next to every result; you should never have to
infer what the right answer was.

## Claim

On a frozen Llama-3.1-8B-Instruct (Q5_K_M), at temperature 0 and seed 42, turning the Niodoo bridge on corrects a
class of last-step errors the same model locks with the bridge off — and it does so by reasoning, not by replaying a
memorized answer.

## Provenance (trust the bytes)

| | |
|---|---|
| model | `Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf`, sha256 `14e10feba0c82a55da198dcd69d137206ad22d116a809926d27fa5f2398c69c7` (bartowski) |
| binary | bridge-feature build; verified build sha256 `8c7778276517bcfc684f7bb008ca859759676e833e3bef54344689806eba95d8` |
| basins | `niodv4/data/results/summaries/ghost_candidate_registry.json` (8 basins) |
| decode | greedy, temperature 0, seed 42, 256 steps |
| off / on | `--bridge-off` / `--bridge-influence-smoke --bridge-influence-smoke-clamp 0.03` |
| GPU | NVIDIA GB10, driver 580.159.03, 2026-06-24 |

## Result

| Prompt | Correct | Off | On | Verdict |
|---|---|---|---|---|
| Bat and ball ($1.10, bat $1 more) | $0.05 | $0.05 | $0.05 | unchanged |
| Count r in "strawberry" | 3 | 2 | 3 | corrected |
| 17 × 24 | 408 | 368 | 408 | corrected |
| 13 × 17 | 221 | 321 | 221 | corrected |
| 23 × 18 | 414 | (no land) | 414 | corrected |
| Count r in "raspberry" | 2 | 2 | 2 | unchanged — replay control |
| Count a in "banana" | 3 | 3 | 3 | unchanged |
| Count s in "mississippi" | 4 | 4 | 6 | broken |

Four corrected, three left correct, one broken.

## How to read it

- **Corrected**: off locked a wrong final answer; on landed the right one. On "17 × 24" the off model computes the
  right parts (340, 68) and then locks 368; the on model catches it and lands 408.
- **Replay control ("raspberry")**: if the bridge injected a memorized "3", raspberry (two r's) would be wrong. It
  stays 2. The correction is reasoning, not replay. This is the single most important row.
- **Broken ("mississippi")**: the bridge turned a correct 4 into 6. The pull is unaimed (toward the nearest basin,
  not toward the correct token), so it sometimes pushes a right answer off. This is the honest cost, and the open
  problem.

## Check it yourself

```bash
./reproduce.sh                              # strawberry, off vs on
./harness/run_battery.sh                    # the full table above
```

Raw model output for each row is under `evidence/`.

## Boundary

This is a narrow claim. It is not broad benchmark superiority, the off arm is not yet a true vanilla baseline, and
only one seed has been run. See `WHITEPAPER.md` sections 4–5 for the mechanism and the full list of limits.
