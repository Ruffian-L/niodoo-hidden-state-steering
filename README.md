# Niodoo

A small local runtime that runs alongside a frozen language model and steers it. It is not a model, and it does not
retrain weights. This repository contains one narrow, reproducible result and the runtime behind it.

Status: not polished, actively worked on. The result here runs end to end. The wider system does not yet.

## The result, in one paragraph

On a frozen Llama-3.1-8B-Instruct (Q5_K_M), the model locks a wrong final answer on a class of prompts even when its
own steps were correct — it computes 340 and 68, then locks 368. With the bridge on, it catches the slip and lands
408. On an eight-prompt battery at temperature 0, seed 42, the bridge corrected four wrong answers, left three
correct answers unchanged, and broke one. It does not replay a memorized answer: a word with two r's stays two, it
is not forced to three. Full numbers, mechanism, and limits are in [`WHITEPAPER.md`](WHITEPAPER.md).

## The runtime, in pictures

The first three are **not claims** — they are work still being built, shown because they are some of the author's
favorite shots of the runtime. The last two are the result this repository reproduces.

### Work in progress — not claimed yet

The internal monitor: the model reasoning against its own monitor signal. The monitor uses TDA (topological) analysis
to detect when the model is looping. Still being built.

![internal monitor](images/internal_monitor.png)

Self-emitted telemetry tags: this model was never fine-tuned, yet it emits its own control tags
(SPIKE / FOCUS / RESET / REMEMBER / LOCK), shown here with their counts across runs.

![telemetry tags](images/telemetry_tags.png)

The towel-drying problem: the model catching itself in a loop and asking to refocus. Messy, shown as-is. LOCK fires
when the model acknowledges the right answer; SPIKE injects a perturbing force during inference to break it out of the loop.

![towel drying](images/towel_drying.png)

### The claimed result — reproduced in this repo

17 × 24: bridge off computes the right parts (340, 68) then locks 368; bridge on catches the slip and lands 408.

![17x24 correction](images/correction_mult_17x24.png)

Counting the r's in "strawberry": bridge off locks 2; bridge on lands 3. Identical at clamp 0.03, 0.5, and 2.0 — the
force is capped — so the correction is the nudge near a decision boundary, not its magnitude.

![strawberry correction](images/correction_strawberry.png)

## Run it

```bash
./reproduce.sh
```

It verifies the binary carries the bridge feature, downloads the exact model only if it is missing, refuses to run
if the model's sha256 does not match the published one, then runs the off and on arms and prints each answer next to
the correct one. Full procedure, hashes, and the working-directory requirement are in [`RUNBOOK.md`](RUNBOOK.md).

The binary is platform-specific (CUDA, `sm_121`). Build-from-source and portability are open work; see the runbook.

## What is here

```
WHITEPAPER.md     the result, the mechanism, the limits
README.md         this file
RUNBOOK.md        how to run, pinned hashes, gotchas
reproduce.sh      one command: verify, run off vs on, print the claim card
CREDITS.md        who did what
claim_card.md     the witnessed-correction result, machine-checkable
harness/          the scripts that produced the battery
evidence/         raw model output from the runs
niodoo/           the runtime source (Rust). The bridge is feature-gated.
niodv4/           the basin registry the binary loads, at the relative path it expects
```

## What this is not

It is not a claim of broad benchmark superiority, it is not a finished product, and the bridge-off arm is not yet a
true vanilla baseline. Those gaps are stated in the whitepaper, not hidden.

## Reproducibility principle

Trust the bytes, not the names. Every model and artifact is identified by sha256. A clone that disagrees on a hash is
telling you it is not running the published configuration.

## Contact

Questions, corrections, or if you think something here is wrong: jasonvanpham@niodoo.com
