# Niodoo — hidden-state / inference-time steering

**GitHub:** [`Ruffian-L/niodoo-hidden-state-steering`](https://github.com/Ruffian-L/niodoo-hidden-state-steering)  
**Lead:** Jason Van Pham ([Ruffian-L](https://github.com/Ruffian-L)) — direction and final accountability  
**Built with:** Grok (xAI) · Claude (Anthropic) · ChatGPT / Codex (OpenAI) · Gemini (Google) — see [`CREDITS.md`](CREDITS.md) · [`AUTHORSHIP.md`](AUTHORSHIP.md)

> A **local runtime** that steers a **frozen** language model (no weight updates).  
> One **narrow, reproducible** correction result + the machinery behind it.  
> **Not** a chat product, **not** a consciousness claim, **not** broad benchmark SOTA.

**Built with Llama 3.1** — see `NOTICE.md` and `licenses/`. Weights download at run time and are **sha256-pinned**.

---

## Best face of this repo (what we actually want you to see)

| Strength | Where it lives |
|----------|----------------|
| **Epistemic rigor** | Losses kept public; regressions published next to wins; hash-pinned model/binary |
| **Narrow claim you can re-run** | `./reproduce.sh` → off vs on on the same 8 prompts |
| **Machine-checkable card** | [`claim_card.md`](claim_card.md) — correct answer printed beside every row |
| **Method writeup** | [`WHITEPAPER.md`](WHITEPAPER.md) |
| **How to build / not get false negatives** | [`RUNBOOK.md`](RUNBOOK.md) |
| **Rolling ladder (including fails)** | [`SCOREBOARD.md`](SCOREBOARD.md) |

If the first screen of an AI research repo only celebrates wins, distrust it.  
Here the **discipline is the product**; the bridge correction is the **example**.

---

## The claim (30 seconds)

On **Llama-3.1-8B-Instruct (Q5_K_M)**, temperature **0**, seed **42**, an 8-prompt trap battery:

| | |
|--|--|
| **Bridge off** | Model can lock a **wrong final** after correct intermediate steps (e.g. 17×24 → parts 340/68, locks **368**) |
| **Bridge on** | Same model, same weights: **corrects 4**, **holds 3 correct**, **breaks 1** (mississippi) |
| **Replay control** | “raspberry” stays **2** r’s — not forced to strawberry’s **3** |

Full table: [`claim_card.md`](claim_card.md). Mechanism and limits: [`WHITEPAPER.md`](WHITEPAPER.md).

### Claimed result — reproduced here

17 × 24: off locks 368; on lands 408.

![17x24 correction](images/correction_mult_17x24.png)

Count r in “strawberry”: off locks 2; on lands 3 (force magnitude capped — nudge near a decision boundary).

![strawberry correction](images/correction_strawberry.png)

---

## Run it (one command)

```bash
./reproduce.sh
```

Verifies the bridge-enabled binary, downloads the model only if missing, **refuses** on sha256 mismatch, runs **off** and **on**, prints answers next to ground truth.

```bash
# GPU (canonical)
cd niodoo && cargo build --release --bin niodoo --features niodv4_bridge

# CPU (functional, not bit-identical)
cargo build --release --bin niodoo --no-default-features --features niodv4_bridge
```

Details, hashes, arch flags: [`RUNBOOK.md`](RUNBOOK.md). Full eight-prompt table: `./harness/run_battery.sh`.

---

## What this is / is not

| This is | This is not |
|---------|-------------|
| Inference-time / residual-style **steering** of a frozen GGUF model | Fine-tuning or weight surgery |
| A **local** research runtime with telemetry | Cloud API product |
| A **hash-pinned** off-vs-on battery | “Beats Llama on everything” |
| Honest **negative** follow-ups on the scoreboard | Consciousness / feelings claims |
| Jason-led multi-AI **collaboration** (named) | Solo-genius folklore |

---

## Map of the tree

```text
README.md          ← you are here (public face)
claim_card.md      ← the checkable result
WHITEPAPER.md       ← mechanism + limits
RUNBOOK.md         ← build, hashes, gotchas
reproduce.sh       ← one-command re-run
SCOREBOARD.md      ← win / mix / fail ladder after the claim
CREDITS.md         ← who did what
AUTHORSHIP.md      ← short provenance
harness/           ← battery + latch run cards
evidence/          ← raw outputs
niodoo/            ← Rust runtime (bridge feature-gated)
niodv4/            ← basin registry the binary expects
niodoo_chat/       ← optional playable levers TUI (not the claim)
images/            ← claim figures + WIP shots
```

---

## Work still building (not the claim)

Favorite runtime shots — **not** part of the published correction claim:

The internal monitor idea (TDA-style loop sensing) — still under construction.

![internal monitor](images/internal_monitor.png)

Self-emitted control tags (SPIKE / FOCUS / …) on an unfine-tuned model — observational.

![telemetry tags](images/telemetry_tags.png)

Towel-drying loop / refocus mess — shown as-is.

![towel drying](images/towel_drying.png)

Optional day-to-day play surface: [`docs/LEVERS_AND_VOCAB.md`](docs/LEVERS_AND_VOCAB.md) · `niodoo_chat` presets.

```bash
cd niodoo_chat && cargo run --release --bin niodoo-chat
```

---

## Honest limits (published on purpose)

These are **features of the record**, not footnotes to hide:

1. **Bridge-off ≠ raw llama.cpp**  
   Niodoo wraps a steering system prompt (doubt-prime + control-tag protocol). On a pure vanilla 8-prompt battery (2026-06-24), **raw llama.cpp can answer 8/8** while niodoo’s wrap regresses on some deterministic facts. Weights identical — the wrap is the difference. Details below and in `runs/phase0_narration_diag.md`.

2. **One broken row on the claim card** (mississippi) — kept in the table.

3. **Later latch / basin experiments** (scoreboard) are **mixed / wash / negative** by design — geometry ≠ always better answers. Default for experimental levers stays conservative.

4. **Not a product.** Status: active research; claim battery is the polished spine.

### Detail — 2026-06-24 vanilla control & regression

Raw `llama.cpp`, same model bytes (sha256-verified), greedy, seed 42: **8/8** on the deterministic-recall set, including items the niodoo bridge-off arm misses. Cause: niodoo’s injected system context (`INTERNAL MONITOR` doubt-prime + `[REQUEST: …]` protocol), not inactive physics (`guardrail_active:false` throughout in that audit). Observed thrash on letter-count trajectories; model sometimes imitates control tags instead of answering.

Open / under test:

- route deterministic tasks to a deterministic answerer (`--tool-augmented`) as diagnostics, not the cognition target  
- equal chat template + guardrail relief so “off” is true vanilla, not “niodoo with bridge off”  
- move more steering off the visible prompt onto hidden-state / trajectory paths  

### Detail — 2026-07-01 prompt contract

A soft runtime prompt (“visible emission is optional”) stopped tag emission at temperature 0; restoring the imperative contract reproduced the claim card on the same binary. Diagnostic: `runs/phase0_narration_diag.md`.

---

## Reproducibility principle

**Trust the bytes, not the names.** Model and artifact identity = sha256. A clone that disagrees on a hash is not running the published configuration.

---

## Contact

Questions, corrections, or “this is wrong”: **jasonvanpham@niodoo.com**

## Licensing

- Project code: MIT (`LICENSE`)  
- Collaboration record: `CREDITS.md` · `AUTHORSHIP.md`  
- Llama 3.1 materials: Community License (`NOTICE.md`, `licenses/`)  
- Rust deps: `THIRD_PARTY_LICENSES.md`
