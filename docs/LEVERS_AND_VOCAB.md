# Niodoo levers & vocabulary (human guide)

What each runtime lever does, in plain English, so you do not have to read the Rust tree.

## Levers

| Key | Default | Safe range | Plain English |
|-----|---------|------------|----------------|
| `bridge_off` | `true` / off | on\|off | **Ghost basin bridge.** `bridge off` = no ghost force. |
| `bridge_influence_smoke` | `false` | bool | RUNBOOK "on arm" influence smoke path. |
| `bridge_influence_smoke_clamp` | `0.03` | ~0.01–0.08 | How hard smoke can push (L2 clamp). Higher = stronger steering. |
| `bridge_influence_selective` | `false` | bool | Only apply influence on selected routes. |
| `bridge_gate34_latch` | `false` | bool | Latch gate34 hold behavior (stickier correction). |
| `visible_request_gate` | `true` | bool | Let visible `[REQUEST: …]` tags open a short force gate. |
| `specialist_correction_apply` | `false` | bool | Apply specialist correction tensors when present. |
| `specialist_correction_clamp` | `0.03` | ~0.01–0.06 | Max specialist push. |
| `correction_packet_clamp` | `0.03` | ~0.01–0.06 | Per-packet correction force clamp. |
| `basin_coherence_gain` | `0.0` | 0.0–0.5 | Extra budget when basins cohere (gentle). |
| `temperature` | `0.0` | 0.0–1.0 | Decode temperature (0 = deterministic). |
| `max_steps` | `256` | 32–512 | Max generation steps. |
| `seed` | `42` | any u64 | RNG seed (repro). |
| `sigma` | `0.15` | ~0.05–0.3 | Physics OU noise ("jiggle"). |
| `theta` | `2.0` | ~1.0–3.0 | Physics mean-reversion / blend. |

### Cold-start only

These are fixed at process start / binary build: `model_path`, CUDA / `niodv4_bridge` feature,
basin registry path, bridge JSON path.

Maps to CLI (see `niodoo/src/cli.rs` + `RUNBOOK.md`):
`--bridge-off`, `--bridge-influence-smoke`, `--bridge-influence-smoke-clamp`,
`--visible-request-gate`, `--specialist-correction-*`, `--correction-packet-clamp`,
`--sigma-override`, `--theta-override`.

---

## Visible request vocabulary

Embed in model text (or type as user) when `visible_request_gate=true`:

| Tag | Meaning |
|-----|---------|
| `[REQUEST: SPIKE]` | Short force spike |
| `[REQUEST: FOCUS]` | Tighten toward current basin/topic |
| `[REQUEST: EXPLORE]` | Loosen / wander more |
| `[REQUEST: RESET]` | Cool down / clear short latch |
| `[REQUEST: REMEMBER]` | Bias memory / correction packets |

Implemented in `niodoo/src/runtime/control_surface.rs`.

---

## Lever bundles

| Name | Intent |
|------|--------|
| `bridge_off_baseline` | Bridge disabled baseline |
| `bridge_on_smoke` | Canonical on-arm smoke (clamp 0.03) |
| `letter_count_friendly` | Stronger smoke + gate34 for spelling/letter traps |
| `arithmetic_safe` | Conservative clamps for arithmetic |

Honest failure: if CUDA or the model is missing, say so — do not claim GPU success.
