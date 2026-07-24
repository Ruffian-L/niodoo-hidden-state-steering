# Niodoo chat levers & vocabulary (human guide)

You do **not** need to reverse-engineer the Rust tree. Use this page + `niodoo-chat`.

**Stop line (2026-07-24):** levers TUI + presets + persona memory are the freeze surface. No more heavy physics surgery required for playable chat.

## Quick start

```bash
cd niodoo_chat
cargo run --release --bin niodoo-chat
```

Commands inside the TUI:

| Command | What it does |
|---------|----------------|
| `/levers` | Show live lever snapshot |
| `/set KEY VALUE` | Change a lever **mid-session** (hot) |
| `/preset NAME` | Load a named preset bundle |
| `/presets` | List presets + one-line intent |
| `/persona shep\|echo\|lumina` | Load agent memory into context |
| `/vocab` | SPIKE / FOCUS / EXPLORE / RESET / REMEMBER |
| `/cli` | Print CLI flags matching current levers |
| `/quit` | Exit |

Plain text (no leading `/`) is a chat turn. Offline mode replies with a mock that still echoes levers/persona (real GPU generation needs the full `niodoo` binary + model).

---

## Live levers (hot in TUI)

| Key (aliases) | Default | Safe range | Plain English |
|---------------|---------|------------|----------------|
| `bridge_off` / `bridge` | `true` / off | on\|off | **Ghost basin bridge.** `bridge off` = no ghost force. `bridge on` clears `bridge_off`. |
| `bridge_influence_smoke` / `smoke` | `false` | bool | RUNBOOK “on arm” influence smoke path. |
| `bridge_influence_smoke_clamp` / `clamp` | `0.03` | ~0.01–0.08 | How hard smoke can push (L2 clamp). Higher = stronger steering. |
| `bridge_influence_selective` / `selective` | `false` | bool | Only apply influence on selected routes. |
| `bridge_gate34_latch` / `gate34` | `false` | bool | Latch gate34 hold behavior (stickier correction). |
| `visible_request_gate` / `request_gate` | `true` | bool | Let visible `[REQUEST: …]` tags open a short force gate. |
| `specialist_correction_apply` / `specialist` | `false` | bool | Apply specialist correction tensors when present. |
| `specialist_correction_clamp` | `0.03` | ~0.01–0.06 | Max specialist push. |
| `correction_packet_clamp` / `packet_clamp` | `0.03` | ~0.01–0.06 | Per-packet correction force clamp. |
| `basin_coherence_gain` / `coherence` | `0.0` | 0.0–0.5 | Extra budget when basins cohere (gentle). |
| `temperature` / `temp` | `0.0` | 0.0–1.0 | Decode temperature (0 = deterministic). |
| `max_steps` / `steps` | `256` | 32–512 | Max generation steps. |
| `seed` | `42` | any u64 | RNG seed (repro). |
| `sigma` | `0.15` | ~0.05–0.3 | Physics OU noise (“jiggle”). |
| `theta` | `2.0` | ~1.0–3.0 | Physics mean-reversion / blend. |

### Cold-start only (not TUI-hot)

These stay process-start / binary build: `model_path`, CUDA/`niodv4_bridge` feature, basin registry path, bridge JSON path. TUI `/cli` shows how to pass **hot** levers into a future `niodoo` spawn; model still required for real tokens.

Maps to existing CLI (see `niodoo/src/cli.rs` + `RUNBOOK.md`):
`--bridge-off`, `--bridge-influence-smoke`, `--bridge-influence-smoke-clamp`, `--visible-request-gate`, `--specialist-correction-*`, `--correction-packet-clamp`, `--sigma-override`, `--theta-override`.

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

Implemented in `niodoo/src/runtime/control_surface.rs` and mirrored in `niodoo_chat` vocab.

---

## Presets (≈5 playable)

| Name | Intent |
|------|--------|
| `bridge_off_baseline` | Bridge disabled baseline |
| `bridge_on_smoke` | Canonical on-arm smoke (clamp 0.03) |
| `letter_count_friendly` | Stronger smoke + gate34 for spelling/letter traps |
| `arithmetic_safe` | Conservative clamps for arithmetic |
| `full_agency` | Bridge on + selective + full visible control surface |

Files: `niodoo_chat/presets/*.json`.

```text
/preset bridge_on_smoke
/set clamp 0.05
/persona shep
hello
```

---

## Personas (Shep / Echo / Lumina)

`/persona shep|echo|lumina` loads non-empty text from:

1. `niodoo_chat/fixtures/{persona}/` (checked-in)
2. `…/vault/research_team/` when ghost preflight is mounted
3. `…/ghost_team/home/{persona}` when present
4. Extra roots via `NIODOO_PERSONA_ROOTS` (colon-separated)

Memory is injected as context bytes for the session (not credits prose).

---

## Offline vs full model

| Mode | When | Behavior |
|------|------|----------|
| **Offline (default)** | No GPU/model | Mock replies; levers/presets/persona **fully live** |
| **Full niodoo** | Built `--features niodv4_bridge` + GGUF | Use `/cli` flags + RUNBOOK one-shot; interactive spawn optional later |

Honest failure: if CUDA/model missing, use offline TUI — do not claim GPU success.
