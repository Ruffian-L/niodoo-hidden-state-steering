# Runbook

How to run Niodoo's bridge correction from a clone of this repository. Read the gotchas; they are the difference
between "it works" and "it looks broken."

## Pinned configuration (verified 2026-06-24, GPU NVIDIA GB10)

```
binary    niodoo/target/release/niodoo   (built with --features niodv4_bridge; sha256 of the verified build
                                          was 8c7778276517bcfc684f7bb008ca859759676e833e3bef54344689806eba95d8)
model     Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf   sha256 14e10feba0c82a55da198dcd69d137206ad22d116a809926d27fa5f2398c69c7  (bartowski)
basins    niodv4/data/results/summaries/ghost_candidate_registry.json   (8 basins; shipped in this repo)
bridge    niodoo/memory/runtime_bridge/niodoo_runtime_bridge.json       (shipped in this repo)
decode    --seed 42 --temperature 0.0 --max-steps 256   (deterministic)
off arm   --bridge-off
on arm    --bridge-influence-smoke --bridge-influence-smoke-clamp 0.03
```

## One command

```bash
./reproduce.sh
```

It checks the binary for the bridge feature, downloads the model only if missing, refuses to run on a hash
mismatch, runs the off and on arms, and prints each answer next to the correct one. To reproduce the full
eight-prompt table from the whitepaper:

```bash
./harness/run_battery.sh
```

## Build the binary

The binary is not committed (it is platform-specific). Build it from the crate. A recent Rust toolchain
(`rustup`, stable) is the only hard requirement.

### GPU build (canonical reproduction — needs an NVIDIA GPU + CUDA toolkit)

```bash
cd niodoo
cargo build --release --bin niodoo --features niodv4_bridge
```

Requirements: the CUDA toolkit (`nvcc`, CUDA 13.x) and an NVIDIA GPU. The kernel is AOT-compiled for `sm_121`
(Blackwell / GB10) by default. **On a different NVIDIA GPU, set the arch** before building, e.g. Ampere:

```bash
NIODOO_CUDA_ARCH=sm_80 cargo build --release --bin niodoo --features niodv4_bridge   # sm_86 / sm_89 / sm_90 …
```

(If `nvcc` is missing the build still succeeds and the runtime JIT-compiles kernels via NVRTC for the live device.)

### CPU build (runs on any machine — no GPU, no CUDA toolkit)

```bash
cd niodoo
cargo build --release --bin niodoo --no-default-features --features niodv4_bridge
```

This drops `cudarc` and builds candle for CPU; no NVIDIA hardware or CUDA libraries are needed. The binary runs on
CPU automatically (`--require-cuda` defaults to `false`). CPU is **functional, not the canonical reproduction** —
results can differ from the GPU run at the last bits of float precision, and it is slower. Use the GPU build to
reproduce the published numbers.

## Gotchas (these caused the recurring "it's broken")

1. **Run from the repo root.** The binary loads the basin registry from the hardcoded relative path
   `niodv4/data/results/summaries/ghost_candidate_registry.json` and the bridge JSON from
   `niodoo/memory/runtime_bridge/`. Run from anywhere else and you get `bridge_enabled=true` but
   `ghost_basins_loaded=0` — the "enabled but empty" state that looks like a regression. It is a wrong working
   directory, not a dead engine.
2. **Confirm the binary carries the bridge feature:** `strings -a niodoo/target/release/niodoo | grep -c ghost_candidate_registry`
   must be greater than 0. A binary built without `--features niodv4_bridge` reports `bridge_enabled` but loads zero basins.
3. **Trust the bytes.** If `reproduce.sh` reports a model hash mismatch, you have a different file than the one that
   produced the published result. The numbers may differ. That is the script working as intended.

## What a healthy run looks like

`bridge_enabled=true`, `ghost_basins_loaded=8`. The model's words appear after `=== TURN 0 OUTPUT ===`. Off locks the
wrong answer; on lands the right one on the corrected prompts (see `claim_card.md`).

## Open controls (not yet done)

- A true vanilla baseline: raw llama.cpp on the same prompts. `--bridge-off` is Niodoo with the bridge disabled, not vanilla.
- Seed robustness: only seed 42, temperature 0 has been run.
- Portability: CPU builds (`--no-default-features`) and other NVIDIA archs (`NIODOO_CUDA_ARCH`) now build and run.
  The published numbers are still pinned to the GPU `sm_121` reproduction; CPU/other-arch parity is not yet characterized.
