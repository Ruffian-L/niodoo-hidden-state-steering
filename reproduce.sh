#!/usr/bin/env bash
# ============================================================================
# Niodoo correction — one-command reproduction (repo-relative).
#   - locates or asks you to build the bridge-feature binary
#   - downloads the EXACT model (bartowski Q5_K_M) if missing, and REFUSES to
#     run unless the sha256 matches (supply-chain / drift defense)
#   - runs bridge-OFF vs bridge-ON on one prompt and prints a claim card
#
# Run from the repo root:
#   ./reproduce.sh
#   ./reproduce.sh "What is 17 multiplied by 24? Give the final number only." 408
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- pinned facts (verified 2026-06-24, GPU NVIDIA GB10) -------------------
MODEL_SHA="14e10feba0c82a55da198dcd69d137206ad22d116a809926d27fa5f2398c69c7"
MODEL_REPO="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
MODEL_FILE="Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
REGISTRY_REL="niodv4/data/results/summaries/ghost_candidate_registry.json"

BIN="${NIODOO_BIN:-$REPO/niodoo/target/release/niodoo}"
MODEL_PATH="${NIODOO_MODEL_PATH:-$REPO/model/$MODEL_FILE}"
PROMPT="${1:-How many times does the letter r appear in the word strawberry? Give the final number only.}"
EXPECTED="${2:-3}"
OUT="$(mktemp -d)"

say(){ printf '%s\n' "$*"; }
die(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }

# ---- 1. binary (must carry the bridge feature) -----------------------------
if [ ! -x "$BIN" ]; then
  die "binary not found at $BIN
  Build it (needs CUDA toolchain):
      cd \"$REPO/niodoo\" && cargo build --release --bin niodoo --features niodv4_bridge
  or point NIODOO_BIN at a prebuilt bridge binary."
fi
hits=$(strings -a "$BIN" 2>/dev/null | grep -c ghost_candidate_registry || true)
[ "$hits" -gt 0 ] || die "binary at $BIN lacks the bridge feature marker (ghost_candidate_registry).
  Rebuild with --features niodv4_bridge. A binary without the marker cannot load basins."
say "[ok] binary has bridge feature ($hits marker hits)"

# ---- 2. basin registry must be present at the relative path the binary expects
[ -f "$REPO/$REGISTRY_REL" ] || die "missing $REGISTRY_REL — the binary loads basins from this exact relative path."
say "[ok] basin registry present"

# ---- 3. model: download if missing, ALWAYS verify the bytes ----------------
if [ ! -f "$MODEL_PATH" ]; then
  say "[..] model missing; downloading the verified file from $MODEL_REPO"
  mkdir -p "$(dirname "$MODEL_PATH")"
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$(dirname "$MODEL_PATH")" || die "huggingface-cli download failed"
  else
    curl -L --fail -o "$MODEL_PATH" "https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_FILE?download=true" \
      || die "curl download failed (install huggingface-cli for a more robust download)"
  fi
fi
say "[..] verifying model sha256 (~30s)"
actual=$(sha256sum "$MODEL_PATH" | awk '{print $1}')
[ "$actual" = "$MODEL_SHA" ] || die "MODEL HASH MISMATCH — refusing to run.
  expected $MODEL_SHA
  got      $actual
  The bytes differ from the published file. This run would not be the published run."
say "[ok] model verified == bartowski Q5_K_M"

# ---- 4. run (CWD must be the repo root so basins/bridge load by relative path)
cd "$REPO" || die "cannot cd $REPO"

run_arm(){ # name  outdir  extra-flags...
  local name="$1" d="$2"; shift 2; mkdir -p "$d"
  "$BIN" --model-path "$MODEL_PATH" --model-size 8b \
    --runtime-speed-profile eval-fast --stdout-profile telemetry --telemetry-profile full \
    --seed 42 --temperature 0.0 --max-steps 256 \
    --prompt "$PROMPT" --telemetry-out "$d/telemetry.jsonl" "$@" \
    >"$d/stdout.txt" 2>"$d/stderr.txt"
  local basins; basins=$(grep -o '"ghost_basins_loaded":[0-9]*' "$d/telemetry.jsonl" | head -1 | grep -o '[0-9]*$')
  local ans; ans=$(awk '/=== TURN 0 OUTPUT ===/{f=1;next} f' "$d/stdout.txt" \
                   | grep -E 'VISIBLE ANSWER:|WORKING ANSWER:' | tail -1 | sed 's/^[[:space:]]*//')
  [ -n "$ans" ] || ans=$(awk '/=== TURN 0 OUTPUT ===/{f=1;next} f' "$d/stdout.txt" \
                   | grep -E 'REQUEST: LOCK\]' | tail -1 | sed 's/^[[:space:]]*//')
  printf '  %-4s | basins=%-2s | %s\n' "$name" "${basins:-?}" "${ans:-<no answer parsed>}"
}

say ""
say "############### NIODOO CLAIM CARD ###############"
say "prompt        : $PROMPT"
say "correct answer: $EXPECTED"
say "model         : $MODEL_FILE (bartowski, sha256 verified)"
say "------------------------------------------------"
run_arm OFF "$OUT/off" --bridge-off
run_arm ON  "$OUT/on"  --bridge-influence-smoke --bridge-influence-smoke-clamp 0.03
say "------------------------------------------------"
say "Expect: ON lands '$EXPECTED' where OFF does not; basins=8 when ON, 0 when OFF."
say "raw outputs: $OUT"
