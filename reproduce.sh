#!/usr/bin/env bash
# ============================================================================
# Niodoo correction — one-command reproduction (repo-relative).
#   - locates or asks you to build the bridge-feature binary
#   - downloads the EXACT model (bartowski Q5_K_M) if missing, and REFUSES to
#     run unless the sha256 matches (supply-chain / drift defense)
#   - verifies the shipped tokenizer bytes the same way
#   - runs bridge-OFF vs bridge-ON on one prompt and prints a claim card
#   - raw outputs (full transcripts + telemetry) are kept in runs/raw/ inside
#     the repo — never /tmp. Gitignored, but every reproducer can read the
#     raw chats behind their own claim card, not just the clean answer.
#
# Run from the repo root:
#   ./reproduce.sh
#   ./reproduce.sh "What is 17 multiplied by 24? Give the final number only." 408
#
# Device: NIODOO_DEVICE=auto|cuda|cpu (default auto). The published claim card
# was produced on CUDA (NVIDIA GB10, CUDA 13.x). CPU is supported for
# portability via the CPU build:
#   cd niodoo && cargo build --release --bin niodoo --no-default-features \
#       --features niodv4_bridge --target-dir target-cpu
# Every claim card states which device produced it — CPU and CUDA results are
# not interchangeable evidence.
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- pinned facts (verified 2026-06-24, GPU NVIDIA GB10) -------------------
MODEL_SHA="14e10feba0c82a55da198dcd69d137206ad22d116a809926d27fa5f2398c69c7"
MODEL_REPO="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
MODEL_FILE="Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
REGISTRY_REL="niodv4/data/results/summaries/ghost_candidate_registry.json"

MODEL_PATH="${NIODOO_MODEL_PATH:-$REPO/model/$MODEL_FILE}"
PROMPT="${1:-How many times does the letter r appear in the word strawberry? Give the final number only.}"
EXPECTED="${2:-3}"

say(){ printf '%s\n' "$*"; }
die(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }

# ---- 0. device: auto-detect, or force with NIODOO_DEVICE=cuda|cpu ----------
DEVICE_MODE="${NIODOO_DEVICE:-auto}"
if [ "$DEVICE_MODE" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    DEVICE_MODE="cuda"
  else
    DEVICE_MODE="cpu"
  fi
fi
case "$DEVICE_MODE" in
  cuda) REQUIRE_CUDA=true;  DEFAULT_BIN="$REPO/niodoo/target/release/niodoo" ;;
  cpu)  REQUIRE_CUDA=false; DEFAULT_BIN="$REPO/niodoo/target-cpu/release/niodoo"
        # fall back to the CUDA-featured binary if no CPU build exists; it can
        # run CPU-side on a machine that still has the CUDA libraries.
        [ -x "$DEFAULT_BIN" ] || DEFAULT_BIN="$REPO/niodoo/target/release/niodoo" ;;
  *) die "NIODOO_DEVICE must be auto, cuda, or cpu (got '$DEVICE_MODE')" ;;
esac
BIN="${NIODOO_BIN:-$DEFAULT_BIN}"
say "[ok] device mode: $DEVICE_MODE"

# ---- raw outputs live in the repo, never /tmp -------------------------------
RUN_ID="${NIODOO_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="${NIODOO_OUT_DIR:-$REPO/runs/raw/reproduce_${RUN_ID}_${DEVICE_MODE}}"
mkdir -p "$OUT"

# ---- 1. binary (must carry the bridge feature) -----------------------------
if [ ! -x "$BIN" ]; then
  die "binary not found at $BIN
  Build it:
      CUDA: cd \"$REPO/niodoo\" && cargo build --release --bin niodoo --features niodv4_bridge
      CPU:  cd \"$REPO/niodoo\" && cargo build --release --bin niodoo --no-default-features --features niodv4_bridge --target-dir target-cpu
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

# ---- 3b. tokenizer: ship-verified bytes, same trust rule as the model -------
# Redistributed Llama 3.1 material (see NOTICE.md). Verified functionally
# identical to the tokenizer embedded in the hash-pinned GGUF above.
TOKENIZER_PATH="$REPO/model/tokenizer.json"
TOKENIZER_SHA="79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4"
[ -f "$TOKENIZER_PATH" ] || die "missing model/tokenizer.json — it ships with this repo; restore it from git."
actual_tok=$(sha256sum "$TOKENIZER_PATH" | awk '{print $1}')
[ "$actual_tok" = "$TOKENIZER_SHA" ] || die "TOKENIZER HASH MISMATCH — refusing to run.
  expected $TOKENIZER_SHA
  got      $actual_tok
  The bytes differ from the published tokenizer. This run would not be the published run."
say "[ok] tokenizer verified"

# ---- 3c. record the environment next to the raw outputs --------------------
{
  echo "run_id: $RUN_ID"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "device_mode: $DEVICE_MODE"
  echo "binary: $BIN"
  echo "binary_sha256: $(sha256sum "$BIN" | awk '{print $1}')"
  echo "model_sha256: $MODEL_SHA (verified)"
  echo "tokenizer_sha256: $TOKENIZER_SHA (verified)"
  echo "uname: $(uname -srm)"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
} > "$OUT/environment.txt"

# ---- 4. run (CWD must be the repo root so basins/bridge load by relative path)
cd "$REPO" || die "cannot cd $REPO"

run_arm(){ # name  outdir  extra-flags...
  local name="$1" d="$2"; shift 2; mkdir -p "$d"
  "$BIN" --model-path "$MODEL_PATH" --model-size 8b \
    --runtime-speed-profile eval-fast --stdout-profile telemetry --telemetry-profile full \
    --require-cuda "$REQUIRE_CUDA" \
    --seed 42 --temperature 0.0 --max-steps 256 \
    --prompt "$PROMPT" --telemetry-out "$d/telemetry.jsonl" "$@" \
    >"$d/stdout.txt" 2>"$d/stderr.txt"
  local basins; basins=$(grep -o '"ghost_basins_loaded":[0-9]*' "$d/telemetry.jsonl" | head -1 | grep -o '[0-9]*$')
  local dev; dev=$(grep -m1 '^DEVICE:' "$d/stdout.txt" | sed 's/^DEVICE: //')
  local ans; ans=$(awk '/=== TURN 0 OUTPUT ===/{f=1;next} f' "$d/stdout.txt" \
                   | grep -E 'VISIBLE ANSWER:|WORKING ANSWER:' | tail -1 | sed 's/^[[:space:]]*//')
  [ -n "$ans" ] || ans=$(awk '/=== TURN 0 OUTPUT ===/{f=1;next} f' "$d/stdout.txt" \
                   | grep -E 'REQUEST: LOCK\]' | tail -1 | sed 's/^[[:space:]]*//')
  printf '  %-4s | %-5s | basins=%-2s | %s\n' "$name" "${dev:-$DEVICE_MODE}" "${basins:-?}" "${ans:-<no answer parsed>}"
}

say ""
say "############### NIODOO CLAIM CARD ###############"
say "prompt        : $PROMPT"
say "correct answer: $EXPECTED"
say "model         : $MODEL_FILE (bartowski, sha256 verified)"
say "device        : $DEVICE_MODE (published card = cuda; CPU/CUDA results are stated, not interchangeable)"
say "------------------------------------------------"
run_arm OFF "$OUT/off" --bridge-off
run_arm ON  "$OUT/on"  --bridge-influence-smoke --bridge-influence-smoke-clamp 0.03
say "------------------------------------------------"
say "Expect: ON lands '$EXPECTED' where OFF does not; basins=8 when ON, 0 when OFF."
say "raw outputs (kept, gitignored): $OUT"
say "  full transcripts: off/stdout.txt on/stdout.txt — the raw chats, not just the clean answer"
