#!/usr/bin/env bash
# Reproduce the full eight-prompt table from the whitepaper / claim card.
# Off (--bridge-off) vs On (--bridge-influence-smoke, clamp 0.03), deterministic (temp 0, seed 42).
# Run from the repo root:  ./harness/run_battery.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SHA="14e10feba0c82a55da198dcd69d137206ad22d116a809926d27fa5f2398c69c7"
BIN="${NIODOO_BIN:-$REPO/niodoo/target/release/niodoo}"
MODEL_PATH="${NIODOO_MODEL_PATH:-$REPO/model/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf}"
OUT="$(mktemp -d)"

[ -x "$BIN" ] || { echo "build the binary first: cd niodoo && cargo build --release --features niodv4_bridge (or set NIODOO_BIN)"; exit 1; }
[ -f "$MODEL_PATH" ] || { echo "model missing; run ./reproduce.sh once to download+verify it, or set NIODOO_MODEL_PATH"; exit 1; }
[ "$(sha256sum "$MODEL_PATH" | awk '{print $1}')" = "$MODEL_SHA" ] || { echo "model hash mismatch — refusing to run"; exit 1; }
cd "$REPO" || exit 1

answer(){ # stdout-file
  local a; a=$(awk '/=== TURN 0 OUTPUT ===/{f=1;next} f' "$1" | grep -E 'VISIBLE ANSWER:|WORKING ANSWER:' | tail -1 | sed 's/^[[:space:]]*//')
  [ -n "$a" ] || a=$(awk '/=== TURN 0 OUTPUT ===/{f=1;next} f' "$1" | grep -E 'REQUEST: LOCK\]' | tail -1 | sed 's/^[[:space:]]*//')
  printf '%s' "${a:-<no answer parsed>}"
}
arm(){ # outfile prompt extra...
  local of="$1" pr="$2"; shift 2
  "$BIN" --model-path "$MODEL_PATH" --model-size 8b --runtime-speed-profile eval-fast \
    --stdout-profile telemetry --telemetry-profile score --seed 42 --temperature 0.0 --max-steps 256 \
    --prompt "$pr" --telemetry-out "$of.jsonl" "$@" >"$of.txt" 2>/dev/null
}

printf '%-34s | %-9s | %-26s | %-26s\n' "prompt" "correct" "OFF" "ON"
printf -- '---------------------------------------------------------------------------------------------------------\n'
i=0
while IFS='|' read -r label correct prompt; do
  [ -z "$label" ] && continue
  i=$((i+1))
  arm "$OUT/${i}_off" "$prompt" --bridge-off
  arm "$OUT/${i}_on"  "$prompt" --bridge-influence-smoke --bridge-influence-smoke-clamp 0.03
  printf '%-34s | %-9s | %-26s | %-26s\n' "$label" "$correct" "$(answer "$OUT/${i}_off.txt")" "$(answer "$OUT/${i}_on.txt")"
done <<'PROMPTS'
bat and ball|$0.05|A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Give the final number.
strawberry r|3|How many times does the letter r appear in the word strawberry? Give the final number only.
17 x 24|408|What is 17 multiplied by 24? Give the final number only.
13 x 17|221|What is 13 multiplied by 17? Give the final number only.
23 x 18|414|What is 23 multiplied by 18? Give the final number only.
raspberry r|2|How many times does the letter r appear in the word raspberry? Give the final number only.
banana a|3|How many times does the letter a appear in the word banana? Give the final number only.
mississippi s|4|How many times does the letter s appear in the word mississippi? Give the final number only.
PROMPTS
echo
echo "raw outputs: $OUT"
