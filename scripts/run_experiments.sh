#!/usr/bin/env bash
# Run the experiments over all three models; results land in data/results/.
#   scripts/run_experiments.sh                 # everything (several hours on an M3 Pro)
#   scripts/run_experiments.sh sparse methods  # only these
set -u
cd "$(dirname "$0")/.."
MODELS="qwen3vl gemma2 siglip2"
ALL="sim_real sparse geometry templates time_arrow residual eyeball eyeball_score delta methods agreement speed"

run() {
  local name=$1
  case $name in
    residual) args="qwen3vl gemma2" ;;
    delta) args="qwen3vl zugspitze" ;;
    eyeball_score|agreement|speed) args="" ;;
    *) args=$MODELS ;;
  esac
  echo "=== $name $args"
  # shellcheck disable=SC2086
  uv run python -m "experiments.$name" $args 2>&1 \
    | grep -v -i -E "^W1|warning|loading weights|token_id|redirects" | tail -45
}

for name in ${*:-$ALL}; do run "$name"; done
echo SUITE-DONE
