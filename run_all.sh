#!/usr/bin/env bash
# Reproduce every analysis from the criticality / SD-scaling thread and store
# each script's full stdout in logs/<name>.log.
#
#   ./run_all.sh          # fast tier only (~15 min)
#   ./run_all.sh all      # everything, including the slow sweeps (~2-3 h)
#   ./run_all.sh <name>   # one script, e.g. ./run_all.sh beta_ceiling
#
# linear_theory.py is the canonical engine and is never modified; everything
# here imports it. Nothing is deleted: logs are overwritten per script only.
set -u -o pipefail

PY=/Users/k2585057/anaconda3/envs/py39/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p logs sweep_data

# verification and theory, minutes each
FAST=(
  proof_checks
  structural_icg
  figure_structural_icg
  ring_block_variance
  composition_is_trivial
  band_binning_invariance
  distance_sufficiency
  band_mechanism
  spatial_networks
)
# sweeps over m, seeds and topologies, tens of minutes each
SLOW=(
  beta_ceiling
  stationarity_test
  dimension_vs_m
  local_global
  ring_vs_disorder
  local_global_topologies
  range_composition
  hm_local_global
)

run_one () {
  local name="$1"
  if [ ! -f "$name.py" ]; then
    echo "!! $name.py not found, skipping"
    return
  fi
  local log="logs/$name.log"
  echo "== $name  ->  $log"
  {
    echo "### $name"
    echo "### started $(date -Iseconds)"
    echo "### git $(git rev-parse --short HEAD 2>/dev/null || echo 'no commit')"
    echo
  } > "$log"
  if "$PY" -W ignore -u "$name.py" >> "$log" 2>&1; then
    echo "### finished $(date -Iseconds)" >> "$log"
    echo "   ok"
  else
    echo "### FAILED $(date -Iseconds)" >> "$log"
    echo "   FAILED (see $log)"
  fi
}

case "${1:-fast}" in
  fast) for n in "${FAST[@]}"; do run_one "$n"; done ;;
  all)  for n in "${FAST[@]}" "${SLOW[@]}"; do run_one "$n"; done ;;
  slow) for n in "${SLOW[@]}"; do run_one "$n"; done ;;
  *)    run_one "$1" ;;
esac

echo
echo "logs in $HERE/logs, data in $HERE/sweep_data"
