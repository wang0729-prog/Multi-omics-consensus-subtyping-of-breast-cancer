#!/usr/bin/env bash
# run_all.sh - sequential execution of the whole BRCA multi-omics pipeline.
# Every stage writes a log to results/log_<stage>.log and its raw numbers to
# results/<stage>/*.csv so that any figure can be re-drawn from the tables.
set -u
PY="${PY:-python}"   # override with: PY=/path/to/python ./scripts/run_all.sh
cd "$(dirname "$0")/.."
mkdir -p results data/raw

run () {
  name="$1"; shift
  echo "=== $name  start $(date +%H:%M:%S) ==="
  "$PY" -u "scripts/$name" "$@" > "results/log_${name%.py}.log" 2>&1
  code=$?
  echo "=== $name  exit=$code  $(date +%H:%M:%S) ==="
  if [ $code -ne 0 ]; then
    echo "--- tail of log ---"; tail -25 "results/log_${name%.py}.log"
  fi
}

run 10_cluster.py
run fig1.py                                        # pass 1 -> subtype signature
run 05_fetch_metabric.py --all-from results/fig1/subtype_signature.csv
run fig1.py                                        # pass 2 -> METABRIC panel filled
run fig2.py
run fig3.py
run fig4_5.py
run fig6.py
run 20_singlecell.py
run figS.py
echo "PIPELINE DONE $(date +%H:%M:%S)"
