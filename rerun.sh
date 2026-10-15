#!/bin/bash
# After run.sh: redo order-lock1200-nonefirst, whose first attempt locked the clock before cooling.
cd "$(dirname "$0")"
OUT=runs/2026-10-15-first-t4
LOG=$OUT/run.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
trap 'sudo nvidia-smi -rgc >/dev/null 2>&1' EXIT
while pgrep -x run.sh >/dev/null || pgrep -f '^/bin/bash \./run\.sh$' >/dev/null; do sleep 30; done
say "START order-lock1200-nonefirst (rerun)"
python3 thermal.py order $OUT/order-lock1200-nonefirst --first none --lock 1200 > $OUT/order-lock1200-nonefirst.out 2>&1
say "END order-lock1200-nonefirst rc=$?"
