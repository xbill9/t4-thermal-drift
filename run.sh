#!/bin/bash
# The first run, 2026-10-15/16, on the Tesla T4 of the GCE VM that hosts gemma4-dev's T4 rigs.
# Waits for gemma4-dev's trimmed T4 jobs to finish (they hold the card), then:
#   1. soak, unlocked                  4. order bias, six conditions
#   2. soak, SM clock locked 1200 MHz  5. replicates: soak unlocked x2, locked x1
#   3. clock curve 1590..1110 MHz
# The decisive comparison (1 against 2) and one of each kind run first, replicates last.
# Start detached: setsid nohup ./run.sh > run.nohup 2>&1 < /dev/null &
cd "$(dirname "$0")"
OUT=runs/2026-10-15-first-t4
mkdir -p $OUT
LOG=$OUT/run.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
job() { local name=$1; shift; say "START $name"; python3 thermal.py "$@" > $OUT/$name.out 2>&1; say "END $name rc=$?"; }
trap 'sudo nvidia-smi -rgc >/dev/null 2>&1' EXIT

TRIM=$HOME/gemma4-dev/gpu-vllm-t4-2b-w4a16/benchmarks/runs/2026-10-15-overnight-t4/trimmed.log
say "waiting for gemma4-dev trimmed jobs"
until grep -q 'TRIMMED DONE' $TRIM && ! pgrep -f 'trimmed2.sh' >/dev/null; do sleep 30; done
if curl -fs -o /dev/null --max-time 3 http://127.0.0.1:8000/health; then say "port 8000 is serving; abort"; exit 1; fi
[ -n "$(awk 'NR>1' /proc/swaps)" ] || sudo swapon /opt1/swapfile
make -s -C $HOME/gemma4-dev/gpu-vllm-t4-2b-w4a16 verify-patch | grep -q Patched || { say "Turing clamp not confirmed; abort"; exit 1; }
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,persistence_mode,serial --format=csv
  PYTHONUSERBASE=/opt1/pyuser /usr/bin/python3.13 -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)" 2>/dev/null | tail -1
  sha256sum $HOME/gemma4-dev/gpu-vllm-t4-2b-w4a16/tpu.env
  git -C $HOME/gemma4-dev rev-parse HEAD; } > $OUT/host.txt
sudo nvidia-smi -lgc 1200,1200 >> $OUT/host.txt 2>&1 && sudo nvidia-smi -rgc >> $OUT/host.txt 2>&1 || { say "cannot lock clocks; abort"; exit 1; }

job soak-unlocked-1 soak $OUT/soak-unlocked-1
job soak-locked1200-1 soak $OUT/soak-locked1200-1 --lock 1200
job curve curve $OUT/curve
job order-nocool-nonefirst order $OUT/order-nocool-nonefirst --first none
job order-nocool-qatfirst order $OUT/order-nocool-qatfirst --first qat-n3
job order-cool50-nonefirst order $OUT/order-cool50-nonefirst --first none --cool 50
job order-cool50-qatfirst order $OUT/order-cool50-qatfirst --first qat-n3 --cool 50
job order-lock1200-nonefirst order $OUT/order-lock1200-nonefirst --first none --lock 1200
job order-lock1200-qatfirst order $OUT/order-lock1200-qatfirst --first qat-n3 --lock 1200
job soak-unlocked-2 soak $OUT/soak-unlocked-2
job soak-locked1200-2 soak $OUT/soak-locked1200-2 --lock 1200
job soak-unlocked-3 soak $OUT/soak-unlocked-3
say "RUN DONE"
