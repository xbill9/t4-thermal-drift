# t4-thermal-drift

**Question:** does a Tesla T4's temperature change LLM decode speed, and is the SM clock how?

On 2026-10-15, in a Gemma 4 drafter benchmark on the T4 of a GCE `n1-standard-2` VM
(`xbill9/gemma4-dev`, `gpu-vllm-t4-2b-w4a16/benchmarks/runs/2026-10-15-drafter-sweep-t4`), the
same no-drafter server decoded 113.8 tok/s at the start of a run (card 52–63 °C, SM clock median
1,515 MHz) and 104.8 tok/s at the end (71–81 °C, 1,478 MHz, dipping to 1,230). Runs that cooled
the card to 50 °C before each server agreed within 1%. Across that run's no-drafter requests,
decode rate against temperature correlates at r = −0.94. The clock was read once per request,
after it finished, so those readings cannot show the mechanism.

**Hypothesis** (from NVIDIA's own documentation, `LITERATURE.md`): a hotter chip leaks more
current, so it draws more power at the same clock, and the T4's 70 W cap pulls the clock down
well before the ~85 °C thermal throttle. Under load this should show as SW Power Cap.

## Method (`thermal.py`)

Each run logs temperature, SM and memory clocks, power, clock event reasons, utilization and
P-state every 100 ms, and the wall time of every streamed token, so each stretch of generation
can be matched to the card's state at that moment.

- **Heat soak**: one greedy prompt back to back for 20 minutes from a card at 50 °C, unlocked
  and with the SM clock locked (`nvidia-smi -lgc`) at 1,200 MHz. If the clock is the mechanism,
  the locked soak decodes at a flat rate while the temperature rises as in the unlocked one.
- **Clock curve**: decode rate at locked clocks from 1,590 to 1,110 MHz on a cool card.
- **Order bias**: speculative decoding's measured speed-up with the baseline first or second,
  with no cool-down, with a 50 °C cool-down and with the clock locked.

Serving setup: vLLM 0.29.0 with the flags of `gemma4-dev/gpu-vllm-t4-2b-w4a16/tpu.env`
(`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`, fp16, Turing Triton clamp), drafter
`google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant` at 3 tokens.

## Status

`run.sh` queued 2026-10-15, after gemma4-dev's T4 jobs. No results yet. One card in one host's
airflow: another T4 may behave differently.
