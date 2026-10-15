# Literature and documentation, 2026-10-15

Sources opened in this session. Quotes are verbatim; anything not opened is marked unverified.

## The mechanism is documented by NVIDIA, for the T4 by name

**TensorRT, "Hardware/Software Environment for Performance Measurements"**
(https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/hw-sw-environment.html, re-read
2026-10-15):

- "higher GPU temperature also leads to more leakage current in the circuits"; "poor cooling can
  lead to lower stabilized clock frequency with power throttling and, thus, worse performance"
- "Power throttling happens by design and is a natural phenomenon when the GPU clock is not
  locked", "especially for GPUs with lower power limits, such as NVIDIA T4 and NVIDIA A2 GPUs"
- "Thermal throttling happens when the GPU temperature reaches a predefined threshold of around
  85 degrees Celsius"
- locking with `sudo nvidia-smi -lgc <freq>`: "it should stay at that frequency unless power or
  thermal throttling occurs"; it "allows more deterministic tactic selections and consistent
  performance measurements", but "the average performance will not be as good as when the clock
  is floating"

So the expected chain is: hotter chip → more leakage → more power at the same clock → the 70 W
cap pulls the clock down, before any thermal throttle. The expected flag is **SW Power Cap during
load**. The one check in the drafter run was taken at an idle moment and showed no reason active,
which this does not contradict. `thermal.py` logs `clocks_event_reasons.active` and `power.draw`
every 100 ms under load, which tests it directly.

**T4 product brief PB-09256-001_v05** (opened by the search agent): 70 W maximum power limit,
boost 1,590 MHz, passive cooling, "requires system air flow to operate the card within its
thermal limits". No airflow figure; the system design guide it points to was not found.

**nvidia-smi documentation** (https://docs.nvidia.com/deploy/nvidia-smi/index.html, opened by the
search agent): the section is now "Clocks Event Reasons"; SW Power Cap means the "SW Power Scaling
algorithm is reducing the clocks below requested clocks because the GPU is consuming too much
power"; HW Slowdown cuts clocks "by a factor of 2 or more", which does not describe a gradual
1,515 → 1,230 MHz drift. `-lgc` and `-rgc`: "Supported on Volta+. Requires root."

## Related measurement work

- Sinha et al., "Not All GPUs Are Created Equal", arXiv 2208.11035 (SC22): performance variation
  across nominally identical GPUs, attributed to manufacturing variability and power management.
  Abstract only; its thermal analysis unverified.
- StarSD, arXiv 2601.21622: a speculative-decoding paper that locks the graphics clock with
  nvidia-smi as a control. Thermal state and run order not discussed.
- arXiv 2603.23640 (edge LLM benchmarking): thermal equilibration and warm-up before measuring.
- MLPerf Inference rules: a 600-second minimum run; no rule on clocks, thermal state or run order.
- Unverified: ClusterBench (arXiv 2608.10956) and PRISM (arXiv 2510.15596) on fleet variation;
  Hoefler and Belli, SC15.

## What looks new

The mechanism is NVIDIA's own documentation. Not found anywhere: a measurement of how thermal
drift biases an A/B speed-up when arms run in sequence (baseline on a cool card, speculative
decoding on a hot one), and the size of the drift on a T4 serving an LLM (about 8% decode from
~55 to ~80 °C, well below the 85 °C throttle point). Those are the contribution, framed as a
consequence of documented behaviour rather than a new hardware effect.
