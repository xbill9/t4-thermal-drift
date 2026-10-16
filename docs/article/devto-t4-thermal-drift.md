---
title: "Don't Assume Your Cloud T4 Runs at Spec: Gemma 4 Decode Falls 7% as the Card Warms"
published: false
description: "The same Tesla T4 on the same Google Cloud VM decoded Gemma 4 E2B at 115 tokens/s at 45 °C and 107 at 77 °C. The card sits at its 70 W limit, a hotter chip draws more power at the same clock, so the clock drops. No thermal throttle fires, and no spec sheet mentions it."
tags: nvidia, googlecloud, llm, performance
cover_image: https://raw.githubusercontent.com/xbill9/t4-thermal-drift/main/docs/article/devto-cover.c2f75cb3.jpg
---

This article provides a step by step guide to measuring how a cloud GPU's temperature changes LLM decode speed, on the NVIDIA Tesla T4 of a Google Cloud VM serving Gemma 4 E2B with vLLM. Every log, script and number is committed.

https://github.com/xbill9/t4-thermal-drift

**The same T4, on the same VM, serving the same model the same prompt, decoded at 115.3 tokens per second at 45 °C and 106.8 at its steady 77 °C: 7.4% slower, for as long as the load lasts. The card runs at its 70 W power limit the whole time. A hotter chip draws more power at the same clock, so the limit pulls the clock down from about 1,457 MHz to about 1,267. The thermal throttle never fires.**

---

#### What is this project trying to Do?

A rented GPU is sold by name and by the hour. A T4 is a T4, and the spec sheet gives it one number: 8.1 TFLOPS of FP32, 65 TFLOPS of FP16. Benchmarks on it are run for a minute or two and quoted as the card's speed.

This project asks whether that number holds when the card is hot. It logs the card's temperature, clock, power and throttle state every 100 ms while it decodes, and lines every request up with the card's state at the moment its tokens were generated.

The answer shapes three things a customer cares about: what speed a T4 delivers under sustained load, how much a short benchmark overstates it, and whether an A/B test run back to back is fair.

---

#### At This Point You Should Have…

- A Google Cloud VM with one NVIDIA T4 and root access, so `nvidia-smi -lgc` can lock the clock
- vLLM serving a model on the card. This one runs vLLM 0.29.0 with `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`, Gemma 4 E2B in 4-bit weights, fp16
- A clone: `git clone https://github.com/xbill9/t4-thermal-drift`

---

#### The Hardware

```text
name, driver_version, memory.total [MiB], power.limit [W], persistence_mode
Tesla T4, 615.71.09, 15360 MiB, 70.00 W, Enabled
```

```text
    Max Clocks
        Graphics                                       : 1590 MHz
        SM                                             : 1590 MHz
```

```text
        Current Power Limit                            : 70.00 W
        Default Power Limit                            : 70.00 W
        Max Power Limit                                : 70.00 W
```

| | |
| :--- | :--- |
| VM | `n1-standard-2`, `us-west2-b` |
| GPU | Tesla T4, passive cooling, 70 W limit, 1,590 MHz maximum clock |
| Server | vLLM 0.29.0, one request at a time, greedy, 256 output tokens |

The T4 has no fan. It depends on the host's airflow, which the cloud provider controls. Google runs the card at the highest power limit it allows.

---

#### Step 1 — Log the Card Every 100 ms

`thermal.py` runs `nvidia-smi` in the background for the whole of each run, writing eight fields every 100 ms:

```text
timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active,utilization.gpu,pstate
```

Each request records the wall time of every streamed token. `analyze.py` takes the samples that fall between a request's first and last token, so each decode rate is matched to the temperature, clock and power while it was generated.

`clocks_event_reasons.active` is the field that says why the clock is below maximum. The two values that matter here are `0x4`, the software power cap, and `0x20` / `0x40`, the software and hardware thermal slowdowns.

---

#### Step 2 — Run a Heat Soak From a Cool Card

The soak waits until the card is below 50 °C, starts the server, and sends one prompt back to back, greedy, 256 tokens each:

```shell
python3 thermal.py soak runs/2026-10-15-first-t4/soak-unlocked-5min-1 --minutes 5
```

```text
soak-start {'temp_c': 41}
10 114.0 (54, 1515)
20 112.3 (63, 1545)
30 110.7 (68, 1470)
40 108.6 (74, 1440)
50 105.7 (79, 1500)
60 106.8 (76, 1425)
70 107.4 (76, 1455)
80 106.4 (77, 1500)
90 106.7 (77, 1500)
100 107.1 (76, 1440)
110 106.9 (76, 1305)
120 106.9 (77, 1455)
soak-end {'requests': 125, 'temp_c': 77}
```

Every tenth request prints its decode rate, then the card's temperature and SM clock read once, after the request. The speed falls as the temperature climbs, and both settle within about a minute. A single clock reading jumps by 200 MHz from one moment to the next, so Step 3 uses the 100 ms samples taken while each request decoded.

A 20-minute soak, run the day before, shows the same shape from a 57 °C start.

---

#### Step 3 — Read Decode Speed by Temperature

```shell
python3 bands.py runs/2026-10-15-first-t4
```

The 5-minute soak, every request grouped by the card's mean temperature while it decoded:

| Card temperature | Requests | Decode tok/s | SM clock MHz | Power W |
| ---: | ---: | ---: | ---: | ---: |
| 45–49 °C | 4 | 115.3 | 1,457 | 69.3 |
| 55–59 °C | 5 | 113.5 | 1,404 | 68.9 |
| 65–69 °C | 10 | 111.1 | 1,350 | 68.2 |
| 70–74 °C | 8 | 109.4 | 1,316 | 67.8 |
| 75–79 °C | 81 | 106.9 | 1,267 | 66.9 |

The 20-minute soak:

| Card temperature | Requests | Decode tok/s | SM clock MHz |
| ---: | ---: | ---: | ---: |
| 55–59 °C | 1 | 112.7 | 1,390 |
| 65–69 °C | 6 | 110.8 | 1,346 |
| 70–74 °C | 7 | 108.0 | 1,296 |
| 75–79 °C | 462 | 105.6 | 1,241 |

```text
== soak-unlocked-1: requests 486, below 95 tok/s 1
after 2 min: n 437, median 105.6 tok/s, trend +0.03 tok/s per 10 min, temp 76-78 C, SM median 1241 MHz
r(decode, temperature) -0.86  r(decode, SM clock) 0.80
== soak-unlocked-5min-1: requests 125, below 95 tok/s 0
after 2 min: n 73, median 106.8 tok/s, trend +1.01 tok/s per 10 min, temp 76-80 C, SM median 1266 MHz
r(decode, temperature) -0.97  r(decode, SM clock) 0.99
```

One request in the 20-minute soak ran at 54.1 tokens per second, 36 seconds in, and is left out of the bands and correlations.

Cool to hot costs 7.4% in one soak and 6.3% in the other, which started warmer. Once the card holds at 76–78 °C the rate stays flat for 18 minutes, so the loss follows temperature, and time on its own adds nothing. Per request, decode tracks the SM clock with a correlation of 0.99.

---

#### Step 4 — Find What Pulls the Clock Down

```text
== clock event reasons seen, all runs (0x1 idle, 0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal)
soak-locked1200-1                             samples  13352 reasons ['0x1', '0x4'] max temp 81 C
soak-unlocked-1                               samples  13384 reasons ['0x1', '0x4'] max temp 81 C
soak-unlocked-5min-1                          samples   3509 reasons ['0x1', '0x4'] max temp 80 C
```

Across every run, 154,815 samples between them, the card reported two reasons only: idle, and the software power cap. The thermal slowdowns never fired, and the hottest reading was 83 °C. During the unlocked soaks the power cap was active in every temperature band, from 45 °C up.

So the card is at its 70 W limit from the first request. At 45 °C those 70 W buy 1,457 MHz. At 77 °C they buy 1,267.

---

#### Step 5 — Measure the Leakage at a Held Clock

Hold the clock fixed and the only thing left to change the power is the chip's temperature:

```shell
python3 thermal.py soak runs/2026-10-15-first-t4/soak-locked1200-1 --lock 1200
```

```text
power at a held 1200 MHz, not capped (6485 samples of 100 ms):
  66 C n=   16 median 62.0 W
  68 C n=   19 median 63.4 W
  70 C n=   22 median 64.0 W
  72 C n=   26 median 64.3 W
  74 C n=   23 median 65.1 W
  76 C n=  261 median 65.7 W
  78 C n=  271 median 66.0 W
  slope 0.26 W per C
```

Same clock, same work, 4.0 W more at 78 °C than at 66 °C. A hotter chip leaks more current. NVIDIA's TensorRT documentation names the effect and names this card: *"higher GPU temperature also leads to more leakage current in the circuits, which increases the power consumed by the GPU at a specific clock frequency. Therefore, for GPUs more likely to be power throttled like NVIDIA T4, poor cooling can lead to lower stabilized clock frequency with power throttling and, thus, worse performance."*

---

#### Step 6 — Lock the Clock and See What It Costs

The same soak with the SM clock locked at 1,200 MHz:

| Card temperature | Requests | Decode tok/s | SM clock MHz | Power-capped |
| ---: | ---: | ---: | ---: | ---: |
| 65–69 °C | 5 | 104.8 | 1,200 | 0% |
| 70–74 °C | 7 | 104.5 | 1,200 | 1% |
| 75–79 °C | 460 | 104.1 | 1,198 | 29% |

Locking shrinks the drift from 7% to 0.7%. It does so by giving up speed: 104.1 tokens per second hot and locked, against 106.8 hot and unlocked and 115.3 cool. Even at 1,200 MHz the card reaches its power limit in 29% of samples once it is hot.

NVIDIA's own guidance says the same of locked clocks: *"the average performance will not be as good as when the clock is floating."*

---

#### Step 7 — Check What a Back-to-Back A/B Test Measures

The project began with a speculative-decoding comparison: Gemma 4 E2B alone against Gemma 4 E2B with its 4-bit drafter, three draft tokens, each on its own server, six prompts, three requests each. Run order and cooling were varied:

```shell
python3 order.py runs/2026-10-15-first-t4
```

```text
order-cool50-nonefirst                     none>qat-n3  none: 111.1 tok/s n=18 56->71 C  qat-n3: 144.3 tok/s n=18 56->70 C  speed-up 1.298x
order-cool50-qatfirst                      qat-n3>none  qat-n3: 144.6 tok/s n=18 56->71 C  none: 111.5 tok/s n=18 55->71 C  speed-up 1.297x
order-nocool-nonefirst                     none>qat-n3  none: 111.3 tok/s n=18 56->71 C  qat-n3: 142.3 tok/s n=18 64->76 C  speed-up 1.279x
order-nocool-qatfirst                      qat-n3>none  qat-n3: 145.1 tok/s n=18 56->65 C  none: 108.5 tok/s n=18 63->77 C  speed-up 1.337x
```

With the card cooled to 50 °C before each server, the speed-up is 1.297x or 1.298x whichever runs first. Without the cool-down, the second server starts on a warm card and runs 1.9% to 2.5% slower than the same server run first, and the same speed-up reads anywhere from 1.279x to 1.337x.

---

#### What Do Google and NVIDIA Promise?

NVIDIA's T4 datasheet lists, in its specifications table:

```text
Single-Precision       8.1 TFLOPS
Mixed-Precision        65 TFLOPS
(FP16/FP32)
INT8                   130 TOPS
Thermal Solution       Passive
```

No clock, no condition and no footnote is attached. Google's Compute Engine GPU page repeats 65, 130 and 260 TOPS for the T4, also without conditions. The Compute Engine SLA defines downtime as *"loss of external connectivity or persistent disk access"*: it covers uptime and says nothing on speed.

The published figure is a maximum-clock figure. FP32 throughput is 2,560 CUDA cores times two operations per clock times the clock, so 8.1 TFLOPS needs 1,582 MHz, the card's 1,590 MHz maximum. The same arithmetic at the clocks measured here:

| SM clock | FP32, by arithmetic | Share of the spec clock |
| ---: | ---: | ---: |
| 1,590 MHz, the maximum | 8.14 TFLOPS | 100% |
| 1,430 MHz, cool, 44 °C | 7.32 TFLOPS | 89.9% |
| 1,241 MHz, hot, 77 °C | 6.35 TFLOPS | 78.1% |
| 1,200 MHz, locked | 6.14 TFLOPS | 75.5% |

These rows are arithmetic from the clock, not measured TFLOPS. Decode fell less than the clock did, 7.4% against 13.0%, because decoding one token also reads every weight from memory, and the memory clock holds.

---

#### 🔎 Tip: Log the Clock Event Reasons With Every Benchmark

`nvidia-smi --query-gpu=clocks_event_reasons.active,clocks.sm,temperature.gpu,power.draw -lms 100` costs nothing and turns "it seemed slower" into a reason code. A benchmark that quotes tokens per second without the clock it ran at leaves out the variable that moved the number by 7% here.

---

#### 🔎 Tip: Cool to the Same Temperature Before Every Server

Stopping the server lets a T4 in this host fall from 71 °C to 50 °C in under four minutes. Loaded with a model and idle, the same card sat at 72–76 °C for 30 minutes and never got there. Stop the server, wait for the temperature, then start the next one.

---

#### Compare and Contrast

| | Cool, unlocked | Hot, unlocked | Hot, locked 1,200 MHz |
| --- | --- | --- | --- |
| Card temperature | 45–49 °C | 75–79 °C | 75–79 °C |
| Decode | 🥇 115.3 tok/s | 🥈 106.8 tok/s | 🥉 104.1 tok/s |
| SM clock | 1,457 MHz | 1,267 MHz | 1,198 MHz |
| How long it lasts | about a minute | as long as the load | as long as the load |
| Drift cool to hot | — | 7.4% | 0.7% |

---

#### So, Which One?

Don't assume. Under sustained load this T4 delivers the hot number, 106.8 tokens per second on Gemma 4 E2B, and the cool number exists for the first minute. Quote the hot number, log the clock beside it, and cool every arm of an A/B test to the same temperature before it starts. Lock the clock when consistency matters more than speed, and count what it costs: 2.5% below hot and unlocked here.

The customer pays the same hourly rate for 115 or 107 tokens per second, and the cooling that decides between them belongs to the provider.

---

#### Summary

The goal of this article was to find whether a cloud T4's temperature changes LLM decode speed, and why. The key to the solution was logging the card's clock, power, temperature and throttle reason every 100 ms and matching each request to the samples taken while it decoded. The results were:

- ❌ Decode fell 7.4%, from 115.3 to 106.8 tokens per second, as the card warmed from 45 °C to its steady 77 °C, and 6.3% in a second soak that started warmer
- ⚠️ The card sits at its 70 W limit from the first request. At a held 1,200 MHz it drew 4.0 W more at 78 °C than at 66 °C, so the limit pulls the clock down as it heats
- 🟢 No thermal throttle fired in 154,815 samples, and the hottest reading was 83 °C
- ⚠️ Locking the clock at 1,200 MHz cut the drift to 0.7% and cost 2.5% against hot and unlocked
- ❌ Without a cool-down between servers, the same speculative-decoding speed-up read 1.279x to 1.337x depending on order. With one it read 1.297x and 1.298x
- ⚠️ Google and NVIDIA publish the T4's throughput with no condition attached, and it corresponds to the 1,590 MHz maximum clock

Scope: one Tesla T4 on one `n1-standard-2` VM in `us-west2-b`, driver 615.71.09, vLLM 0.29.0, Gemma 4 E2B in 4-bit weights, one request at a time, measured 2026-10-15 and 2026-10-16. Two unlocked soaks (20 and 5 minutes) and one locked soak at 1,200 MHz; each order condition ran once. Other T4 hosts, batched serving and other models were not run. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed output files.

The strategy for measuring thermal drift on a cloud T4 was validated with an incremental step by step approach.

---

#### References

- Code, logs and this article's evidence: https://github.com/xbill9/t4-thermal-drift
- NVIDIA T4 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
- NVIDIA TensorRT, Hardware/Software Environment for Performance Measurements: https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/hw-sw-environment.html
- Google Compute Engine GPUs: https://docs.cloud.google.com/compute/docs/gpus
- Compute Engine SLA: https://cloud.google.com/compute/sla
- nvidia-smi documentation: https://docs.nvidia.com/deploy/nvidia-smi/index.html
