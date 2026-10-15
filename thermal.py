"""Does the T4's temperature slow LLM decode, and is the SM clock the mechanism?

    python3 thermal.py soak  OUT [--minutes 20] [--lock MHZ] [--start-below C]
    python3 thermal.py curve OUT [--clocks 1590,1500,...] [--requests 10] [--start-below C]
    python3 thermal.py order OUT --first none|qat-n3 [--cool C] [--lock MHZ]

Every subcommand starts its own vLLM server (serve.py, the gemma4-dev rig's serving flags) and,
for its whole length, a sampler: `nvidia-smi -lms 100` writing temperature, SM and memory clocks,
power, the clock event reasons bitmask, utilization and P-state every 100 ms to OUT/gpu.csv.
Each request records the wall time of every streamed token (OUT/requests.jsonl), so any stretch
of generation can be matched to the clock and temperature at that moment.

  soak   no drafter; one fixed greedy prompt (256 tokens) sent back to back for --minutes, starting
         from a card at or below --start-below. With --lock the SM clock is pinned
         (sudo nvidia-smi -lgc MHZ,MHZ) for the whole soak. If the clock is the mechanism, a locked
         soak decodes at a flat rate while the temperature climbs as in an unlocked one.
  curve  no drafter; for each clock, cool, lock, two warm-up requests, then --requests timed
         requests. Decode rate against locked clock on a cool card; the unlocked soak's hot points
         should land on this curve.
  order  speculative decoding's measured speed-up as a function of run order: two servers back to
         back, --first then the other, six prompts x 3 greedy each. --cool C cools to C before each
         server (the 2026-10-15 drafter sweep's method); without it the second server starts on
         whatever card the first left. --lock pins the clock for both.

The clock lock is always released (sudo nvidia-smi -rgc) on exit, including on error.
"""

import argparse, hashlib, json, os, signal, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve import load_env, start, stop, stream_chat  # noqa: E402

FIELDS = "timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active,utilization.gpu,pstate"
SOAK_PROMPT = "Write a short story about a lighthouse keeper who finds a message in a bottle."
PROMPTS = {
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "chat": "I just moved to a new city and feel a bit lonely. Any advice?",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a refrigerator works to a ten-year-old.",
    "facts": "List the planets of the solar system with one fact about each.",
    "summary": "Summarize the causes of the French Revolution in one paragraph.",
}


def gpu_now():
    out = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split(",")
    )
    return int(out[0]), int(out[1])


def cool(below, limit_s=1800):
    t0 = time.time()
    while gpu_now()[0] > below:
        if time.time() - t0 > limit_s:
            print(f"cool: still {gpu_now()[0]} C after {limit_s} s, going on", flush=True)
            return
        time.sleep(5)


def lock(mhz):
    subprocess.run(["sudo", "nvidia-smi", "-lgc", f"{mhz},{mhz}"], check=True, capture_output=True)


def unlock():
    subprocess.run(["sudo", "nvidia-smi", "-rgc"], capture_output=True)


class Sampler:
    def __init__(self, out):
        self.f = open(os.path.join(out, "gpu.csv"), "a")
        self.p = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=self.f,
            stderr=subprocess.STDOUT,
        )

    def close(self):
        self.p.terminate()
        self.p.wait()
        self.f.close()


class Run:
    def __init__(self, out, kind, args):
        os.makedirs(out, exist_ok=True)
        self.out, self.env = out, load_env()
        self.url, self.model = self.env["ENDPOINT"], self.env["MODEL_NAME"]
        self.req = open(os.path.join(out, "requests.jsonl"), "a")
        self.ev = open(os.path.join(out, "events.jsonl"), "a")
        self.event("begin", kind=kind, args=vars(args), model=self.model)

    def event(self, what, **kw):
        self.ev.write(json.dumps(dict(t=time.time(), event=what, **kw)) + "\n")
        self.ev.flush()
        print(what, kw, flush=True)

    def request(self, phase, prompt, name, temperature=0, max_tokens=256):
        text, n, t0, times = stream_chat(
            self.url,
            self.model,
            {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature},
        )
        dec = (n - 1) / (times[-1] - times[0]) if len(times) > 1 else None
        rec = dict(
            phase=phase,
            prompt=name,
            t0=t0,
            n=n,
            decode_tps=dec,
            token_times=times,
            sha=hashlib.sha256(text.encode()).hexdigest()[:16],
        )
        self.req.write(json.dumps(rec) + "\n")
        self.req.flush()
        return rec

    def serve(self, tag, draft=None, n=None):
        self.event("server-start", tag=tag, draft=draft, n=n)
        srv = start(self.env, os.path.join(self.out, f"server-{tag}.log"), draft, n)
        self.event("server-healthy", tag=tag)
        self.request("warmup", "Hello", "hello", max_tokens=16)
        return srv


def soak(a):
    r = Run(a.out, "soak", a)
    s = Sampler(a.out)
    try:
        cool(a.start_below)
        if a.lock:
            lock(a.lock)
            r.event("lock", mhz=a.lock)
        srv = r.serve("soak")
        try:
            r.event("soak-start", temp_c=gpu_now()[0])
            end = time.time() + a.minutes * 60
            i = 0
            while time.time() < end:
                rec = r.request("soak", SOAK_PROMPT, "story")
                i += 1
                if i % 10 == 0:
                    print(i, round(rec["decode_tps"], 1), gpu_now(), flush=True)
            r.event("soak-end", requests=i, temp_c=gpu_now()[0])
        finally:
            stop(*srv)
    finally:
        unlock()
        r.event("unlock")
        s.close()


def curve(a):
    r = Run(a.out, "curve", a)
    s = Sampler(a.out)
    try:
        srv = r.serve("curve")
        try:
            for mhz in [int(x) for x in a.clocks.split(",")]:
                unlock()
                cool(a.start_below)
                lock(mhz)
                r.event("lock", mhz=mhz, temp_c=gpu_now()[0])
                for _ in range(2):
                    r.request(f"curve-warm-{mhz}", SOAK_PROMPT, "story")
                for _ in range(a.requests):
                    r.request(f"curve-{mhz}", SOAK_PROMPT, "story")
                r.event("point-end", mhz=mhz, temp_c=gpu_now()[0])
        finally:
            stop(*srv)
    finally:
        unlock()
        r.event("unlock")
        s.close()


def order(a):
    r = Run(a.out, "order", a)
    s = Sampler(a.out)
    env = r.env
    arms = ["none", "qat-n3"] if a.first == "none" else ["qat-n3", "none"]
    try:
        cool(a.start_below)  # every condition starts from the same card; a locked clock idles too hot to cool
        if a.lock:
            lock(a.lock)
            r.event("lock", mhz=a.lock)
        for arm in arms:
            if a.cool and arm != arms[0]:
                cool(a.cool)
            draft, n = (env["SPECULATIVE_MODEL"], 3) if arm == "qat-n3" else (None, None)
            srv = r.serve(arm, draft, n)
            try:
                r.event("arm-start", arm=arm, temp_c=gpu_now()[0])
                for name, prompt in PROMPTS.items():
                    for _ in range(3):
                        r.request(arm, prompt, name)
                r.event("arm-end", arm=arm, temp_c=gpu_now()[0])
            finally:
                stop(*srv)
    finally:
        unlock()
        r.event("unlock")
        s.close()


def main():
    # SIGTERM as SystemExit, so the finally blocks release the clock lock and stop the sampler
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("soak")
    p.add_argument("out")
    p.add_argument("--minutes", type=float, default=20)
    p.add_argument("--lock", type=int, default=0)
    p.add_argument("--start-below", type=int, default=50)
    p = sp.add_parser("curve")
    p.add_argument("out")
    p.add_argument("--clocks", default="1590,1500,1395,1305,1230,1110")
    p.add_argument("--requests", type=int, default=10)
    p.add_argument("--start-below", type=int, default=50)
    p = sp.add_parser("order")
    p.add_argument("out")
    p.add_argument("--first", choices=["none", "qat-n3"], required=True)
    p.add_argument("--cool", type=int, default=0)
    p.add_argument("--lock", type=int, default=0)
    p.add_argument("--start-below", type=int, default=50)
    a = ap.parse_args()
    {"soak": soak, "curve": curve, "order": order}[a.cmd](a)


if __name__ == "__main__":
    main()
