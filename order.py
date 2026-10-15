"""Order-bias conditions: per server, geometric mean over the six prompts of the mean decode tok/s
of each prompt's three greedy requests, the card temperature at its start and end, and the
drafter's speed-up when both servers finished all 18 requests.

    python3 order.py RUN_DIR
"""
import glob, json, math, os, sys


def gm(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


for d in sorted(glob.glob(os.path.join(sys.argv[1], "order-*/"))):
    reqs = [json.loads(l) for l in open(d + "requests.jsonl")]
    ev = [json.loads(l) for l in open(d + "events.jsonl")]
    starts = {e["arm"]: e["temp_c"] for e in ev if e["event"] == "arm-start"}
    ends = {e["arm"]: e["temp_c"] for e in ev if e["event"] == "arm-end"}
    arms = {}
    for r in reqs:
        if r["phase"] in ("none", "qat-n3") and r["decode_tps"]:
            arms.setdefault(r["phase"], {}).setdefault(r["prompt"], []).append(r["decode_tps"])
    out = {a: (gm([sum(v) / len(v) for v in p.values()]), sum(len(v) for v in p.values())) for a, p in arms.items()}
    order = [e["arm"] for e in ev if e["event"] == "arm-start"]
    line = f"{os.path.basename(d.rstrip('/')):42s} {'>'.join(order):12s} "
    for a in order:
        if a in out:
            line += f"{a}: {out[a][0]:.1f} tok/s n={out[a][1]} {starts.get(a)}->{ends.get(a)} C  "
    if all(a in out and out[a][1] == 18 for a in ("none", "qat-n3")):
        line += f"speed-up {out['qat-n3'][0] / out['none'][0]:.3f}x"
    print(line)
