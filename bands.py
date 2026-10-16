"""Figures for the article, from the soak runs: decode by temperature band, correlations, the
trend once the card is hot, power against temperature at a held clock, and the clock event
reasons seen in every run.

    python3 bands.py RUN_ROOT
"""
import glob, os, statistics as s, sys
from analyze_lib import load, during

root = sys.argv[1]
OUTLIER = 95  # tok/s; one request in soak-unlocked-1 ran at 54.1


def rows(d):
    G, R = load(d); ts = [g[0] for g in G]
    rs = [r for r in R if r["phase"] == "soak"]; t0 = rs[0]["token_times"][0]
    out = []
    for r in rs:
        ss = during(G, ts, r)
        out.append(dict(t=r["token_times"][0] - t0, tps=r["decode_tps"], temp=s.mean(x[1] for x in ss),
                        sm=s.mean(x[2] for x in ss), power=s.median(x[3] for x in ss),
                        capped=sum(1 for x in ss if x[4] & 4) / len(ss)))
    return G, out


for name in ("soak-unlocked-1", "soak-unlocked-5min-1", "soak-locked1200-1"):
    G, rr = rows(os.path.join(root, name))
    ok = [r for r in rr if r["tps"] > OUTLIER]
    print(f"== {name}: requests {len(rr)}, below {OUTLIER} tok/s {len(rr) - len(ok)}")
    f = rr[0]
    print(f"first request: {f['tps']:.1f} tok/s at {f['temp']:.0f} C, SM {f['sm']:.0f} MHz")
    print("band       n   median tok/s  median SM MHz  median power W  power-capped %")
    for lo in range(40, 85, 5):
        b = [r for r in ok if lo <= r["temp"] < lo + 5]
        if b:
            print(f"{lo}-{lo + 4} C {len(b):5d} {s.median(r['tps'] for r in b):12.1f} {s.median(r['sm'] for r in b):14.0f}"
                  f" {s.median(r['power'] for r in b):15.1f} {100 * s.mean(r['capped'] for r in b):15.0f}")
    hot = [r for r in ok if r["t"] >= 120]
    print(f"after 2 min: n {len(hot)}, median {s.median(r['tps'] for r in hot):.1f} tok/s, "
          f"trend {s.linear_regression([r['t'] for r in hot], [r['tps'] for r in hot]).slope * 600:+.2f} tok/s per 10 min, "
          f"temp {min(r['temp'] for r in hot):.0f}-{max(r['temp'] for r in hot):.0f} C, "
          f"SM median {s.median(r['sm'] for r in hot):.0f} MHz")
    print(f"r(decode, temperature) {s.correlation([r['tps'] for r in ok], [r['temp'] for r in ok]):.2f}  "
          f"r(decode, SM clock) {s.correlation([r['tps'] for r in ok], [r['sm'] for r in ok]):.2f}")
    if name.startswith("soak-locked"):
        ts = [g[0] for g in G]
        R = load(os.path.join(root, name))[1]
        ss = [x for r in R if r["phase"] == "soak" for x in during(G, ts, r) if x[2] == 1200 and not x[4] & 4]
        print(f"power at a held 1200 MHz, not capped ({len(ss)} samples of 100 ms):")
        for t in sorted({x[1] for x in ss}):
            p = [x[3] for x in ss if x[1] == t]
            if len(p) >= 5:
                print(f"  {t} C n={len(p):5d} median {s.median(p):.1f} W")
        print(f"  slope {s.linear_regression([x[1] for x in ss], [x[3] for x in ss]).slope:.2f} W per C")
        allss = [x for r in R if r["phase"] == "soak" for x in during(G, ts, r)]
        print(f"  power-capped share of samples during requests: {100 * sum(1 for x in allss if x[4] & 4) / len(allss):.0f}%")

print("== clock event reasons seen, all runs (0x1 idle, 0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal)")
for d in sorted(glob.glob(os.path.join(root, "*/"))):
    G, _ = load(d.rstrip("/"))
    if not G:
        continue
    seen = sorted({1 << b for g in G for b in range(16) if g[4] >> b & 1})
    print(f"{os.path.basename(d.rstrip('/')):45s} samples {len(G):6d} reasons {[hex(x) for x in seen]} max temp {max(g[1] for g in G)} C")
