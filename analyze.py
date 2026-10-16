"""Summaries per run directory: soak in 2-minute bins, curve per locked clock. Each request is
matched to the 100 ms GPU samples taken while its tokens streamed.

    python3 analyze.py RUN_DIR [RUN_DIR ...]
"""
import json,csv,bisect,datetime,statistics as s,sys
from analyze_lib import load,during
for d in sys.argv[1:]:
    G,R=load(d); ts=[g[0] for g in G]
    print('==',d)
    if 'curve' in d:
        ph={}
        for r in R:
            if r['phase'].startswith('curve-') and 'warm' not in r['phase']: ph.setdefault(r['phase'],[]).append(r)
        for p,rs in ph.items():
            ss=[x for r in rs for x in during(G,ts,r)]
            print(p, 'tok/s',round(s.median(r['decode_tps'] for r in rs),1),'temp',min(x[1] for x in ss),max(x[1] for x in ss),'sm median',s.median(x[2] for x in ss),'min',min(x[2] for x in ss),'power med',round(s.median(x[3] for x in ss)),'powercap% ',round(100*sum(1 for x in ss if x[4]&4)/len(ss)))
        continue
    rs=[r for r in R if r['phase']=='soak']; t0=rs[0]['token_times'][0]
    bins={}
    for r in rs:
        m=int((r['token_times'][0]-t0)//120)
        bins.setdefault(m,[]).append(r)
    for m,b in sorted(bins.items()):
        ss=[x for r in b for x in during(G,ts,r)]
        print(f"{m*2:3d}-{m*2+2} min tok/s {s.median(r['decode_tps'] for r in b):6.1f} temp {min(x[1] for x in ss)}-{max(x[1] for x in ss)} sm med {s.median(x[2] for x in ss):.0f} min {min(x[2] for x in ss)} power {s.median(x[3] for x in ss):.0f}W powercap {100*sum(1 for x in ss if x[4]&4)/len(ss):.0f}%")
