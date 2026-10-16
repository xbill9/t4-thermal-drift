"""GPU samples and requests of one run directory, and the samples taken while a request streamed."""
import json,csv,bisect,datetime
def load(d):
    G=[]
    for row in csv.reader(open(d+'/gpu.csv')):
        try:
            t=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
            G.append((t,int(row[1]),int(row[2]),float(row[4]),int(row[5],16)))
        except Exception: pass
    R=[json.loads(l) for l in open(d+'/requests.jsonl')]
    return G,R
def during(G,ts,r):
    tt=r['token_times']; return G[bisect.bisect(ts,tt[0]):bisect.bisect(ts,tt[-1])]
