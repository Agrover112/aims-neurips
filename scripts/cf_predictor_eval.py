#!/usr/bin/env python3
"""Score a collaborative-filtering item-difficulty predictor under the real protocol.

Evidence rule (the honest one): at budget b, every evaluation task has revealed
only the first b items of its own acquisition order. Nothing from any task's
evaluation pool is ever visible -- those are the labels being scored. Item
difficulty for a target item therefore comes only from OTHER tasks that happened
to acquire that same item.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

B=(0,1,3,7,15,31); W=dict(zip(B,(0.1,0.2,0.2,0.2,0.2,0.1)))

def sig(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--corpus",type=Path,required=True)
    ap.add_argument("--split",default="validation"); ap.add_argument("--orders",type=int,default=10)
    ap.add_argument("--json",type=Path); a=ap.parse_args()

    ev=pd.read_parquet(a.corpus/"eval_items.parquet"); ev["task_id"]=ev.task_id.astype(str)
    ev=ev[ev.split==a.split]
    od=pd.read_parquet(a.corpus/"eval_orders.parquet"); od["task_id"]=od.task_id.astype(str)
    label={(t,i):l for t,i,l in zip(ev.task_id,ev.item_index,ev.label)}
    tasks=sorted(ev.task_id.unique())
    tgt={t:g[g.pool=="evaluation"] for t,g in ev.groupby("task_id")}
    gm=float(ev.label.mean())

    rows=[]
    for seed in range(a.orders):
        o=od[od.order_seed==seed].sort_values("acquisition_rank")
        seq={t:g.item_index.to_numpy() for t,g in o.groupby("task_id")}
        for b in B:
            # global revealed pool at this budget: first b acquisitions of EVERY task
            ev_items={}        # item -> [labels from other tasks]
            for t in tasks:
                for i in seq.get(t,[])[:b]:
                    ev_items.setdefault(i,[]).append((t,label.get((t,i),np.nan)))
            for t in tasks:
                g=tgt[t]; y=g.label.to_numpy(float); items=g.item_index.to_numpy()
                own=[label.get((t,i),np.nan) for i in seq.get(t,[])[:b]]
                own=np.array([v for v in own if v==v],dtype=float)
                rate=(own.sum()+2.0)/(len(own)+4.0)           # Beta posterior = ability prior
                # CF difficulty per target item from OTHER tasks only
                diff=np.full(len(items),np.nan); nev=np.zeros(len(items))
                for k,i in enumerate(items):
                    o_=[l for (tt,l) in ev_items.get(i,[]) if tt!=t and l==l]
                    if o_: diff[k]=np.mean(o_); nev[k]=len(o_)
                # shrink CF difficulty toward the global mean by evidence count
                kk=2.0
                p_item=np.where(nev>0,(diff*nev+gm*kk)/(nev+kk),gm)
                # combine: ability (own rate) + item deviation, in logit space
                lo=lambda p: np.log(np.clip(p,1e-3,1-1e-3)/(1-np.clip(p,1e-3,1-1e-3)))
                pred=sig(lo(rate)+ (lo(p_item)-lo(gm)))
                beta=np.full(len(items),rate)
                rows.append({"order":seed,"budget":b,"task_id":t,
                    "benchmark_id":g.benchmark_id.iloc[0],
                    "brier_cf":float(np.mean((pred-y)**2)),
                    "brier_beta":float(np.mean((beta-y)**2)),
                    "cov":float((nev>0).mean()),"mean_ev":float(nev.mean())})
    r=pd.DataFrame(rows)
    r["w"]=r.budget.map(W)
    alc=r.groupby(["task_id","order"]).apply(lambda d:pd.Series({
        "cf":(d.w*d.brier_cf).sum(),"beta":(d.w*d.brier_beta).sum()}),include_groups=False
        ).groupby("task_id").mean()
    d=(alc.cf-alc.beta).to_numpy()
    rng=np.random.default_rng(0); bs=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
    out={"split":a.split,"tasks":int(len(alc)),"orders":a.orders,
      "alc_cf":round(float(alc.cf.mean()),6),"alc_beta":round(float(alc.beta.mean()),6),
      "delta":round(float(d.mean()),6),"ci":[round(float(np.quantile(bs,.025)),6),round(float(np.quantile(bs,.975)),6)],
      "tasks_better":int((d<0).sum()),
      "coverage_by_budget":r.groupby("budget")[["cov","mean_ev"]].mean().round(4).reset_index().to_dict("records"),
      "brier_by_budget":r.groupby("budget")[["brier_cf","brier_beta"]].mean().round(6).reset_index().to_dict("records")}
    print(json.dumps(out,indent=2))
    if a.json: a.json.write_text(json.dumps(out,indent=2)+"\n")

if __name__=="__main__": main()
