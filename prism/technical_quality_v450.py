"""PRISM v4.50: signal-time entry quality audit, Seed B only.
Seed A TECH80 ATR4 frozen. Compare baseline vs fixed, predeclared quality
ranking and quality gate for TECH80/100/150. No fitted parameters.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v445 import UNIVERSE,CORE100
from .universe_v413 import UNIVERSE as CORE80
from .trend_v48 import load_prices,portfolio
from .atr_frontier_v433 import CUTOFF
INITIAL=10_000_000.
def features(d,entry):
 dt=pd.Timestamp(entry)
 if dt not in d.index:return None
 i=d.index.get_loc(dt)-1
 if i<65:return None
 c=d.Close.astype(float);h=d.High.astype(float);l=d.Low.astype(float);v=d.Volume.astype(float)
 # All features use the previous session's fully completed bar.
 cl=float(c.iloc[i]);prior_high=float(h.iloc[i-20:i].max())
 if cl<=0 or prior_high<=0:return None
 ret=c.pct_change()
 vol20=float(v.iloc[i-20:i].mean())
 vol_ratio=float(v.iloc[i]/vol20) if vol20>0 else 0.
 recent=ret.iloc[i-9:i+1].std();long=ret.iloc[i-39:i+1].std()
 compression=float(recent/long) if long>0 and np.isfinite(long) else 9.
 base_high=float(h.iloc[i-24:i+1].max())
 base_low=float(l.iloc[i-24:i+1].min())
 base_width=base_high/base_low-1 if base_low>0 else 9.
 old_low=float(l.iloc[i-19:i-9].min());new_low=float(l.iloc[i-9:i+1].min())
 defended=new_low/old_low-1 if old_low>0 else -1.
 ma20=float(c.iloc[i-19:i+1].mean())
 ma60=float(c.iloc[i-59:i+1].mean())
 return {"breakout_strength":cl/prior_high-1,"volume_ratio":vol_ratio,
  "compression":compression,"base_width":base_width,"low_defense":defended,
  "ma20_ratio":cl/ma20-1,"ma60_ratio":cl/ma60-1}
def quality(f):
 # Fixed monotonic bounded score; no future returns or fitted weights.
 clip=lambda x:float(np.clip(x,0,1))
 return 100*(.25*clip((f["breakout_strength"]+.02)/.07)
  +.20*clip((f["volume_ratio"]-.8)/1.7)
  +.20*clip((1.15-f["compression"])/.65)
  +.15*clip((.24-f["base_width"])/.20)
  +.10*clip((f["low_defense"]+.05)/.10)
  +.10*clip((f["ma20_ratio"]+.01)/.09))
def main():
 p=argparse.ArgumentParser();p.add_argument("--candidates",default="results/prism-v445/candidate_exits.csv")
 p.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv")
 p.add_argument("--out",default="results/prism-v450");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 c=pd.read_csv(a.candidates)
 if not {"ticker","entry_date","exit_date","entry_price","exit_price","edge","detector","reason"}.issubset(c):raise RuntimeError("Missing candidate columns")
 c=c[c.detector.isin(["VCP_BREAKOUT","BCH_CONFIRMED"])].copy()
 if c.empty:raise RuntimeError("No Seed B candidates")
 prices={};unavailable=[]
 for ticker in UNIVERSE:
  try:prices[ticker]=load_prices(ticker,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
  except Exception as e:unavailable.append({"ticker":ticker,"error":str(e)})
 # Only use candidates with actually loaded OHLC, never invent valuations.
 c=c[c.ticker.isin(prices)].copy()
 if c.empty:raise RuntimeError("No usable price histories")
 feats=[];missing=[]
 for r in c.itertuples(index=False):
  f=features(prices[r.ticker],r.entry_date)
  if f is None:missing.append({"ticker":r.ticker,"entry_date":r.entry_date});continue
  feats.append({**r._asdict(),**f,"quality":quality(f)})
 scored=pd.DataFrame(feats)
 if scored.empty:raise RuntimeError("No valid signal-time features")
 scored.to_csv(out/"signal_time_quality.csv",index=False)
 pd.DataFrame(missing).to_csv(out/"missing_features.csv",index=False)
 (out/"price_errors.json").write_text(json.dumps(unavailable,ensure_ascii=False,indent=2))
 seed=pd.read_csv(a.seed_a);seed=seed[seed.variant=="ATR4_CONTROL"].copy()
 seed.date=pd.to_datetime(seed.date).dt.normalize()
 aa=seed.set_index("date").equity.sort_index()
 if aa.empty or aa.index.duplicated().any():raise RuntimeError("Invalid Seed A")
 groups={"TECH80":set(CORE80),"TECH100":set(CORE100),"TECH150":set(UNIVERSE)}
 report={"version":"v4.50","status":"HISTORICAL QUALITY CHALLENGER / NOT INDEPENDENT OOS",
  "coverage":{"requested":150,"available":len(prices),"missing_features":len(missing),"unavailable":unavailable},
  "strategies":{},"limitations":[
   "All scores computed from the fully completed bar before entry open; no future outcome labels.",
   "Same historical candidate signals and MA30 exits; new entry confirmation/delayed fills are NOT yet tested.",
   "Quality weights and threshold 55 are exploratory, not trained or OOS validated.",
   "Ranking is applied only to candidates sharing an entry date; occupied seed cannot rotate.",
   "Universe retrospectively selected; unavailable prices and unresolved exits may bias results.",
   "Seed A ATR4 contains possible terminal synthetic valuation.",
   "Daily-close MDD excludes intraday troughs."]}
 ledgers=[];daily=[]
 for group,allowed in groups.items():
  subset=scored[scored.ticker.isin(allowed)]
  for detector in ("VCP_BREAKOUT","BCH_CONFIRMED","BOTH"):
   z=subset if detector=="BOTH" else subset[subset.detector==detector]
   if z.empty:continue
   for variant in ("BASE","QUALITY_RANK","QUALITY_GATE55"):
    zz=z if variant!="QUALITY_GATE55" else z[z.quality>=55]
    if zz.empty:continue
    rows=[]
    for r in zz.itertuples(index=False):
     # Deduplicate exact ticker+entry collisions after choosing strongest score.
     rows.append({"ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,
      "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),
      "edge":float(r.edge if variant=="BASE" else r.quality),
      "reason":r.reason})
    rows=list({(x["ticker"],x["entry_date"]):x for x in sorted(rows,key=lambda x:x["edge"]) }.values())
    name=f"{group}_{detector}_{variant}"
    perf,ledger,curve=portfolio(rows,prices,name,rotation=False)
    b=pd.DataFrame(curve);b.date=pd.to_datetime(b.date).dt.normalize()
    bb=b.set_index("date").equity
    calendar=aa.index.union(bb.index).sort_values()
    xa=aa.reindex(calendar);xb=bb.reindex(calendar)
    xa.loc[calendar<aa.index.min()]=INITIAL;xb.loc[calendar<bb.index.min()]=INITIAL
    if xa.isna().any() or xb.isna().any():raise RuntimeError("Internal equity gap "+name)
    total=xa+xb;peak=np.maximum.accumulate(np.r_[2*INITIAL,total.to_numpy()])[1:]
    report["strategies"][name]={"seed_b":perf,"combined_return":float(total.iloc[-1]/(2*INITIAL)-1),
     "combined_mdd":float(np.min(total.to_numpy()/peak-1)),"eligible":len(rows)}
    for x in ledger:x["experiment"]=name
    ledgers+=ledger
    daily.extend({"date":str(day.date()),"experiment":name,"seed_a":float(av),
      "seed_b":float(bv),"total":float(av+bv)} for day,av,bv in zip(calendar,xa,xb))
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(daily).to_csv(out/"combined_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['seed_b']['return']:.2%}</td><td>{v['combined_return']:.2%}</td><td>{v['combined_mdd']:.2%}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.50</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.50 · Technical Entry Quality</h1><p>Historical challenger only; Seed A frozen.</p><table><tr><th>Experiment</th><th>Seed B return</th><th>Combined</th><th>MDD</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","experiments":len(report["strategies"])}))
if __name__=="__main__":main()
