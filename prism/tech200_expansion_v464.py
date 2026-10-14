"""PRISM v4.64 TECH150 vs TECH200 controlled Seed B expansion.
Seed A remains frozen TECH80 ATR4; this does NOT validate Seed A on 200.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v445 import UNIVERSE as TECH150
from .universe_v464 import UNIVERSE as TECH200,NEW50
from .trend_v48 import load_prices,portfolio
from .seed_b_discovery_v442 import discover
from .trend_v49 import exit_trade
from .technical_quality_v450 import features,quality
from .atr_frontier_v433 import CUTOFF
START=10_000_000.
def main():
 out=Path("results/prism-v464");out.mkdir(parents=True,exist_ok=True)
 prices={};errors=[]
 for t in TECH200:
  try:
   d=load_prices(t,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
   if len(d)<65:raise ValueError("Insufficient genuine price history")
   prices[t]=d
  except Exception as e:errors.append({"ticker":t,"error":str(e)})
 if len(prices)<100:raise RuntimeError("Insufficient TECH200 price coverage")
 q=load_prices("QQQ","2021-01-01").loc[lambda x:x.index<=CUTOFF].Close
 seed=pd.read_csv("results/prism-v436/daily_equity.csv")
 seed=seed[seed.variant=="ATR4_CONTROL"].copy()
 seed.date=pd.to_datetime(seed.date).dt.normalize()
 aa=seed.set_index("date").equity.sort_index()
 if aa.empty or aa.index.duplicated().any():raise RuntimeError("Seed A curve invalid")
 candidates=[];unresolved=[]
 for t,d in prices.items():
  for s in discover(d,q,t):
   if s["detector"]!="VCP_BREAKOUT":continue
   try:z=exit_trade(d,s["date"],"CONFIRM30_1")
   except (ValueError,KeyError):z=None
   if z is None:
    unresolved.append({"ticker":t,"signal_date":str(s["date"].date())});continue
   e,x,ep,xp,reason=z
   f=features(d,d.index[e])
   if f is None:
    unresolved.append({"ticker":t,"signal_date":str(s["date"].date()),"reason":"MISSING_FEATURE"});continue
   candidates.append({"ticker":t,"signal_date":str(s["date"].date()),
    "entry_date":str(d.index[e].date()),"exit_date":str(d.index[x].date()),
    "entry_price":float(ep),"exit_price":float(xp),"reason":reason,
    "original_edge":float(s["edge"]),"quality":float(quality(f)),
    "is_new50":t in NEW50})
 df=pd.DataFrame(candidates)
 if df.empty:raise RuntimeError("No valid breakout candidates")
 df.to_csv(out/"candidate_exits.csv",index=False)
 pd.DataFrame(unresolved).to_csv(out/"unresolved.csv",index=False)
 (out/"coverage_errors.json").write_text(json.dumps(errors,indent=2))
 report={"version":"v4.64","status":"HISTORICAL TECH200 EXPANSION NOT OOS",
 "coverage":{"requested":200,"available":len(prices),"new50_available":len(set(NEW50)&set(prices)),
 "excluded":errors,"unresolved":len(unresolved)},"results":{},
 "limitations":[
 "Seed A remains frozen TECH80 ATR4; not a TECH200 Seed A rerun.",
 "Only Seed B VCP Breakout with MA30 exit and v4.50 Quality Rank is expanded.",
 "TECH150 and TECH200 use same downloaded price snapshot in this run.",
 "Quality score is uncalibrated; 200-stock universe was selected retrospectively.",
 "No historical delisted membership or independently untouched holdout.",
 "Unresolved exits are excluded; report their count.",
 "Daily close MDD excludes intraday risk; terminal marks may be synthetic."]}
 ledger_all=[];curve_all=[]
 for label,allowed in (("TECH150",set(TECH150)),("TECH200",set(TECH200))):
  z=df[df.ticker.isin(allowed)].copy()
  for mode in ("BASE","QUALITY_RANK"):
   z["rank_edge"]=z.original_edge if mode=="BASE" else z.quality
   z=z.sort_values(["rank_edge","ticker"],ascending=[False,True]).drop_duplicates(["ticker","entry_date"])
   rows=[{"ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,
    "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),
    "edge":float(r.rank_edge),"reason":r.reason} for r in z.itertuples(index=False)]
   name=label+"_"+mode
   perf,ledger,curve=portfolio(rows,prices,name,rotation=False)
   b=pd.DataFrame(curve)
   b.date=pd.to_datetime(b.date).dt.normalize()
   bb=b.set_index("date").equity.sort_index()
   cal=aa.index.union(bb.index).sort_values()
   xa=aa.reindex(cal);xb=bb.reindex(cal)
   xa.loc[cal<aa.index.min()]=START;xb.loc[cal<bb.index.min()]=START
   if xa.isna().any() or xb.isna().any():raise RuntimeError("Equity calendar mismatch")
   total=xa+xb;peak=np.maximum.accumulate(np.r_[2*START,total.to_numpy()])[1:]
   report["results"][name]={"seed_b":perf,"combined_return":float(total.iloc[-1]/(2*START)-1),
    "combined_mdd":float(np.min(total.to_numpy()/peak-1)),"candidates":len(rows),
    "new50_executed":sum(r["ticker"] in NEW50 for r in ledger)}
   for r in ledger:r["experiment"]=name
   for r in curve:r["experiment"]=name
   ledger_all+=ledger;curve_all+=curve
 pd.DataFrame(ledger_all).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curve_all).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['seed_b']['return']:.2%}</td><td>{v['combined_return']:.2%}</td><td>{v['combined_mdd']:.2%}</td><td>{v['new50_executed']}</td></tr>" for k,v in report["results"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.64</title><style>body{background:#0c1728;color:#eaf5ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.64 TECH200</h1><p>Seed A frozen TECH80 ATR4; Seed B Breakout expanded; historical research only.</p><table><tr><th>Strategy</th><th>Seed B</th><th>Dual</th><th>MDD</th><th>New50 trades</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","variants":list(report["results"])}))
if __name__=="__main__":main()
