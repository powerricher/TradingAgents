"""PRISM v4.57 — full Seed A/B strategy consolidation and audit.
No strategy promotion from repeatedly researched historical data.
"""
import argparse,json,html,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v445 import UNIVERSE,CORE100
from .universe_v413 import UNIVERSE as CORE80
from .trend_v48 import load_prices,portfolio
from .atr_frontier_v433 import CUTOFF
INITIAL=10_000_000.
REGISTRY={
 "A_VCP_ATR4":{"status":"CORE_RESEARCH_BASELINE","reason":"Existing long-trend Seed A reference; not live validated"},
 "B_BREAKOUT":{"status":"CORE_CHALLENGER","reason":"Strong historical returns; concentrated in few winners"},
 "B_BREAKOUT_QUALITY":{"status":"CHALLENGER_UNVALIDATED","reason":"Ranking improvement observed, weak score calibration"},
 "B_BCH":{"status":"SECONDARY_RESEARCH","reason":"Universe expansion reduced robustness"},
 "B_QUALITY_GATE55":{"status":"REJECTED_FOR_PROMOTION","reason":"Material winner exclusion and deteriorating backtest"},
 "B_RS_DIVERGENCE":{"status":"REJECTED_FOR_PROMOTION","reason":"Poor historical return and drawdown"},
 "B_ALL_DISCOVERY":{"status":"REJECTED_FOR_PROMOTION","reason":"Mixed selection underperformed"},
 "B_ASCENDING_TRIANGLE":{"status":"ARCHIVED_RESEARCH","reason":"Insufficient incremental portfolio performance"},
 "B_DESCENDING_TRIANGLE":{"status":"ARCHIVED_RESEARCH","reason":"Winner concentration and small sample"},
 "B_FALLING_WEDGE":{"status":"ARCHIVED_RESEARCH","reason":"High drawdown; confirmation variants underperformed"},
 "B_WEDGE_CONFIRMATIONS":{"status":"REJECTED_FOR_PROMOTION","reason":"3/5-day and structural confirmations did not improve performance"},
 "CATALYST_FUNDAMENTAL":{"status":"DATA_BLOCKED","reason":"Point-in-time evidence coverage insufficient"}
}
def metrics(curve):
 vals=np.asarray(curve,dtype=float)
 peak=np.maximum.accumulate(np.r_[2*INITIAL,vals])[1:]
 return {"return":float(vals[-1]/(2*INITIAL)-1),"mdd":float(np.min(vals/peak-1)),
  "final_krw":float(vals[-1])}
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv")
 ap.add_argument("--candidates",default="results/prism-v445/candidate_exits.csv")
 ap.add_argument("--quality",default="results/prism-v450/signal_time_quality.csv")
 ap.add_argument("--out",default="results/prism-v457")
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 seed=pd.read_csv(a.seed_a)
 seed=seed[seed.variant=="ATR4_CONTROL"].copy()
 seed.date=pd.to_datetime(seed.date).dt.normalize()
 aa=seed.set_index("date").equity.sort_index()
 if aa.empty or aa.index.duplicated().any():raise RuntimeError("Invalid frozen Seed A")
 source=pd.read_csv(a.candidates)
 scored=pd.read_csv(a.quality)
 needed={"ticker","entry_date","exit_date","entry_price","exit_price","edge","detector","reason"}
 if not needed.issubset(source):raise RuntimeError("Missing frozen candidates")
 if not {"ticker","entry_date","detector","quality"}.issubset(scored):raise RuntimeError("Missing quality scores")
 # Use exact frozen candidate identity; never substitute missing score with zero.
 q=scored[["ticker","entry_date","detector","quality"]].drop_duplicates(["ticker","entry_date","detector"])
 merged=source.merge(q,on=["ticker","entry_date","detector"],how="left",validate="many_to_one")
 if merged.quality.isna().any():raise RuntimeError("Missing quality values in frozen candidates")
 prices={};errors=[]
 for ticker in sorted(source.ticker.unique()):
  try:prices[ticker]=load_prices(ticker,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
  except Exception as e:errors.append({"ticker":ticker,"error":str(e)})
 if errors:raise RuntimeError("Missing historical OHLC: "+str(errors[:10]))
 groups={"TECH80":set(CORE80),"TECH100":set(CORE100),"TECH150":set(UNIVERSE)}
 report={"version":"v4.57","status":"HISTORICAL CONSOLIDATION / NOT INDEPENDENT OOS",
  "seed_a":{"source":a.seed_a,"variant":"ATR4_CONTROL","equity_start":float(aa.iloc[0]),
    "equity_end":float(aa.iloc[-1]),"date_start":str(aa.index.min().date()),
    "date_end":str(aa.index.max().date())},
  "strategy_registry":REGISTRY,"coverage":{},"results":{},
  "limitations":[
   "Seed A is the frozen TECH80 ATR4 curve, not independently revalidated from raw trades.",
   "Only Seed B Breakout and BCH variants are recomputed; archived strategies are catalogued, not rerun.",
   "Existing candidate exit dates and fills are frozen from earlier runs; not a fresh signal generation.",
   "Quality score was tuned on repeatedly observed historical data and is not an independent predictor.",
   "TECH150 is a retrospective list; missing delisted symbols and survivorship bias remain.",
   "Candidate list excludes unresolved exits and final 220 calendar days; this can bias returns.",
   "Portfolio uses one full-capital slot per Seed and historical 10bps modeled cost.",
   "Daily-close MDD omits intraday losses; terminal valuations may be synthetic.",
   "No strategy is promoted to production from this audit."]}
 ledger_all=[];curve_all=[];collisions=[]
 for label,allowed in groups.items():
  sub=merged[merged.ticker.isin(allowed)].copy()
  report["coverage"][label]={"requested":len(allowed),"candidate_tickers":int(sub.ticker.nunique()),"candidate_rows":len(sub)}
  for detector in ("VCP_BREAKOUT","BCH_CONFIRMED"):
   x=sub[sub.detector==detector].copy()
   for variant in ("BASE","QUALITY_RANK"):
    z=x.copy()
    if z.empty:continue
    z["rank_edge"]=z.edge if variant=="BASE" else z.quality
    z=z.sort_values(["rank_edge","ticker"],ascending=[False,True]).drop_duplicates(["ticker","entry_date"])
    rows=[{"ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,
      "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),
      "edge":float(r.rank_edge),"reason":r.reason} for r in z.itertuples(index=False)]
    name=f"{label}_{detector}_{variant}"
    perf,ledger,curve=portfolio(rows,prices,name,rotation=False)
    bb=pd.DataFrame(curve).assign(date=lambda x:pd.to_datetime(x.date).dt.normalize()).set_index("date").equity.sort_index()
    cal=aa.index.union(bb.index).sort_values()
    xa=aa.reindex(cal);xb=bb.reindex(cal)
    xa.loc[cal<aa.index.min()]=INITIAL;xb.loc[cal<bb.index.min()]=INITIAL
    if xa.isna().any() or xb.isna().any():raise RuntimeError("Internal equity gap "+name)
    combined=metrics(xa.to_numpy()+xb.to_numpy())
    pnl=[];prev=INITIAL
    for r in ledger:
     gain=float(r["capital_after"])-prev;prev=float(r["capital_after"])
     pnl.append(gain)
    positive=sorted([v for v in pnl if v>0],reverse=True)
    gross=sum(positive)
    report["results"][name]={"seed_b":perf,"combined":combined,"candidate_count":len(rows),
      "executed":len(ledger),"top1_gross_profit_share":float(sum(positive[:1])/gross) if gross else None,
      "top3_gross_profit_share":float(sum(positive[:3])/gross) if gross else None}
    for r in ledger:r["experiment"]=name
    for r in curve:r["experiment"]=name
    ledger_all+=ledger;curve_all+=curve
    if variant=="QUALITY_RANK":
     selected={(r["ticker"],r["entry_date"]) for r in ledger}
     for day,g in z.groupby("entry_date"):
      if len(g)<2:continue
      ranked=g.sort_values(["rank_edge","ticker"],ascending=[False,True]).head(3)
      for rank,r in enumerate(ranked.itertuples(index=False),1):
       collisions.append({"experiment":name,"entry_date":day,"rank":rank,
        "ticker":r.ticker,"score":float(r.rank_edge),
        "executed":(r.ticker,str(day)[:10]) in selected})
 pd.DataFrame(ledger_all).to_csv(out/"trade_ledger.csv",index=False)
 pd.DataFrame(curve_all).to_csv(out/"seed_b_equity.csv",index=False)
 pd.DataFrame(collisions).to_csv(out/"same_day_opportunities.csv",index=False)
 (out/"strategy_registry.json").write_text(json.dumps(REGISTRY,ensure_ascii=False,indent=2))
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['seed_b']['return']:.2%}</td><td>{v['combined']['return']:.2%}</td><td>{v['combined']['mdd']:.2%}</td><td>{v['executed']}</td></tr>" for k,v in report["results"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.57</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.57 · Strategy Consolidation</h1><p>Frozen Seed A; Seed B baseline and quality rank replay. Historical, not independent OOS.</p><table><tr><th>Strategy</th><th>Seed B return</th><th>Combined return</th><th>Combined MDD</th><th>Trades</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","strategies":len(report["results"]),"seed_a":"frozen"}))
if __name__=="__main__":main()
