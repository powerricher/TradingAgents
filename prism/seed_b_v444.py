"""PRISM v4.44: BCH winner stress, entry/holding/exit regime attribution,
and fixed-priority breakout/BCH selection audit. Historical research only.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v413 import UNIVERSE
from .trend_v48 import load_prices,portfolio
from .regime_adaptive_v436 import regimes
from .atr_frontier_v433 import CUTOFF
SEED=10_000_000.
def canonical(df,score_col="edge"):
 return [{"ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,
 "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),
 "edge":float(getattr(r,score_col)),"reason":r.reason} for r in df.itertuples(index=False)]
def attribution(rows,ledger,market,calendar):
 candidates={(r["ticker"],r["entry_date"]):r for r in rows}
 z=pd.DataFrame(ledger).sort_values("exit_date").copy()
 z["prev_capital"]=z.capital_after.shift().fillna(SEED)
 z["pnl"]=z.capital_after-z.prev_capital
 result=[]
 for r in z.itertuples(index=False):
  key=(r.ticker,r.entry_date)
  if key not in candidates:raise RuntimeError("Unmatched executed trade "+str(key))
  entry=pd.Timestamp(r.entry_date);exit_=pd.Timestamp(r.exit_date)
  start=market.get(entry,"UNKNOWN");finish=market.get(exit_,"UNKNOWN")
  holding=market.loc[(market.index>=entry)&(market.index<exit_)]
  result.append({"ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,
   "pnl_krw":float(r.pnl),"entry_regime":start,"exit_regime":finish,
   "holding_bull_sessions":int((holding=="BULL").sum()),
   "holding_bear_sessions":int((holding=="BEAR").sum()),
   "holding_sideways_sessions":int((holding=="SIDEWAYS").sum())})
 return pd.DataFrame(result)
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--v442",default="results/prism-v442")
 ap.add_argument("--v443",default="results/prism-v443")
 ap.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv")
 ap.add_argument("--out",default="results/prism-v444")
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 base=pd.read_csv(Path(a.v442)/"candidate_exits.csv")
 bch=pd.read_csv(Path(a.v443)/"bch_confirmed_candidates.csv")
 br=base[base.detector=="VCP_BREAKOUT"].copy()
 if br.empty or bch.empty:raise RuntimeError("Missing comparison candidates")
 prices={t:load_prices(t,"2021-01-01").loc[lambda x:x.index<=CUTOFF] for t in UNIVERSE}
 market=regimes(min(pd.to_datetime(br.entry_date).min(),pd.to_datetime(bch.entry_date).min()))
 seed=pd.read_csv(a.seed_a)
 seed=seed[seed.variant=="ATR4_CONTROL"].copy()
 seed.date=pd.to_datetime(seed.date).dt.normalize()
 aa=seed.set_index("date").equity.sort_index()
 if aa.empty:raise RuntimeError("Missing frozen Seed A")
 variants={
  "BREAKOUT":canonical(br),
  "BCH_CONFIRMED":canonical(bch),
  # Fixed, predeclared tie-breaker, not trained on future realized returns.
  "BCH_FIRST":canonical(bch.assign(edge=100+bch.edge))+canonical(br),
  "BREAKOUT_FIRST":canonical(br.assign(edge=100+br.edge))+canonical(bch)
 }
 report={"version":"v4.44","status":"HISTORICAL DIAGNOSTIC / NOT INDEPENDENT OOS",
 "variants":{},"bch_winner_stress":{},"limitations":[
 "BCH winner exclusions are chosen ex post and not a tradable rule.",
 "Entry/exit market regimes use prior-session QQQ data, not ex-post calendar year labels.",
 "Holding-regime session counts are exposures, NOT attributed mark-to-market P&L.",
 "BCH_FIRST and BREAKOUT_FIRST use arbitrary predeclared rank offsets; cross-detector scores are not calibrated.",
 "The two discovery universes overlap and duplicate ticker-entry candidates may appear; portfolio sorts deterministically.",
 "All variants use historical TECH80, the same MA30 exit and no leverage or cross-seed transfers.",
 "Signals near end of sample excluded upstream; terminal Seed A valuation remains synthetic.",
 "Repeated historical tuning and winner inspection create multiple-testing bias; no independent OOS.",
 "Daily close MDD excludes intraday drawdowns."]}
 all_ledgers=[];all_curves=[];all_attr=[];combined=[]
 for name,rows in variants.items():
  perf,ledger,curve=portfolio(rows,prices,name,rotation=False)
  if not ledger:raise RuntimeError("No executions "+name)
  attr=attribution(rows,ledger,market,None)
  attr["variant"]=name;all_attr.append(attr)
  b=pd.DataFrame(curve)
  b.date=pd.to_datetime(b.date).dt.normalize()
  bb=b.set_index("date").equity.sort_index()
  cal=aa.index.union(bb.index).sort_values()
  xa=aa.reindex(cal);xb=bb.reindex(cal)
  xa.loc[cal<aa.index.min()]=SEED
  xb.loc[cal<bb.index.min()]=SEED
  if xa.isna().any() or xb.isna().any():raise RuntimeError("Internal valuation gap "+name)
  total=xa+xb
  peak=np.maximum.accumulate(np.r_[2*SEED,total.to_numpy()])[1:]
  report["variants"][name]={"seed_b":perf,"combined_return":float(total.iloc[-1]/(2*SEED)-1),
   "combined_mdd":float(np.min(total.to_numpy()/peak-1)),
   "entry_regime_pnl_krw":{str(k):float(g.pnl_krw.sum()) for k,g in attr.groupby("entry_regime")},
   "exit_regime_pnl_krw":{str(k):float(g.pnl_krw.sum()) for k,g in attr.groupby("exit_regime")},
   "executed_tickers":int(attr.ticker.nunique())}
  for r in ledger:r["variant"]=name
  for r in curve:r["variant"]=name
  all_ledgers+=ledger;all_curves+=curve
  combined.extend({"date":str(day.date()),"variant":name,"seed_a_equity":float(x),
   "seed_b_equity":float(y),"total_equity":float(x+y)} for day,x,y in zip(cal,xa,xb))
  if name=="BCH_CONFIRMED":
   wins=attr[attr.pnl_krw>0].sort_values("pnl_krw",ascending=False)
   top=wins.head(3)[["ticker","entry_date","pnl_krw"]].to_dict("records")
   report["bch_winner_stress"]["top3"]=top
   for n in (1,2,3):
    exclude={(x["ticker"],x["entry_date"]) for x in top[:n]}
    filtered=[r for r in rows if (r["ticker"],r["entry_date"]) not in exclude]
    res,_,_=portfolio(filtered,prices,"BCH_EX_TOP"+str(n),rotation=False)
    report["bch_winner_stress"]["exclude_top"+str(n)]=res
 pd.concat(all_attr,ignore_index=True).to_csv(out/"regime_trade_attribution.csv",index=False)
 pd.DataFrame(all_ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(all_curves).to_csv(out/"daily_equity.csv",index=False)
 pd.DataFrame(combined).to_csv(out/"combined_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 rows="".join(f"<tr><td>{html.escape(k)}</td><td>{v['seed_b']['return']:.2%}</td><td>{v['combined_return']:.2%}</td><td>{v['combined_mdd']:.2%}</td></tr>" for k,v in report["variants"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.44</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;padding:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.44 · BCH Robustness &amp; Regime Audit</h1><p>Historical research only; do not treat winner deletion as a forecast.</p><table><tr><th>Variant</th><th>Seed B return</th><th>Combined return</th><th>Combined MDD</th></tr>"+rows+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","variants":{k:v["combined_return"] for k,v in report["variants"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
