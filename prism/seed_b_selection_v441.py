"""PRISM v4.41: Seed B signal-time RS/VCP challengers; Seed A ATR4 frozen.
Research-only fixed-form hypotheses, NOT fitted alpha models.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices,portfolio
from .regime_adaptive_v436 import regimes
from .atr_frontier_v433 import CUTOFF
START=10_000_000.
def zscore(x):
 s=x.std(ddof=0)
 return (x-x.mean())/s if np.isfinite(s) and s>1e-12 else x*0.
def main():
 p=argparse.ArgumentParser();p.add_argument("--scored",default="results/prism-v420/scored_signals.csv")
 p.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv")
 p.add_argument("--out",default="results/prism-v441");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored)
 for col in ("date","entry_date","exit_date"):df[col]=pd.to_datetime(df[col]).dt.normalize()
 if df.empty or df.rank_VCP_ALL.isna().any():raise RuntimeError("Missing frozen signals")
 prices={t:load_prices(t,g.date.min()).loc[lambda x:x.index<=CUTOFF] for t,g in df.groupby("ticker")}
 q=load_prices("QQQ",df.date.min()).loc[lambda x:x.index<=CUTOFF].Close
 market=regimes(df.entry_date.min())
 df["qqq_ret20"]=np.nan;df["rs_qqq20"]=np.nan
 df["breakout20"]=np.nan;df["vol_ratio"]=np.nan
 for ticker,g in df.groupby("ticker"):
  d=prices[ticker];c=d.Close;v=d.Volume
  f=pd.DataFrame(index=d.index)
  f["rs_qqq20"]=c.pct_change(20)-q.pct_change(20).reindex(d.index)
  f["breakout20"]=c/c.shift(1).rolling(20,min_periods=20).max()-1
  f["vol_ratio"]=v/v.shift(1).rolling(20,min_periods=20).mean()
  for ix,r in g.iterrows():
   if r.date not in f.index:raise RuntimeError("Missing signal date "+ticker)
   for col in ("rs_qqq20","breakout20","vol_ratio"):
    df.loc[ix,col]=f.at[r.date,col]
   df.loc[ix,"qqq_ret20"]=q.pct_change(20).get(r.date,np.nan)
 if df[["rs_qqq20","breakout20","vol_ratio"]].isna().any().any():raise RuntimeError("Missing signal-time RS/VCP data")
 df["entry_regime"]=df.entry_date.map(market)
 if df.entry_regime.isna().any() or (df.entry_regime=="UNKNOWN").any():raise RuntimeError("Unknown entry regime")
 # Date-local z-scores use only the day's simultaneously eligible candidates.
 # No future outcomes or hindsight thresholds enter any ranking.
 for col in ("rs_qqq20","breakout20","vol_ratio","rank_VCP_ALL"):
  df["z_"+col]=df.groupby("entry_date")[col].transform(zscore).fillna(0.)
 df["score_CONTROL"]=df.rank_VCP_ALL
 df["score_RS"]=df.z_rs_qqq20
 df["score_BREAKOUT"]=.65*df.z_breakout20+.35*df.z_vol_ratio
 df["score_COMBINED"]=.5*df.z_rs_qqq20+.35*df.z_breakout20+.15*df.z_vol_ratio
 modes={"B_CONTROL":"score_CONTROL","B_RS":"score_RS","B_BREAKOUT":"score_BREAKOUT",
  "B_COMBINED":"score_COMBINED","B_COMBINED_REGIME":"score_COMBINED"}
 base=pd.read_csv(a.seed_a)
 base=base[base.variant=="ATR4_CONTROL"][["date","equity"]].copy()
 base.date=pd.to_datetime(base.date).dt.normalize()
 if base.empty or base.date.duplicated().any():raise RuntimeError("Missing Seed A")
 base=base.set_index("date").equity.sort_index()
 report={"version":"v4.41","status":"HISTORICAL HYPOTHESIS / NOT INDEPENDENT OOS",
 "seed_a":"ATR4_CONTROL (frozen)","seed_b":{},"limitations":[
 "Same 80-stock frozen VCP eligibility; this does NOT discover new candidates or expand to 100 stocks.",
 "Seed B RS, breakout and volume scores are predeclared linear rules, not trained prediction models.",
 "Combined regime challenger permits BULL/SIDEWAYS but excludes new BEAR entries; it is a testable hypothesis, not a validated rule.",
 "Same-day ranking z-scores use only contemporaneous eligible candidates.",
 "Seed B exits are original frozen MA30 exits; Seed A ATR4 unchanged.",
 "No market-neutrality is claimed; shorting, hedging and sector-relative strength are not implemented.",
 "Terminal mark-to-market is a synthetic valuation in the existing portfolio engine.",
 "Historical period repeatedly researched; survivorship bias and multiple comparisons remain.",
 "No independent forward validation; daily-close MDD excludes intraday lows."]}
 all_ledger=[];all_curves=[];daily=[];scored=df.copy()
 for mode,col in modes.items():
  candidates=df if mode!="B_COMBINED_REGIME" else df[df.entry_regime!="BEAR"]
  rows=[{"ticker":r.ticker,"mode":mode,"entry_date":str(r.entry_date.date()),"exit_date":str(r.exit_date.date()),
   "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),"edge":float(getattr(r,col)),"reason":r.reason}
   for r in candidates.itertuples(index=False)]
  perf,ledger,curve=portfolio(rows,prices,mode,rotation=False)
  b=pd.DataFrame(curve).assign(date=lambda x:pd.to_datetime(x.date).dt.normalize()).set_index("date").equity
  aa,bb=base.align(b,join="outer")
  if aa.isna().any() or bb.isna().any():raise RuntimeError("Misaligned account days "+mode)
  total=aa+bb
  peak=np.maximum.accumulate(np.r_[2*START,total.to_numpy()])[1:]
  combined={"return":float(total.iloc[-1]/(2*START)-1),
   "daily_close_mdd":float(np.min(total.to_numpy()/peak-1)),
   "final_krw":float(total.iloc[-1])}
  report["seed_b"][mode]={"seed_b":perf,"combined":combined,"eligible_signals":len(rows)}
  for x in ledger:x["variant"]=mode
  for x in curve:x["variant"]=mode
  all_ledger+=ledger;all_curves+=curve
  daily.extend({"date":str(day.date()),"variant":mode,"seed_a_equity":float(x),"seed_b_equity":float(y),
   "combined_equity":float(x+y)} for day,x,y in zip(total.index,aa,bb))
 scored.to_csv(out/"candidate_scores.csv",index=False)
 pd.DataFrame(all_ledger).to_csv(out/"seed_b_ledger.csv",index=False)
 pd.DataFrame(all_curves).to_csv(out/"seed_b_daily_equity.csv",index=False)
 pd.DataFrame(daily).to_csv(out/"combined_daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['seed_b']['return']:.2%}</td><td>{v['combined']['return']:.2%}</td><td>{v['combined']['daily_close_mdd']:.2%}</td></tr>" for k,v in report["seed_b"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.41</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.41 · Seed B RS and Breakout</h1><p>Seed A ATR4 frozen; Seed B independent 10M KRW; historical hypothesis only.</p><table><tr><th>Seed B</th><th>B return</th><th>Combined return</th><th>Combined MDD</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","seed_b":{k:v["combined"]["return"] for k,v in report["seed_b"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
