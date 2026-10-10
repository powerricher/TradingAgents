"""PRISM v4.43: breakout winner-deletion stress and BCH setup->breakout.
Historical diagnostics, no independent OOS. Fixed TECH80 and frozen Seed A.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v413 import UNIVERSE
from .trend_v48 import load_prices,portfolio
from .trend_v49 import exit_trade
from .atr_frontier_v433 import CUTOFF
from .regime_adaptive_v436 import regimes
START=10_000_000.
def bch_confirm(d):
 c=d.Close.astype(float);h=d.High.astype(float);l=d.Low.astype(float);v=d.Volume.astype(float)
 high25=h.rolling(25,min_periods=25).max()
 low25=l.rolling(25,min_periods=25).min()
 prior_low=l.shift(10).rolling(10,min_periods=10).min()
 recent_low=l.rolling(10,min_periods=10).min()
 ret=c.pct_change()
 setup=((c/high25>=.97)&(c<=high25)&(high25/low25-1<=.18)&
   (recent_low>=prior_low*.97)&(c>=c.rolling(10,min_periods=10).mean())&
   (ret.rolling(10,min_periods=10).std()<=ret.rolling(30,min_periods=30).std()))
 # Setup must be observed in the PRIOR 1-10 trading sessions; breakout
 # compares with prior 25-session high, and uses only current close/volume.
 recent_setup=setup.shift(1).rolling(10,min_periods=1).max().fillna(0).astype(bool)
 prior_high=h.shift(1).rolling(25,min_periods=25).max()
 prior_vol=v.shift(1).rolling(20,min_periods=20).mean()
 signal=recent_setup&(c>prior_high)&(v>1.2*prior_vol)&(c>c.shift(1))
 valid=signal.fillna(False)&(d.index>=pd.Timestamp("2022-01-01"))&(d.index<=CUTOFF-pd.Timedelta(days=220))
 return [(dt,float(c.loc[dt]/prior_high.loc[dt]-1)) for dt in d.index[valid]]

def make_rows(raw,prices):
 rows=[];issues=[]
 for r in raw:
  d=prices[r["ticker"]]
  try:z=exit_trade(d,pd.Timestamp(r["date"]),"CONFIRM30_1")
  except (KeyError,ValueError):z=None
  if z is None:
   issues.append(r);continue
  e,x,entry,px,reason=z
  rows.append({"ticker":r["ticker"],"entry_date":str(d.index[e].date()),
   "exit_date":str(d.index[x].date()),"entry_price":float(entry),"exit_price":float(px),
   "edge":float(r["edge"]),"reason":reason,"signal_date":str(pd.Timestamp(r["date"]).date())})
 if issues:raise RuntimeError("Unresolved exits "+str(len(issues))+" first "+str(issues[:2]))
 return rows

def run(rows,prices,tag):
 if not rows:return None,[],[]
 p,l,c=portfolio(rows,prices,tag,rotation=False)
 return p,l,c

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--v442",default="results/prism-v442")
 ap.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv")
 ap.add_argument("--out",default="results/prism-v443");a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 prices={t:load_prices(t,"2021-01-01").loc[lambda x:x.index<=CUTOFF] for t in UNIVERSE}
 src=pd.read_csv(Path(a.v442)/"candidate_exits.csv")
 b=src[src.detector=="VCP_BREAKOUT"]
 if b.empty:raise RuntimeError("Missing breakout baseline")
 baseline=[{"ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,
  "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),"edge":float(r.edge),"reason":r.reason}
  for r in b.itertuples(index=False)]
 perf,ledger,curve=run(baseline,prices,"BREAKOUT")
 if perf is None:raise RuntimeError("No breakout portfolio")
 df=pd.DataFrame(ledger).sort_values("exit_date").copy()
 df["prior_capital"]=df.capital_after.shift().fillna(START)
 df["pnl"]=df.capital_after-df.prior_capital
 winners=df[df.pnl>0].sort_values("pnl",ascending=False)
 top=winners[["ticker","entry_date","pnl"]].head(3).to_dict("records")
 stress={}
 for n in (1,2,3):
  excluded={(x["ticker"],x["entry_date"]) for x in top[:n]}
  filtered=[r for r in baseline if (r["ticker"],r["entry_date"]) not in excluded]
  result,_,_=run(filtered,prices,"BREAKOUT_EX_TOP"+str(n))
  stress["EX_TOP"+str(n)]=result
 raw=[{"ticker":ticker,"date":dt,"edge":edge} for ticker,d in prices.items() for dt,edge in bch_confirm(d)]
 rows=make_rows(raw,prices)
 bch_result,bch_ledger,bch_curve=run(rows,prices,"BCH_CONFIRMED")
 seed=pd.read_csv(a.seed_a)
 seed=seed[seed.variant=="ATR4_CONTROL"].copy()
 seed.date=pd.to_datetime(seed.date).dt.normalize()
 aa=seed.set_index("date").equity.sort_index()
 report={"version":"v4.43","status":"HISTORICAL RESEARCH / NOT INDEPENDENT OOS",
 "breakout":{"baseline":perf,"top3_winners":top,"stress":stress},
 "bch_confirmed":{"candidate_count":len(rows),"result":bch_result},
 "combined":{},"limitations":[
 "Winner deletion is ex-post stress testing, not an executable selection policy.",
 "BCH setup must precede breakout; no future bar is used for signal.",
 "BCH thresholds and breakout volume 1.2x are exploratory and selected after viewing historical charts.",
 "Same retrospectively selected TECH80 and repeatedly researched 2022-2026 sample; not independent OOS.",
 "Signals in final 220 calendar days are excluded to ensure MA30 exit resolution.",
 "Seed A ATR4 terminal valuation is synthetic, not realized sale.",
 "Daily-close MDD omits intraday losses."]}
 allcurves=[];allledgers=[];daily=[]
 for tag,result,l,c in [("BREAKOUT",perf,ledger,curve),("BCH_CONFIRMED",bch_result,bch_ledger,bch_curve)]:
  if result is None:continue
  bb=pd.DataFrame(c).assign(date=lambda x:pd.to_datetime(x.date).dt.normalize()).set_index("date").equity
  calendar=aa.index.union(bb.index).sort_values()
  xa=aa.reindex(calendar);xb=bb.reindex(calendar)
  xa.loc[calendar<aa.index.min()]=START
  xb.loc[calendar<bb.index.min()]=START
  if xa.isna().any() or xb.isna().any():raise RuntimeError("Internal calendar gaps "+tag)
  total=xa+xb;peak=np.maximum.accumulate(np.r_[2*START,total.to_numpy()])[1:]
  report["combined"][tag]={"return":float(total.iloc[-1]/(2*START)-1),
   "mdd":float(np.min(total.to_numpy()/peak-1))}
  daily.extend({"date":str(day.date()),"strategy":tag,"seed_a":float(x),"seed_b":float(y),"total":float(x+y)}
   for day,x,y in zip(total.index,xa,xb))
  for r in l:r["strategy"]=tag
  for r in c:r["strategy"]=tag
  allledgers+=l;allcurves+=c
 pd.DataFrame(allledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(allcurves).to_csv(out/"daily_equity.csv",index=False)
 pd.DataFrame(daily).to_csv(out/"combined_equity.csv",index=False)
 pd.DataFrame(rows).to_csv(out/"bch_confirmed_candidates.csv",index=False)
 df.to_csv(out/"breakout_attribution.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.43</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.43 · Breakout Stress &amp; BCH Confirmation</h1><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","breakout":perf["return"],"bch":bch_result["return"] if bch_result else None}))
if __name__=="__main__":main()
