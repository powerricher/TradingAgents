"""PRISM v4.52: 120-170 session triangle/wedge discovery.
Confirmed pivot extrema only; breakout at close, entry next open.
Seed A remains frozen TECH80 ATR4; Seed B MA30 CONFIRM30_1.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v445 import UNIVERSE,CORE100
from .universe_v413 import UNIVERSE as CORE80
from .trend_v48 import load_prices,portfolio
from .trend_v49 import exit_trade
from .atr_frontier_v433 import CUTOFF
INITIAL=10_000_000.
PATTERNS=("ASCENDING_TRIANGLE","DESCENDING_TRIANGLE_REVERSAL","FALLING_WEDGE")
def pivots(arr,kind):
 # Three-session pivot confirmation, no future data at decision time.
 ix=[]
 for j in range(3,len(arr)-3):
  a=arr[j-3:j+4]
  if (arr[j]>=max(a) if kind=="high" else arr[j]<=min(a)):
   ix.append(j)
 return ix
def fitline(points,values):
 xs=np.asarray(points,dtype=float)
 ys=np.asarray(values,dtype=float)
 if len(xs)<3 or xs.max()-xs.min()<35:return None
 m,b=np.polyfit(xs,ys,1)
 return float(m),float(b)
def discover(d,lookback=150):
 c=d.Close.to_numpy(dtype=float);h=d.High.to_numpy(dtype=float)
 l=d.Low.to_numpy(dtype=float);v=d.Volume.to_numpy(dtype=float)
 out=[]
 # 1-session step for all observations, no future pivots.
 for i in range(max(lookback+6,180),len(d)-1):
  day=d.index[i]
  if day<pd.Timestamp("2022-01-01") or day>CUTOFF-pd.Timedelta(days=220):continue
  begin=i-lookback
  # Pivots must be fully confirmed BEFORE the breakout bar.
  hi=pivots(h[begin:i-1],"high")
  lo=pivots(l[begin:i-1],"low")
  if len(hi)<3 or len(lo)<3:continue
  upper=fitline(hi[-5:],[h[begin+x] for x in hi[-5:]])
  lower=fitline(lo[-5:],[l[begin+x] for x in lo[-5:]])
  if upper is None or lower is None:continue
  mu,bu=upper;ml,bl=lower
  # Trendlines are based on log prices to normalize scale.
  # Convert price slopes to relative change per session.
  mid=max(1.,float(np.median(c[begin:i])))
  su=mu/mid;sl=ml/mid
  width_start=(mu*20+bu)-(ml*20+bl)
  width_end=(mu*(lookback-2)+bu)-(ml*(lookback-2)+bl)
  if width_start<=0 or width_end<=0 or width_end>=.75*width_start:continue
  resistance=mu*(lookback-1)+bu
  if resistance<=0 or c[i]<=1.01*resistance or c[i-1]>resistance:continue
  avgvol=np.mean(v[i-20:i])
  if avgvol<=0 or v[i]<1.3*avgvol:continue
  if abs(su)<.00035 and sl>.00015:kind="ASCENDING_TRIANGLE"
  elif su<-.00015 and abs(sl)<.00035:kind="DESCENDING_TRIANGLE_REVERSAL"
  elif su<-.00015 and sl<-.00015 and sl>su:kind="FALLING_WEDGE"
  else:continue
  out.append({"ticker":"","signal_date":str(day.date()),"pattern":kind,
   "edge":float(c[i]/resistance-1),
   "width_ratio":float(width_end/width_start),
   "upper_slope":float(su),"lower_slope":float(sl),
   "volume_ratio":float(v[i]/avgvol)})
 return out
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv")
 ap.add_argument("--out",default="results/prism-v452");a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 prices={};errors=[]
 for ticker in UNIVERSE:
  try:prices[ticker]=load_prices(ticker,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
  except Exception as exc:errors.append({"ticker":ticker,"error":str(exc)})
 if len(prices)<80:raise RuntimeError("Insufficient price coverage")
 seed=pd.read_csv(a.seed_a)
 seed=seed[seed.variant=="ATR4_CONTROL"].copy()
 seed.date=pd.to_datetime(seed.date).dt.normalize()
 aa=seed.set_index("date").equity.sort_index()
 if aa.empty:raise RuntimeError("Missing frozen Seed A")
 allsignals=[];trades=[];unresolved=[]
 for ticker,d in prices.items():
  if len(d)<190:continue
  for r in discover(d):
   r["ticker"]=ticker
   allsignals.append(r)
   try:z=exit_trade(d,pd.Timestamp(r["signal_date"]),"CONFIRM30_1")
   except (KeyError,ValueError):z=None
   if z is None:
    unresolved.append({"ticker":ticker,"signal_date":r["signal_date"]});continue
   e,x,entry,px,reason=z
   trades.append({**r,"entry_date":str(d.index[e].date()),"exit_date":str(d.index[x].date()),
    "entry_price":float(entry),"exit_price":float(px),"reason":reason})
 signals=pd.DataFrame(allsignals,columns=["ticker","signal_date","pattern","edge","width_ratio","upper_slope","lower_slope","volume_ratio"])
 candidates=pd.DataFrame(trades,columns=list(signals.columns)+["entry_date","exit_date","entry_price","exit_price","reason"])
 signals.to_csv(out/"pattern_signals.csv",index=False)
 candidates.to_csv(out/"candidate_exits.csv",index=False)
 pd.DataFrame(unresolved).to_csv(out/"unresolved_exits.csv",index=False)
 (out/"price_errors.json").write_text(json.dumps(errors,ensure_ascii=False,indent=2))
 report={"version":"v4.52","status":"EXPLORATORY HISTORICAL PATTERN DISCOVERY",
  "coverage":{"requested":150,"available":len(prices),"price_errors":len(errors)},
  "signals":len(signals),"resolved_candidates":len(candidates),"unresolved":len(unresolved),
  "by_pattern":signals.pattern.value_counts().to_dict() if len(signals) else {},
  "strategies":{},"limitations":[
   "150-session long formation used (approximately 7 months); 120 and 170 session sensitivity NOT tested.",
   "Pivot extrema confirmed by 3 later bars, all before the breakout session.",
   "Linear high/low regressions and slope thresholds are exploratory chart-pattern approximations.",
   "Only upside breakouts considered; rising wedge acts as a risk concept but is NOT implemented.",
   "New patterns are independent entry signals, not combined with existing BCH or VCP in this version.",
   "Signals within final 220 calendar days excluded to avoid unresolved MA30 exits.",
   "Retrospective universe and historical tuning create selection bias; not independent OOS.",
   "No entry gap limit or stop-fill liquidity simulation; daily close MDD excludes intraday losses."]}
 groups={"TECH80":set(CORE80),"TECH100":set(CORE100),"TECH150":set(UNIVERSE)}
 ledgers=[];curves=[];combined=[]
 for group,allowed in groups.items():
  for pattern in (*PATTERNS,"ALL_LONG_PATTERNS"):
   z=candidates[candidates.ticker.isin(allowed)]
   if pattern!="ALL_LONG_PATTERNS":z=z[z.pattern==pattern]
   if z.empty:
    report["strategies"][group+"_"+pattern]={"status":"NO_SIGNALS"};continue
   rows=[{"ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,
    "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),
    "edge":float(r.edge),"reason":r.reason} for r in z.itertuples(index=False)]
   name=group+"_"+pattern
   try:perf,ledger,curve=portfolio(rows,prices,name,rotation=False)
   except Exception as exc:
    report["strategies"][name]={"status":"INVALID","error":str(exc)};continue
   bb=pd.DataFrame(curve).assign(date=lambda x:pd.to_datetime(x.date).dt.normalize()).set_index("date").equity
   cal=aa.index.union(bb.index).sort_values()
   xa=aa.reindex(cal);xb=bb.reindex(cal)
   xa.loc[cal<aa.index.min()]=INITIAL;xb.loc[cal<bb.index.min()]=INITIAL
   if xa.isna().any() or xb.isna().any():raise RuntimeError("Internal valuation gap")
   total=xa+xb;peak=np.maximum.accumulate(np.r_[2*INITIAL,total.to_numpy()])[1:]
   report["strategies"][name]={"seed_b":perf,"combined_return":float(total.iloc[-1]/(2*INITIAL)-1),
    "combined_mdd":float(np.min(total.to_numpy()/peak-1)),"candidate_count":len(rows)}
   for x in ledger:x["strategy"]=name
   for x in curve:x["strategy"]=name
   ledgers+=ledger;curves+=curve
   combined.extend({"date":str(day.date()),"strategy":name,"seed_a_equity":float(av),
    "seed_b_equity":float(bv),"total_equity":float(av+bv)} for day,av,bv in zip(cal,xa,xb))
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"seed_b_daily_equity.csv",index=False)
 pd.DataFrame(combined).to_csv(out/"combined_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.52</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.52 · Long Formation Patterns</h1><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","signals":len(signals),"strategies":len(report["strategies"])}))
if __name__=="__main__":main()
