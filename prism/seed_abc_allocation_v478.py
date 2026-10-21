"""PRISM v4.78: lagged monthly allocation over independent A/B/C sleeve returns.
Research proxy only: NOT a shared-cash trade replay, NOT OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
SEEDS=["A","B","C"]
START=30_000_000.
def main():
 out=Path("results/prism-v478");out.mkdir(parents=True,exist_ok=True)
 f=pd.read_csv("results/prism-v477/daily_equity.csv",parse_dates=["date"]).set_index("date").sort_index()
 if f.empty or not set(SEEDS).issubset(f):raise RuntimeError("Missing v4.77 sleeves")
 if f.index.duplicated().any() or (f[SEEDS]<=0).any().any():raise RuntimeError("Bad source curves")
 daily=f[SEEDS].pct_change().fillna(0)
 if (daily<=-1).any().any():raise RuntimeError("Invalid sleeve returns")
 # Rebalance only at month start using observations strictly before that day.
 # 63 trading days lookback, minimum 40; capped 60% per sleeve, 10% floor.
 months=daily.index.to_period("M")
 weights=np.ones(3)/3
 rows=[];decisions=[];equal=START;adaptive=START
 for i,(day,r) in enumerate(daily.iterrows()):
  if i==0 or months[i]!=months[i-1]:
   history=daily.iloc[max(0,i-63):i]
   if len(history)>=40:
    avg=history.mean().to_numpy()
    vol=history.std(ddof=1).to_numpy()
    # Positive lagged return/vol score; a flat sleeve receives no extra preference.
    score=np.maximum(avg,0)/np.maximum(vol,1e-6)
    if np.isfinite(score).all() and score.sum()>0:
     raw=score/score.sum()
     # Convex combination with equal allocation, then bounded projection.
     w=.5*raw+.5/3
     for _ in range(12):
      w=np.clip(w,.1,.6)
      delta=1-w.sum()
      free=(w<.6-1e-10) if delta>0 else (w>.1+1e-10)
      if not free.any():break
      w[free]+=delta/free.sum()
     weights=w/w.sum()
    else:weights=np.ones(3)/3
   else:weights=np.ones(3)/3
   decisions.append({"date":str(day.date()),"history_days":len(history),
      **{s:float(weights[j]) for j,s in enumerate(SEEDS)}})
  ret=r.to_numpy(dtype=float)
  # Constant-mix daily weights is a daily-rebalanced synthetic proxy;
  # rebalance decisions are monthly but within-month drift is NOT modeled.
  # Use buy-and-hold drift of capital sleeves between monthly decisions instead.
  if i==0 or months[i]!=months[i-1]:
   alloc=adaptive*weights.copy()
  alloc*=1+ret
  adaptive=float(alloc.sum())
  equal*=float(1+ret.mean())
  rows.append({"date":str(day.date()),"equal_equity":equal,"adaptive_equity":adaptive,
   **{s+"_weight":float(alloc[j]/adaptive) for j,s in enumerate(SEEDS)}})
 result=pd.DataFrame(rows);result.to_csv(out/"daily_equity.csv",index=False)
 pd.DataFrame(decisions).to_csv(out/"allocation_decisions.csv",index=False)
 annual=[]
 for yr,g in result.groupby(pd.to_datetime(result.date).dt.year):
  first=result.index[result.date==g.date.iloc[0]][0]
  for policy,col in (("EQUAL_DAILY","equal_equity"),("LAGGED_MONTHLY","adaptive_equity")):
   begin=START if first==0 else float(result.loc[first-1,col])
   annual.append({"year":int(yr),"policy":policy,"return":float(g[col].iloc[-1]/begin-1)})
 pd.DataFrame(annual).to_csv(out/"yearly_returns.csv",index=False)
 summary={}
 for name,col in (("EQUAL_DAILY","equal_equity"),("LAGGED_MONTHLY","adaptive_equity")):
  v=result[col].to_numpy();peak=np.maximum.accumulate(np.r_[START,v])[1:]
  summary[name]={"final_krw":float(v[-1]),"return":float(v[-1]/START-1),
    "mdd":float((v/peak-1).min())}
 report={"version":"v4.78","status":"SYNTHETIC_SLEEVE_ALLOCATION_NOT_EXECUTABLE_NOT_OOS",
 "strategies":summary,"yearly":annual,"allocation_decisions":len(decisions),
 "rule":"63 prior sessions positive mean/vol, 40 min; monthly target, 10-60% bounds, 50% shrinkage to equal",
 "limitations":["Independent Seed A/B/C returns, no shared cash or overlapping-position reconciliation.",
 "Monthly target weights drift intra-month; equal benchmark is daily constant-mix.",
 "No Seed D, no short/inverse engine.",
 "Allocation transaction costs, tax, and market impact NOT charged; source sleeves may include costs.",
 "Repeated historical research; not independent OOS.",
 "No guarantee of +200% annual returns."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join("<tr><td>%s</td><td>%.2f%%</td><td>%.2f%%</td></tr>"%(x["year"],100*x["return"],100*next(y["return"] for y in annual if y["year"]==x["year"] and y["policy"]=="LAGGED_MONTHLY")) for x in annual if x["policy"]=="EQUAL_DAILY")
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.78</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#5de4bd}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM v4.78 Lagged Capital Allocation</h1><p>Synthetic sleeves; not shared cash or OOS. Allocation costs excluded.</p><table><tr><th>Year</th><th>Equal</th><th>Lagged monthly</th></tr>"+trs+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"V478_COMPLETE","strategies":summary}))
if __name__=="__main__":main()
