"""PRISM v4.77: independent A/B/C capital-sleeve aggregation, D availability audit.
Not a shared-cash execution engine and not an OOS test.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
START=10_000_000.
def main():
 out=Path("results/prism-v477");out.mkdir(parents=True,exist_ok=True)
 sources={
  "A":("results/prism-v436/daily_equity.csv","variant","ATR4_CONTROL"),
  "B":("results/prism-v457/seed_b_equity.csv","experiment","TECH150_VCP_BREAKOUT_QUALITY_RANK"),
  "C":("results/prism-v476/daily_equity.csv","strategy","V476_REGIME_GATE")}
 series={}
 for seed,(file,key,value) in sources.items():
  p=Path(file)
  if not p.exists():raise RuntimeError("Missing Seed "+seed+" source "+file)
  df=pd.read_csv(p)
  if key not in df or "equity" not in df or "date" not in df:raise RuntimeError("Bad schema "+seed)
  df=df[df[key]==value].copy()
  if df.empty:raise RuntimeError("Empty Seed "+seed)
  df["date"]=pd.to_datetime(df.date).dt.normalize()
  if df.date.duplicated().any():raise RuntimeError("Duplicate equity date "+seed)
  series[seed]=df.sort_values("date").set_index("date").equity.astype(float)
 dates=series["A"].index.union(series["B"].index).union(series["C"].index).sort_values()
 aligned={}
 for seed,s in series.items():
  a=s.reindex(dates).ffill()
  a.loc[dates<s.index.min()]=START
  if a.isna().any():raise RuntimeError("Missing values "+seed)
  aligned[seed]=a
 frame=pd.DataFrame(aligned,index=dates)
 frame["combined"]=frame[["A","B","C"]].sum(axis=1)
 frame.to_csv(out/"daily_equity.csv",index_label="date")
 yearly=[]
 for year,g in frame.groupby(frame.index.year):
  previous=frame.loc[frame.index<g.index.min()]
  for seed in ("A","B","C","combined"):
   initial=float(previous[seed].iloc[-1]) if not previous.empty else START*(3 if seed=="combined" else 1)
   ending=float(g[seed].iloc[-1])
   yearly.append({"year":int(year),"seed":seed,"start_equity":initial,"end_equity":ending,"return":ending/initial-1})
 pd.DataFrame(yearly).to_csv(out/"yearly_returns.csv",index=False)
 vals=frame.combined.to_numpy();peak=np.maximum.accumulate(np.r_[3*START,vals])[1:]
 report={"version":"v4.77","status":"A_B_C_INDEPENDENT_SLEEVES_NOT_SHARED_CASH_NOT_OOS",
 "seed_d":"NOT_IMPLEMENTED_OR_VALIDATED",
 "starting_capital_krw":3*START,"final_krw":float(vals[-1]),
 "combined_return":float(vals[-1]/(3*START)-1),
 "combined_mdd":float(np.min(vals/peak-1)),
 "yearly_returns":yearly,
 "limitations":["A/B/C are independently capitalized sleeves, not a single shared-cash engine.",
 "D is absent; no A/B/C/D return claim is permitted.",
 "Different source strategy definitions and date coverage; earlier missing dates initialized at cash.",
 "Historical reused datasets and repeated tuning are not untouched OOS.",
 "Daily close MDD misses intraday extremes; check source costs and synthetic terminal marks.",
 "No new short or inverse strategy implemented; annual +200% is a target, not a forecast."]}
 (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
 trs="".join("<tr><td>%s</td><td>%s</td><td>%.2f%%</td></tr>"%(r["year"],r["seed"],100*r["return"]) for r in yearly)
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.77</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#5de4bd}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM v4.77 ABC independent sleeve audit</h1><p>NOT shared cash. Seed D unavailable. Historical, NOT OOS.</p><table><tr><th>Year</th><th>Seed</th><th>Return</th></tr>"+trs+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"V477_A_B_C_AUDIT","combined_return":report["combined_return"],"combined_mdd":report["combined_mdd"]}))
if __name__=="__main__":main()
