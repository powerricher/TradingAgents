"""PRISM v4.61: frozen Seed A/B continuation versus transition audit.
No changes to entries, exits, ranks, or capital allocation.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd

def classify(r):
 # Fixed exploratory labels, based only on prior-close features.
 near=(r.distance_252>=-.07)
 aligned=bool(r.ma_aligned) and bool(r.ma200_rising)
 if near and aligned:return "CONTINUATION"
 if r.distance_252<-.15 and (not bool(r.ma_aligned) or not bool(r.ma200_rising)):
  return "TRANSITION_PROXY"
 return "MIXED_UNCLASSIFIED"

def main():
 out=Path("results/prism-v461");out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv("results/prism-v460/candidate_high_proximity.csv")
 required={"seed","ticker","entry_date","distance_50","distance_252",
  "ma_aligned","ma200_rising","candidate_gross_return","executed","pnl","terminal_mark"}
 if not required.issubset(df.columns):
  raise RuntimeError("Missing v4.60 fields: "+str(sorted(required-set(df.columns))))
 if df.empty:raise RuntimeError("Empty v4.60 candidates")
 for col in ("ma_aligned","ma200_rising","executed","terminal_mark"):
  if df[col].dtype==object:
   df[col]=df[col].astype(str).str.lower().map({"true":True,"false":False})
 if df[["ma_aligned","ma200_rising","executed"]].isna().any().any():
  raise RuntimeError("Unknown boolean feature values")
 df["type"]=df.apply(classify,axis=1)
 df["positive_candidate"]=df.candidate_gross_return>0
 df.to_csv(out/"classified_candidates.csv",index=False)
 trades=df[df.executed].copy()
 trades.to_csv(out/"classified_executed_trades.csv",index=False)
 groups=[]
 for (seed,typ),g in df.groupby(["seed","type"]):
  t=g[g.executed]
  groups.append({"seed":seed,"type":typ,"candidates":len(g),"executed":len(t),
   "candidate_win_fraction":float(g.positive_candidate.mean()),
   "candidate_mean_return":float(g.candidate_gross_return.mean()),
   "candidate_median_return":float(g.candidate_gross_return.median()),
   "executed_win_fraction":float((t.pnl>0).mean()) if len(t) else None,
   "executed_mean_pnl":float(t.pnl.mean()) if len(t) else None,
   "terminal_marks":int(t.terminal_mark.fillna(False).sum())})
 pd.DataFrame(groups).to_csv(out/"type_calibration.csv",index=False)
 winners=[]
 for seed in ("A","B"):
  t=trades[(trades.seed==seed)&(trades.pnl>0)].sort_values("pnl",ascending=False)
  for rank,r in enumerate(t.head(5).itertuples(index=False),1):
   winners.append({"seed":seed,"rank":rank,"ticker":r.ticker,"entry_date":r.entry_date,
    "type":r.type,"pnl":float(r.pnl),"distance_50":float(r.distance_50),
    "distance_252":float(r.distance_252),"terminal_mark":bool(r.terminal_mark)})
 pd.DataFrame(winners).to_csv(out/"top_winner_types.csv",index=False)
 coverage={"candidate_count":len(df),"executed_count":len(trades),
  "type_counts":df.type.value_counts().to_dict(),
  "unclassified":int((df.type=="MIXED_UNCLASSIFIED").sum())}
 report={"version":"v4.61","status":"FROZEN TREND TYPE DIAGNOSTIC / NOT OOS",
  "coverage":coverage,"group_calibration":groups,
  "limitations":[
   "Continuation: <=7% below 252-session high, MA50>MA150>MA200 and rising MA200.",
   "Transition proxy: >15% below 252-session high and MA not aligned or MA200 not rising.",
   "Mixed/unclassified is retained; no forced binary classification.",
   "Transition proxy is NOT confirmation of actual trend reversal.",
   "252-day history coverage excludes at least one previously profitable IPO candidate.",
   "Candidate returns are hypothetical frozen exits; executed PnL includes terminal marks.",
   "No portfolio simulation of type filters, no new entry/exit rules or capital allocation.",
   "Retrospective TECH universe and repeated in-sample inspection; not independent OOS."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 table="".join("<tr><td>"+html.escape(str(r["seed"]))+"</td><td>"+html.escape(str(r["type"]))+"</td><td>"+str(r["candidates"])+"</td><td>"+str(r["executed"])+"</td><td>"+f'{r["candidate_mean_return"]:.2%}'+"</td></tr>" for r in groups)
 page="<html lang='ko'><meta charset='utf-8'><title>PRISM v4.61</title><style>body{background:#0c182a;color:#edf5ff;font:15px system-ui;margin:28px}h1{color:#60dfba}table{width:100%;border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #34455c;text-align:left}</style><h1>PRISM v4.61 · Trend Continuation vs Transition</h1><p>Historical diagnostic only. Seed A/B trading rules unchanged.</p><table><tr><th>Seed</th><th>Type</th><th>Candidates</th><th>Executed</th><th>Candidate mean return</th></tr>"+table+"</table><p>See CSV for top winners, uncategorized cases and coverage.</p></html>"
 (out/"report.html").write_text(page,encoding="utf-8")
 print(json.dumps({"status":"generated","coverage":coverage},ensure_ascii=False))
if __name__=="__main__":main()
