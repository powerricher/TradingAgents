"""PRISM v4.16 Risk Gate effectiveness / loss attribution.
Research-only, descriptive comparisons. No tuning using future returns.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .fast_lab import load_history

def safe_rate(x):
 return float(x.mean()) if len(x) else None
def cohort(df,h):
 col=f"fwd{h}"
 z=df[df[col].notna()]
 r=z[col].to_numpy()
 return {"n":int(len(z)),"mean":float(r.mean()) if len(r) else None,
         "median":float(np.median(r)) if len(r) else None,
         "p_up10":safe_rate(r>=.10),"p_down5":safe_rate(r<=-.05),
         "p_down10":safe_rate(r<=-.10)}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--candidates",default="results/prism-v413/candidate_scores.csv.gz")
 p.add_argument("--signals",default="results/prism-v413/signals.csv")
 p.add_argument("--executed",default="results/prism-v415/executed_attribution.csv")
 p.add_argument("--out",default="results/prism-v416")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 candidates=pd.read_csv(a.candidates)
 signals=pd.read_csv(a.signals)
 executed=pd.read_csv(a.executed)
 candidates.date=pd.to_datetime(candidates.date)
 for col in ("risk_pass","quality_pass","market_pass"):
  candidates[col]=candidates[col].astype(bool)
 label_rows=[];errors=[]
 for ticker,g in candidates.groupby("ticker"):
  try:
   d=load_history(ticker,"2021-01-01","2026-10-06").sort_index()
   d.index=pd.to_datetime(d.index).tz_localize(None).normalize()
   close=d.Close.astype(float)
   labels=pd.DataFrame({"date":d.index,"ticker":ticker})
   for h in (5,10,20):
    labels[f"fwd{h}"]=(close.shift(-h)/close-1).to_numpy()
   label_rows.append(g.merge(labels,on=["ticker","date"],how="left",validate="one_to_one"))
  except Exception as e:errors.append({"ticker":ticker,"error":str(e)})
 if errors:raise RuntimeError("Incomplete price coverage: "+json.dumps(errors))
 panel=pd.concat(label_rows,ignore_index=True)
 report={"version":"v4.16","status":"DESCRIPTIVE / NOT INDEPENDENT OOS","universe":"TECH80","cohorts":{},"folds":{},"loss_profiles":{},"notes":[
 "All outcomes are future-close research labels, not executable portfolio returns.",
 "Risk Gate was fitted in each historical fold; selection and research universe remain retrospectively studied.",
 "A higher missed-upside count alone does not imply gate relaxation improves P&L.",
 "Different cohorts have different market-regime exposure; comparisons are observational, not causal.",
 "Do not optimize gates on these historical results and call the same data OOS.",
 "A 20-session label near dataset end is unavailable; it is excluded, never filled."]}
 for h in (5,10,20):
  report["cohorts"][str(h)]={name:cohort(frame,h) for name,frame in [
    ("PASS",panel[panel.risk_pass]),("REJECT",panel[~panel.risk_pass]),
    ("PASS_EDGE_POS",panel[panel.risk_pass&(panel.edge>0)]),
    ("REJECT_EDGE_POS",panel[(~panel.risk_pass)&(panel.edge>0)])]}
 for fold,g in panel.groupby("fold"):
  report["folds"][str(int(fold))]={"pass":cohort(g[g.risk_pass],20),"reject":cohort(g[~g.risk_pass],20)}
 selected=signals[signals.variant.eq("TECH80")][["ticker","date"]].drop_duplicates()
 selected.date=pd.to_datetime(selected.date)
 scored=selected.merge(panel,on=["ticker","date"],how="left",validate="one_to_one")
 if scored.edge.isna().any():raise RuntimeError("Missing selected signal features")
 # Actual executed loss profile: join signal at preceding session date via candidate signal dates.
 executed.entry_date=pd.to_datetime(executed.entry_date)
 executed.exit_date=pd.to_datetime(executed.exit_date)
 candidates_by_ticker={t:g.sort_values("date") for t,g in scored.groupby("ticker")}
 profiles=[]
 for r in executed.itertuples(index=False):
  g=candidates_by_ticker.get(r.ticker)
  if g is None:continue
  prior=g[g.date<r.entry_date]
  if prior.empty:continue
  s=prior.iloc[-1]
  profiles.append({"ticker":r.ticker,"entry_date":str(r.entry_date.date()),"pnl_krw":float(r.pnl_krw),
   "edge":float(s.edge),"p_down":float(s.p_down),"rel5":float(s.rel5),
   "stock_vol20":float(s.stock_vol20),"qqq_ret5":float(s.qqq_ret5),
   "qqq_vs_ma20":float(s.qqq_vs_ma20)})
 prof=pd.DataFrame(profiles)
 if len(prof)!=len(executed):raise RuntimeError(f"Matched {len(prof)} of {len(executed)} actual entries")
 prof["loss"]=prof.pnl_krw<0
 for label,g in [("WIN",prof[~prof.loss]),("LOSS",prof[prof.loss])]:
  report["loss_profiles"][label]={"n":len(g),"mean_features":{col:float(g[col].mean()) if len(g) else None for col in ["edge","p_down","rel5","stock_vol20","qqq_ret5","qqq_vs_ma20"]}}
 panel[["ticker","date","fold","risk_pass","edge","p_down","fwd5","fwd10","fwd20"]].to_csv(out/"gate_cohorts.csv.gz",index=False,compression="gzip")
 prof.to_csv(out/"executed_loss_profiles.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 rows="".join(f"<tr><td>{h}</td><td>{report['cohorts'][h]['PASS']['p_up10']:.1%}</td><td>{report['cohorts'][h]['REJECT']['p_up10']:.1%}</td><td>{report['cohorts'][h]['PASS']['p_down10']:.1%}</td><td>{report['cohorts'][h]['REJECT']['p_down10']:.1%}</td></tr>" for h in ("5","10","20"))
 page="<html lang='ko'><meta charset='utf-8'><title>PRISM v4.16</title><style>body{background:#0b1728;color:#eaf3ff;font:15px system-ui;margin:30px}h1{color:#62e5c3}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.16 · Risk Gate Effectiveness</h1><p>TECH80 · observational diagnostics · NOT a causal estimate or portfolio backtest</p><table><tr><th>Horizon</th><th>Pass: +10%</th><th>Reject: +10%</th><th>Pass: -10%</th><th>Reject: -10%</th></tr>"+rows+"</table><p>See summary.json for fold breakdown and executed loss profiles.</p></html>"
 (out/"report.html").write_text(page,encoding="utf-8")
 print(json.dumps({"status":"generated","cohorts":report["cohorts"],"loss_profiles":report["loss_profiles"]},ensure_ascii=False))
if __name__=="__main__":main()
