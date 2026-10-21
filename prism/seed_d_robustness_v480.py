"""PRISM v4.80: bearish signal robustness and regime diagnostic, NOT executable P&L."""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
from .regime_adaptive_v436 import regimes
def main():
 out=Path("results/prism-v480");out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv("results/prism-v479/signal_forward_returns.csv")
 df=df[df.detector.isin(["D1_SHORT_BREAKDOWN_PROXY","D3_OVERSOLD_LONG_PROXY"])].copy()
 if df.empty:raise RuntimeError("No bearish candidates")
 df["entry_date"]=pd.to_datetime(df.entry_date).dt.normalize()
 market=regimes(pd.Timestamp("2022-01-01"))
 df["regime"]=df.entry_date.map(market)
 if df.regime.isna().any() or (df.regime=="UNKNOWN").any():raise RuntimeError("Regime coverage incomplete")
 rows=[]
 for (det,reg),g in df.groupby(["detector","regime"]):
  for h in (1,3,5):
   a=g[f"ret_{h}"].astype(float)
   if not np.isfinite(a).all():raise RuntimeError("Nonfinite returns")
   lo,hi=a.quantile([.05,.95])
   rows.append({"detector":det,"regime":reg,"horizon":h,"signals":len(a),
    "mean":float(a.mean()),"median":float(a.median()),"win_rate":float((a>0).mean()),
    "winsorized_mean":float(a.clip(lo,hi).mean()),"q05":float(lo),
    "net_30bps_mean":float(a.mean()-.003),
    "net_60bps_mean":float(a.mean()-.006)})
 pd.DataFrame(rows).to_csv(out/"regime_robustness.csv",index=False)
 yearly=[]
 for (det,year),g in df.groupby(["detector",df.entry_date.dt.year]):
  a=g.ret_1.astype(float);lo,hi=a.quantile([.05,.95])
  yearly.append({"detector":det,"year":int(year),"signals":len(a),
    "d1_mean":float(a.mean()),"d1_winsor":float(a.clip(lo,hi).mean()),
    "d1_net30bps":float(a.mean()-.003)})
 pd.DataFrame(yearly).to_csv(out/"yearly_d1.csv",index=False)
 report={"version":"v4.80","status":"SIGNAL_DIAGNOSTIC_ONLY_NOT_PORTFOLIO_NOT_OOS",
 "rows":rows,"yearly":yearly,
 "limitations":["Independent overlapping candidate returns; no cash occupancy or trade execution.",
 "D1 inverse-price return is NOT actual short-sale P&L; borrow availability and fees absent.",
 "30/60bps fixed cost sensitivity is not a verified trading cost.",
 "Historical current-listed universe has survivorship bias.",
 "QQQ prior-close regime uses historical prices; repeated strategy selection is in-sample.",
 "No intraday stops, D2 or D4 backtest; no automatic trading."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join("<tr><td>%s</td><td>%s</td><td>%d</td><td>%d</td><td>%.2f%%</td><td>%.2f%%</td></tr>"%(html.escape(x["detector"]),x["regime"],x["horizon"],x["signals"],100*x["mean"],100*x["winsorized_mean"]) for x in rows)
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.80</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#5de4bd}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM v4.80 Bear Signal Robustness</h1><p>Independent signals only, not executable short P&L or OOS.</p><table><tr><th>Detector</th><th>Regime</th><th>Horizon</th><th>N</th><th>Mean</th><th>Winsor</th></tr>"+trs+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"V480_COMPLETE","rows":len(rows),"signals":len(df)}))
if __name__=="__main__":main()
