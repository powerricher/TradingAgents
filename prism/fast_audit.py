"""Fast research audit from a previously frozen PRISM run; no rolling refit.
Not a D+3/D+5 predictive model. Never use future returns for signal selection.
"""
import argparse,json,html
from pathlib import Path
import pandas as pd,numpy as np
BIO=set("VKTX CRNX IONS RARE MIRM SRRK PCVX IMTX".split())
def pf(a):
 p=a[a>0].sum();l=-a[a<0].sum()
 return float(p/l) if l>0 else None
def main():
 p=argparse.ArgumentParser();p.add_argument("--trades",required=True);p.add_argument("--signals",required=True);p.add_argument("--out",default="results/prism-v44-fast")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 trades=pd.read_csv(a.trades);s=json.loads(Path(a.signals).read_text());sig=pd.DataFrame(s["selected_rows"])
 trades["sector"]=np.where(trades.ticker.isin(BIO),"BIO","TECH/OTHER")
 trades["year"]=pd.to_datetime(trades.entry_date).dt.year
 data={"provenance":{"signal_source":"frozen v4.3 50-stock rolling OOS","models":"A next-close; B red candle; C trend","prediction_horizon":"D+1 only; D+3/D+5 forecasts NOT trained","status":"retrospective challenger audit"},"models":{},"segments":[],"warnings":[]}
 for m,g in trades.groupby("mode"):
  r=g.ret.to_numpy();top=max(1,int(np.ceil(len(r)*.05)))
  ordered=np.sort(r)[::-1]
  costs={}
  for bps in [0,5,10,20]:
   z=r-bps/10000
   costs[str(bps)]={"mean_net":float(z.mean()),"pf_net":pf(z)}
  data["models"][m]={"n":len(g),"mean":float(r.mean()),"median":float(np.median(r)),"win_rate":float((r>0).mean()),"pf":pf(r),"avg_hold":float(g.holding_days.mean()),"cost_stress_bps":costs,"top5pct_share_of_net_sum":float(ordered[:top].sum()/r.sum()) if abs(r.sum())>1e-9 else None,"mean_without_top5pct":float(ordered[top:].mean()),"worst_trade":float(r.min()),"p05":float(np.quantile(r,.05))}
  for (sector,year),h in g.groupby(["sector","year"]):
   data["segments"].append({"mode":m,"sector":sector,"year":int(year),"n":len(h),"mean":float(h.ret.mean()),"pf":pf(h.ret.to_numpy())})
 if len(trades.groupby("mode").size().unique())!=1:data["warnings"].append("Unequal exit samples; paired comparison invalid")
 data["warnings"]+=["C strategy uses same-bar closing price for trend exit; actionable only after close, requires next-open execution validation.","Selection universe chosen retrospectively; survivorship and universe look-ahead must be audited.","Overlapping signals may require more capital than independent 10m KRW tickets.","BIO split is diagnostic only; insufficient evidence for separate trained BIO model."]
 (out/"summary.json").write_text(json.dumps(data,indent=2,ensure_ascii=False))
 trades.to_csv(out/"trades.csv",index=False)
 rows="".join("<tr><td>%s</td><td>%s</td><td>%.3f%%</td><td>%.3f%%</td><td>%.2f</td><td>%.2f</td></tr>"%(m,v["n"],v["mean"]*100,v["median"]*100,v["pf"] or 0,v["avg_hold"]) for m,v in data["models"].items())
 report="<meta charset='utf-8'><style>body{background:#0b1325;color:#e7f2ff;font:16px Arial;margin:30px}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #344466}h1{color:#5eead4}</style><h1>PRISM FAST — Frozen Entry Exit Audit</h1><p>50-stock 2022–2026 historical signals. No retraining. NOT a validated D+3/D+5 predictor.</p><table><tr><th>Exit</th><th>N</th><th>Mean</th><th>Median</th><th>PF</th><th>Hold</th></tr>"+rows+"</table><h2>Cost / concentration / sector audit</h2><pre>"+html.escape(json.dumps(data,indent=2,ensure_ascii=False))+"</pre>"
 (out/"report.html").write_text(report)
 print("FAST AUDIT COMPLETE",out)
if __name__=="__main__":main()
