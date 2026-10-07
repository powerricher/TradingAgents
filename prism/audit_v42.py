"""PRISM v4.2 frozen-result validation/audit.

Consumes a completed v4.2 report.json. It does NOT retrain or tune the model.
Checks selection concentration, regime/year/ticker robustness, transaction-cost
sensitivity, and diagnoses zero-selection folds.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

def _pf(r):
 r=np.asarray(r,dtype=float); gp=r[r>0].sum(); gl=-r[r<0].sum()
 return float(gp/gl) if gl>0 else (999.0 if gp>0 else 0.0)

def _stats(df, cost_bps=0):
 if df.empty: return {"n":0}
 r=df.d1_return.astype(float).to_numpy()-cost_bps/10000.0
 return {"n":int(len(r)),"avg_return":float(r.mean()),"positive_rate":float((r>0).mean()),
         "profit_factor":_pf(r),"minus3_rate":float((r<=-.03).mean()),
         "net_pnl_krw":int(np.round(r.sum()*10_000_000))}

def run(source,out,cost_bps=(0,5,10,20)):
 src=json.loads(Path(source).read_text(encoding="utf-8"))
 meta=src["meta"]; df=pd.DataFrame(src.get("selected_rows",[]))
 if not df.empty:
  df["date"]=pd.to_datetime(df["date"]); df["year"]=df.date.dt.year
  df["regime"]=np.select(
   [df.regime_riskoff.astype(float)>0,df.regime_bear.astype(float)>0],
   ["RISK_OFF","BEAR"],default="NORMAL")
 ticker={k:_stats(g) for k,g in df.groupby("ticker")} if not df.empty else {}
 year={str(k):_stats(g) for k,g in df.groupby("year")} if not df.empty else {}
 regime={str(k):_stats(g) for k,g in df.groupby("regime")} if not df.empty else {}
 costs={str(b):_stats(df,float(b)) for b in cost_bps}
 # Concentration: how much of total absolute P&L is attributable to each ticker.
 if not df.empty:
  pnl=df.groupby("ticker").d1_return.sum().sort_values(ascending=False)
  denom=float(df.d1_return.abs().sum()) or 1.0
  concentration={"top5_tickers":pnl.head(5).to_dict(),
                 "top5_abs_pnl_share":float(pnl.head(5).abs().sum()/denom)}
 else: concentration={"top5_tickers":{},"top5_abs_pnl_share":0.0}
 zero=[]
 for f in meta.get("folds",[]):
  if f.get("selected",{}).get("n",0)==0:
   rp=f.get("risk_pass",{}); zero.append({
    "fold":f["fold"],"test":f["test"],"test_n":f.get("test_n",0),
    "risk_pass_n":rp.get("n",0),
    "diagnosis":"risk gate produced candidates, but no positive uncertainty-adjusted edge survived daily selection" if rp.get("n",0)>0 else "no candidates survived risk gate"
   })
 # Frozen acceptance remains reported, but audit adds practical validation flags.
 c10=costs.get("10",{})
 flags={
  "original_pass":bool(meta.get("verdict",{}).get("pass")),
  "pf_after_10bps_ge_1_15":bool((c10.get("profit_factor") or 0)>=1.15),
  "avg_after_10bps_positive":bool(c10.get("avg_return",0)>0),
  "no_single_zero_selection_fold":len(zero)==0,
  "sample_ge_200":len(df)>=200
 }
 result={"model":"PRISM v4.2 FROZEN VALIDATION/AUDIT","source_model":meta.get("model"),
         "frozen_parameters":{"top_pct":meta.get("top_pct"),"uncertainty_penalty":meta.get("uncertainty_penalty"),
                              "acceptance_rules":meta.get("acceptance_rules")},
         "pooled":_stats(df),"transaction_cost_bps_roundtrip":costs,
         "by_ticker":ticker,"by_year":year,"by_regime":regime,
         "concentration":concentration,"zero_selection_folds":zero,"audit_flags":flags}
 p=Path(out); p.parent.mkdir(parents=True,exist_ok=True)
 p.with_suffix(".json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
 def pct(x): return f"{x:.2%}"
 rows="".join(f"<tr><td>{b}</td><td>{s.get('n',0)}</td><td>{pct(s.get('avg_return',0))}</td><td>{s.get('profit_factor',0):.2f}</td><td>{pct(s.get('minus3_rate',0))}</td><td>{s.get('net_pnl_krw',0):,}</td></tr>" for b,s in costs.items())
 zrows="".join(f"<tr><td>{z['fold']}</td><td>{z['test'][0]}→{z['test'][1]}</td><td>{z['test_n']}</td><td>{z['risk_pass_n']}</td><td>{z['diagnosis']}</td></tr>" for z in zero) or "<tr><td colspan=5>None</td></tr>"
 html=f"""<!doctype html><meta charset='utf-8'><title>PRISM v4.2 Validation</title>
<style>body{{font-family:Arial;background:#0b1020;color:#eef3ff;margin:32px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{background:#151d35;padding:16px;border-radius:12px}}table{{width:100%;border-collapse:collapse;margin:18px 0}}td,th{{padding:9px;border-bottom:1px solid #27314d;text-align:right}}td:last-child,th:last-child{{text-align:left}}</style>
<h1>PRISM v4.2 — Frozen Validation / Audit</h1>
<p>No retraining · no parameter tuning · post-PASS robustness audit</p>
<div class=cards><div class=card>N <b>{len(df)}</b></div><div class=card>Original PF <b>{_stats(df).get('profit_factor',0):.2f}</b></div><div class=card>10bps PF <b>{c10.get('profit_factor',0):.2f}</b></div><div class=card>10bps Avg <b>{pct(c10.get('avg_return',0))}</b></div><div class=card>Zero folds <b>{len(zero)}</b></div></div>
<h2>Transaction-cost stress</h2><table><tr><th>Round-trip bps</th><th>N</th><th>Avg D+1</th><th>PF</th><th>-3%</th><th>10M KRW/trade P&L</th></tr>{rows}</table>
<h2>Zero-selection diagnosis</h2><table><tr><th>Fold</th><th>Period</th><th>Test N</th><th>Risk-pass N</th><th>Diagnosis</th></tr>{zrows}</table>
<pre>{json.dumps(flags,indent=2)}</pre>"""
 p.write_text(html,encoding="utf-8")
 return result
