"""PRISM v4.65: TECH200 Seed B displacement and sector audit,
plus explicitly separate Seed A VCP proxy expansion feasibility.
No silent substitution of trained Seed A rank.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v445 import UNIVERSE as TECH150
from .universe_v464 import UNIVERSE as TECH200,NEW50,CANDIDATE_ADDITIONS
from .trend_v48 import load_prices,portfolio
from .atr_frontier_v433 import exit_atr,CUTOFF
from .price_action_v419 import feature_frame
START=10_000_000.
def main():
 out=Path("results/prism-v465");out.mkdir(parents=True,exist_ok=True)
 candidates=pd.read_csv("results/prism-v464/candidate_exits.csv")
 ledger=pd.read_csv("results/prism-v464/ledger.csv")
 if candidates.empty or ledger.empty:raise RuntimeError("Missing v4.64 inputs")
 if not {"ticker","entry_date","exit_date","quality","original_edge","entry_price","exit_price"}.issubset(candidates):
  raise RuntimeError("v4.64 candidate schema mismatch")
 sector={t:"CORE_TECH150" for t in TECH150}
 for category,tickers in CANDIDATE_ADDITIONS.items():
  for t in tickers:sector[t]=category
 candidates["sector"]=candidates.ticker.map(sector)
 candidates["is_new50"]=candidates.ticker.isin(NEW50)
 candidates["candidate_gross_return"]=candidates.exit_price/candidates.entry_price-1
 candidates.to_csv(out/"candidate_sector_audit.csv",index=False)
 # Compare executed transactions, not merely differences in signal counts.
 a=ledger[ledger.experiment=="TECH150_QUALITY_RANK"].copy()
 b=ledger[ledger.experiment=="TECH200_QUALITY_RANK"].copy()
 if a.empty or b.empty:raise RuntimeError("Missing frozen quality ledgers")
 a["entry_date"]=a.entry_date.astype(str).str[:10]
 b["entry_date"]=b.entry_date.astype(str).str[:10]
 def audit_exec(x,tag):
  x=x.sort_values(["exit_date","entry_date"]).copy()
  x["pnl_krw"]=x.capital_after.diff().fillna(x.capital_after-START)
  x["sector"]=x.ticker.map(sector)
  x["strategy"]=tag
  return x
 a=audit_exec(a,"TECH150");b=audit_exec(b,"TECH200")
 pd.concat([a,b]).to_csv(out/"executed_trade_comparison.csv",index=False)
 keya=set(zip(a.ticker,a.entry_date))
 keyb=set(zip(b.ticker,b.entry_date))
 displaced=a[[ (t,d) not in keyb for t,d in zip(a.ticker,a.entry_date)]].copy()
 added=b[[ (t,d) not in keya for t,d in zip(b.ticker,b.entry_date)]].copy()
 displaced["change"]="REMOVED_FROM_EXECUTION"
 added["change"]="ADDED_TO_EXECUTION"
 pd.concat([displaced,added]).to_csv(out/"execution_displacement.csv",index=False)
 sector_table=candidates.groupby(["sector","is_new50"]).agg(
  candidates=("ticker","size"),unique_tickers=("ticker","nunique"),
  mean_candidate_gross=("candidate_gross_return","mean"),
  median_candidate_gross=("candidate_gross_return","median")).reset_index()
 sector_table.to_csv(out/"sector_candidate_summary.csv",index=False)
 # Seed A original historical TECH80 is a frozen trained model.
 # Additional TECH200 VCP proxy signals are exploratory, not comparable to its rank.
 prices={};errors=[];a_rows=[];a_unresolved=[]
 for t in TECH200:
  try:prices[t]=load_prices(t,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
  except Exception as e:errors.append({"ticker":t,"error":str(e)})
 for t,d in prices.items():
  if len(d)<200:continue
  f=feature_frame(d)
  c=d.Close.astype(float)
  # Prespecified price-action VCP proxy, NOT trained historical VCP_ALL.
  eligible=(f.contraction_ratio<.70)&(f.range_contraction<.80)&(f.volume_dryup<.85)&(c>c.rolling(50).mean())
  trigger=eligible&(~eligible.shift(1).fillna(False).astype(bool))
  trigger=trigger&(d.index>=pd.Timestamp("2022-01-01"))&(d.index<=CUTOFF-pd.Timedelta(days=220))
  for day in d.index[trigger.fillna(False)]:
   ix=d.index.get_loc(day)
   if ix+1>=len(d):continue
   entry=d.index[ix+1]
   try:xd,xp,reason=exit_atr(d,entry,4)
   except Exception as e:
    a_unresolved.append({"ticker":t,"date":str(day.date()),"error":str(e)});continue
   a_rows.append({"ticker":t,"signal_date":str(day.date()),"entry_date":str(entry.date()),
    "exit_date":str(xd.date()),"entry_price":float(d.Open.iloc[ix+1]),"exit_price":float(xp),
    "edge":float(1-f.contraction_ratio.loc[day]),"reason":reason,
    "sector":sector[t],"is_new50":t in NEW50})
 pd.DataFrame(a_rows).to_csv(out/"seed_a_vcp_proxy_candidates.csv",index=False)
 pd.DataFrame(a_unresolved).to_csv(out/"seed_a_unresolved.csv",index=False)
 (out/"coverage_errors.json").write_text(json.dumps(errors,indent=2))
 report={"version":"v4.65","status":"TECH200 SELECTION AUDIT / HISTORICAL NOT OOS",
  "seed_b":{"tech150_executed":len(a),"tech200_executed":len(b),
    "removed_executions":len(displaced),"added_executions":len(added),
    "new50_executed":int(b.ticker.isin(NEW50).sum()),
    "new50_realized_pnl_sum":float(b[b.ticker.isin(NEW50)].pnl_krw.sum()),
    "note":"P&L sum is path-dependent; not causal attribution of universe expansion"},
  "seed_a":{"original_trained_model":"TECH80 FROZEN / NOT EXPANDED",
    "proxy_candidates":len(a_rows),"proxy_new50_candidates":sum(x["is_new50"] for x in a_rows),
    "proxy_unresolved":len(a_unresolved),"status":"SEPARATE EXPLORATORY PROXY / NOT ORIGINAL MODEL"},
  "limitations":[
   "TECH200 Seed A trained VCP rank not reproduced; no false like-for-like comparison.",
   "VCP proxy is a separate untrained experiment; its candidates are not executed here.",
   "Seed B uses frozen v4.64 candidate/ledger files; no new exits or rank weights.",
   "Execution displacement is path-dependent; cannot attribute return difference to a single trade without replay.",
   "Sector groups reflect proposed additions; original TECH150 remains CORE_TECH150.",
   "Historical winner selection, delisting omissions, survivorship bias and incomplete exits remain.",
   "No strategy promoted, no live trading or alerts."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.65</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.65 · TECH200 Selection Audit</h1><p>Seed B execution displacement and sector composition; Seed A original rank frozen, proxy isolated.</p><pre>"+html.escape(json.dumps(report,indent=2,ensure_ascii=False))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"AUDIT_GENERATED","seed_b_added":len(added),"seed_a_proxy_candidates":len(a_rows)}))
if __name__=="__main__":main()
