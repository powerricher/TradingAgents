"""PRISM v4.15: FP/FN/capital lockup attribution, diagnostic not trading optimization.
Missed-upside is an opportunity proxy, never additive portfolio P&L.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .fast_lab import load_history
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--candidates",default="results/prism-v413/candidate_scores.csv.gz")
 p.add_argument("--signals",default="results/prism-v413/signals.csv")
 p.add_argument("--ledger",default="results/prism-v414/ledger.csv")
 p.add_argument("--trades",default="results/prism-v414/trades.csv")
 p.add_argument("--out",default="results/prism-v415")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 candidates=pd.read_csv(a.candidates);signals=pd.read_csv(a.signals)
 ledger=pd.read_csv(a.ledger);trades=pd.read_csv(a.trades)
 for frame in (candidates,signals,ledger,trades):
  frame["ticker"]=frame["ticker"].astype(str)
 base=signals[signals.variant=="TECH80"][["ticker","date"]].drop_duplicates()
 candidates=candidates.merge(base.assign(selected=True),on=["ticker","date"],how="left")
 candidates["selected"]=candidates.selected.fillna(False).astype(bool)
 candidates["date"]=pd.to_datetime(candidates.date)
 # Forward returns are diagnostic labels only. No future outcome may enter selection.
 issues=[];parts=[]
 for ticker,g in candidates.groupby("ticker"):
  try:
   d=load_history(ticker,"2021-01-01","2026-10-06").sort_index()
   if isinstance(d.index,pd.DatetimeIndex):
    d.index=d.index.tz_localize(None).normalize()
   close=d.Close.astype(float)
   labels=pd.DataFrame({"date":d.index})
   labels["ticker"]=ticker
   for h in (5,10,20):
    labels["fwd"+str(h)]=(close.shift(-h)/close-1).to_numpy()
   parts.append(g.merge(labels,on=["ticker","date"],how="left",validate="one_to_one"))
  except Exception as e:
   issues.append({"ticker":ticker,"error":str(e)})
 if issues:raise RuntimeError("Incomplete label coverage: "+json.dumps(issues))
 allrows=pd.concat(parts,ignore_index=True)
 # Losses from actually executed TECH80 trades; no double counting candidate signals.
 executed=ledger[ledger.variant=="TECH80"].copy()
 actual=trades[trades.variant=="TECH80"].copy()
 actual["entry_date"]=pd.to_datetime(actual.entry_date)
 actual["exit_date"]=pd.to_datetime(actual.exit_date)
 executed["entry_date"]=pd.to_datetime(executed.entry_date)
 executed["exit_date"]=pd.to_datetime(executed.exit_date)
 matched=executed.merge(actual[["ticker","entry_date","exit_date","ret","edge","reason"]],on=["ticker","entry_date","exit_date"],how="left",validate="one_to_one")
 if matched.ret.isna().any():raise RuntimeError("Ledger/trades mismatch")
 matched["prior_capital"]=matched.capital_after.shift(1).fillna(10_000_000.)
 matched["pnl_krw"]=matched.capital_after-matched.prior_capital
 losses=matched[matched.pnl_krw<0].copy()
 winners=matched[matched.pnl_krw>0].copy()
 # Missed high upside at the signal date; not an achievable trade-return claim.
 rejected=allrows[~allrows.selected].copy()
 thresholds={}
 for h in (5,10,20):
  col="fwd"+str(h)
  valid=rejected[rejected[col].notna()].copy()
  strong=valid[valid[col]>=.10].copy()
  strong["reason"]=np.select([~strong.risk_pass,strong.edge<=0],["risk_gate","nonpositive_edge"],default="positive_edge_not_selected")
  thresholds[str(h)]={"rejected_evaluable":len(valid),"missed_10pct_moves":len(strong),
    "miss_rate":float(len(strong)/len(valid)) if len(valid) else None,
    "reason_counts":{str(k):int(v) for k,v in strong.reason.value_counts().items()}}
 # Capital lockup: selected signals that overlap actual held positions.
 selected=allrows[allrows.selected].copy()
 holdings=executed[["entry_date","exit_date","ticker"]].copy()
 overlap=[]
 for r in selected.itertuples(index=False):
  d=r.date
  active=holdings[(holdings.entry_date<=d)&(holdings.exit_date>d)]
  if not active.empty:
   overlap.append({"ticker":r.ticker,"date":str(d.date()),"held_ticker":str(active.iloc[0].ticker),"fwd5":r.fwd5,"fwd10":r.fwd10,"fwd20":r.fwd20})
 locked=pd.DataFrame(overlap)
 result={"version":"v4.15","status":"DIAGNOSTIC / NOT OOS VALIDATED","scope":"TECH80 + fixed MA30",
  "false_positive":{"executed":len(matched),"losing_trades":len(losses),"loss_rate":float(len(losses)/len(matched)) if len(matched) else None,
   "total_loss_krw":float(losses.pnl_krw.sum()),"total_win_krw":float(winners.pnl_krw.sum()),
   "worst_trade_return":float(matched.ret.min()),"losses_by_entry_year":{str(k):int(v) for k,v in losses.groupby(losses.entry_date.dt.year).size().items()}},
  "false_negative_proxy":thresholds,
  "capital_lockup":{"overlapping_selected_signals":len(locked),"selected_signals":len(selected),"overlap_share":float(len(locked)/len(selected)) if len(selected) else None,
    "overlapping_positive_10pct_20d":int((locked.fwd20>=.10).sum()) if not locked.empty else 0},
  "notes":["False positive means realized losing executed trade, not classification error against D+1 target.",
   "False negative means rejected ticker-date with >=10% forward CLOSE return; not necessarily a tradable missed gain.",
   "Capital lockup counts overlapping selected signals, NOT additive forgone P&L; overlap may include same ticker.",
   "Missed-upside diagnostic uses future prices only after signals are frozen.",
   "Fixed 80-stock universe chosen retrospectively; survival bias and repeated historical research.",
   "Cost and stop slippage already included in v4.14 portfolio ledger; intraday MDD not assessed."]}
 matched.to_csv(out/"executed_attribution.csv",index=False)
 locked.to_csv(out/"overlap_opportunities.csv",index=False)
 rejected[["ticker","date","fold","edge","risk_pass","fwd5","fwd10","fwd20"]].to_csv(out/"rejected_candidates.csv.gz",index=False,compression="gzip")
 (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
 pretty=html.escape(json.dumps(result,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.15</title><style>body{background:#0b1629;color:#eaf4ff;font:15px system-ui;padding:25px}h1{color:#60e6c4}pre{white-space:pre-wrap}</style><h1>PRISM v4.15 · False Positive / Missed Upside / Capital Lockup</h1><p>Diagnostic only; not incremental realizable profit. Same historical TECH80 and fixed MA30.</p><pre>"+pretty+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","losses":len(losses),"missed_20d":thresholds["20"]["missed_10pct_moves"],"overlap":len(locked)},ensure_ascii=False))
if __name__=="__main__":main()
