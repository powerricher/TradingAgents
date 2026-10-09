"""PRISM v4.37: adaptive exit robustness audit.
Freeze v4.36 strategies; analyze concentration, regime transitions and
leave-top-winner-out portfolio replay. No post-hoc optimization.
"""
import argparse,json,html
from pathlib import Path
import pandas as pd,numpy as np
from .trend_v48 import load_prices,portfolio
from .regime_adaptive_v436 import regimes
from .atr_frontier_v433 import CUTOFF

INITIAL=10_000_000.
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--input",default="results/prism-v436")
 p.add_argument("--out",default="results/prism-v437")
 a=p.parse_args();root=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 candidates=pd.read_csv(root/"candidate_exits.csv")
 ledger=pd.read_csv(root/"ledger.csv")
 equity=pd.read_csv(root/"daily_equity.csv")
 if not len(candidates) or not len(ledger):raise RuntimeError("Missing v4.36 records")
 prices={t:load_prices(t,g.entry_date.min()).loc[lambda x:x.index<=CUTOFF] for t,g in candidates.groupby("ticker")}
 states=regimes(pd.to_datetime(candidates.entry_date).min())
 report={"version":"v4.37","status":"HISTORICAL ROBUSTNESS / NOT INDEPENDENT OOS","strategies":{},
 "limitations":[
 "Removing a historically best winner is an ex-post stress test, never a tradable selection rule.",
 "Replaying after removing a candidate changes later positions and opportunity costs.",
 "Original v4.36 adaptive bull MA30 and original MA30 control use different exit engines; this audit flags but does not fully reconcile them.",
 "Entry-regime performance is descriptive and depends on different holding lengths.",
 "Terminal marks in legacy portfolio engine are synthetic liquidation, not realized sales.",
 "2022-2026 has been repeatedly researched; no independent OOS or 600-percent return assurance."]}
 rows=[];stress=[]
 for variant in ("MA30_CONTROL","ATR4_CONTROL","ADAPTIVE_FIXED","ADAPTIVE_DYNAMIC"):
  sub=candidates[candidates.variant.eq(variant)].copy()
  executed=ledger[ledger.variant.eq(variant)].copy()
  curve=equity[equity.variant.eq(variant)].copy()
  if sub.empty or executed.empty or curve.empty:raise RuntimeError("Incomplete variant "+variant)
  sub["entry_date"]=pd.to_datetime(sub.entry_date).dt.strftime("%Y-%m-%d")
  sub["exit_date"]=pd.to_datetime(sub.exit_date).dt.strftime("%Y-%m-%d")
  executed["entry_date"]=pd.to_datetime(executed.entry_date).dt.strftime("%Y-%m-%d")
  executed["exit_date"]=pd.to_datetime(executed.exit_date).dt.strftime("%Y-%m-%d")
  key=sub.set_index(["ticker","entry_date"])
  executed=executed.sort_values("exit_date").copy()
  executed["prior_capital"]=executed.capital_after.shift().fillna(INITIAL)
  executed["pnl_krw"]=executed.capital_after-executed.prior_capital
  executed["entry_regime"]=[key.loc[(r.ticker,r.entry_date),"entry_regime"] for r in executed.itertuples()]
  executed["terminal_mark"]=[key.loc[(r.ticker,r.entry_date),"reason"]=="terminal_mark_to_market" for r in executed.itertuples()]
  executed["variant"]=variant
  rows.append(executed)
  gains=executed.loc[executed.pnl_krw>0].sort_values("pnl_krw",ascending=False)
  gross=float(gains.pnl_krw.sum())
  best=gains.iloc[0] if len(gains) else None
  report["strategies"][variant]={
   "trades":len(executed),
   "total_pnl_krw":float(executed.pnl_krw.sum()),
   "terminal_pnl_krw":float(executed.loc[executed.terminal_mark,"pnl_krw"].sum()),
   "terminal_count":int(executed.terminal_mark.sum()),
   "top1_share_of_gross_wins":float(gains.head(1).pnl_krw.sum()/gross) if gross else None,
   "top3_share_of_gross_wins":float(gains.head(3).pnl_krw.sum()/gross) if gross else None,
   "entry_regime_pnl_krw":{str(k):float(z.pnl_krw.sum()) for k,z in executed.groupby("entry_regime")},
   "top_winner":({"ticker":str(best.ticker),"entry_date":str(best.entry_date),"pnl_krw":float(best.pnl_krw)} if best is not None else None)}
  if best is not None:
   # Diagnostic only: remove top winner candidate and replay all future opportunities.
   filt=sub[~((sub.ticker==best.ticker)&(sub.entry_date==best.entry_date))]
   records=filt[["ticker","entry_date","exit_date","entry_price","exit_price","edge","reason"]].to_dict("records")
   perf,_,_=portfolio(records,prices,variant+"_WITHOUT_TOP",rotation=False)
   report["strategies"][variant]["without_top_replay"]={"return":perf["return"],"mdd":perf["daily_close_mdd"],"trades":perf["trades"]}
   stress.append({"variant":variant,"excluded_ticker":best.ticker,"excluded_entry":best.entry_date,**perf})
  # Calendar-year account returns from daily marked equity.
  curve["date"]=pd.to_datetime(curve.date)
  years={};prior=INITIAL
  for year,z in curve.sort_values("date").groupby(curve.date.dt.year):
   end=float(z.equity.iloc[-1]);years[str(year)]=end/prior-1;prior=end
  report["strategies"][variant]["annual_marked_returns"]=years
 pd.concat(rows,ignore_index=True).to_csv(out/"trade_attribution.csv",index=False)
 pd.DataFrame(stress).to_csv(out/"top_winner_stress.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 table="".join(f"<tr><td>{html.escape(k)}</td><td>{v['total_pnl_krw']/10000:,.0f}</td><td>{v['top1_share_of_gross_wins']:.1%}</td><td>{v['without_top_replay']['return']:.2%}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.37</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.37 · Regime Exit Robustness</h1><p>Historical stress test. Ex-post exclusion is NOT investable.</p><table><tr><th>Variant</th><th>P&amp;L (10K KRW)</th><th>Top winner share</th><th>Return without top winner</th></tr>"+table+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","variants":list(report["strategies"])}))
if __name__=="__main__":main()
