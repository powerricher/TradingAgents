"""PRISM v4.63 — frozen Seed A/B ledger audit. Recovery on HOLD.
Historical diagnostics only; no rule changes, no promotion.
"""
import json,html
from pathlib import Path
import pandas as pd,numpy as np
def main():
 out=Path("results/prism-v463");out.mkdir(parents=True,exist_ok=True)
 a=pd.read_csv("results/prism-v433/trade_attribution.csv")
 b=pd.read_csv("results/prism-v457/trade_ledger.csv")
 if not {"variant","ticker","entry_date","exit_date","pnl_krw"}.issubset(a):
  raise RuntimeError("Seed A attribution fields missing")
 if not {"experiment","ticker","entry_date","exit_date","capital_after"}.issubset(b):
  raise RuntimeError("Seed B ledger fields missing")
 a=a[a.variant=="ATR4"].copy()
 b=b[b.experiment=="TECH150_VCP_BREAKOUT_QUALITY_RANK"].copy()
 if a.empty or b.empty:raise RuntimeError("Missing frozen core ledgers")
 def audit(df,kind):
  df=df.sort_values(["exit_date","entry_date"]).copy()
  df["entry_date"]=pd.to_datetime(df.entry_date)
  df["exit_date"]=pd.to_datetime(df.exit_date)
  df["holding_days"]=(df.exit_date-df.entry_date).dt.days
  if kind=="B":
   df["pnl_krw"]=df.capital_after.diff().fillna(df.capital_after-10_000_000.)
  gains=df[df.pnl_krw>0].pnl_krw.sort_values(ascending=False)
  losses=df[df.pnl_krw<0].pnl_krw
  total_positive=float(gains.sum())
  return df,{"trades":len(df),"wins":len(gains),"losses":len(losses),
   "win_rate":float((df.pnl_krw>0).mean()),
   "net_pnl_krw":float(df.pnl_krw.sum()),
   "median_holding_calendar_days":float(df.holding_days.median()),
   "mean_win_krw":float(gains.mean()) if len(gains) else None,
   "mean_loss_krw":float(losses.mean()) if len(losses) else None,
   "top1_gross_win_share":float(gains.head(1).sum()/total_positive) if total_positive else None,
   "top3_gross_win_share":float(gains.head(3).sum()/total_positive) if total_positive else None,
   "terminal_mark_count":int(df.terminal_mark.sum()) if "terminal_mark" in df else None}
 a,am=audit(a,"A");b,bm=audit(b,"B")
 a.to_csv(out/"seed_a_trades.csv",index=False)
 b.to_csv(out/"seed_b_trades.csv",index=False)
 overlaps=[]
 for x in a.itertuples(index=False):
  for y in b.itertuples(index=False):
   start=max(x.entry_date,y.entry_date);end=min(x.exit_date,y.exit_date)
   if end>=start:
    overlaps.append({"seed_a_ticker":x.ticker,"seed_b_ticker":y.ticker,
     "overlap_start":str(start.date()),"overlap_end":str(end.date()),
     "same_ticker":bool(x.ticker==y.ticker),"overlap_calendar_days":(end-start).days+1})
 pd.DataFrame(overlaps,columns=["seed_a_ticker","seed_b_ticker","overlap_start","overlap_end","same_ticker","overlap_calendar_days"]).to_csv(out/"overlaps.csv",index=False)
 report={"version":"v4.63","status":"FROZEN_CORE_AUDIT_NOT_OOS",
 "seed_a":am,"seed_b":bm,"overlap_pairs":len(overlaps),
 "recovery":"HOLD","triangle_wedge":"HOLD",
 "limitations":["Seed A ATR4 from v4.33 frozen ledger, Seed B TECH150 Quality from v4.57 frozen ledger.",
 "This is an attribution audit, not a rerun of signal selection or prices.",
 "P&L for Seed B reconstructed from capital_after ordered by exit date.",
 "Trade overlap counts calendar intervals, not market-day exposures.",
 "Top-winner removal and market-regime attribution not implemented here.",
 "Terminal marks are not necessarily realized profits.",
 "Retrospective universe, historical optimization, and survivorship bias remain.",
 "No automated orders or live strategy changes."]}
 (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
 page="<html lang='ko'><meta charset='utf-8'><title>PRISM v4.63</title><style>body{background:#0d1829;color:#e8f2ff;font:15px system-ui;margin:28px}h1{color:#61e0b8}pre{white-space:pre-wrap}</style><h1>PRISM v4.63 — Core Strategy Audit</h1><p>Seed A ATR4 + Seed B Breakout Quality · Recovery HOLD</p><pre>"+html.escape(json.dumps(report,indent=2,ensure_ascii=False))+"</pre></html>"
 (out/"report.html").write_text(page,encoding="utf-8")
 print(json.dumps({"status":"AUDIT_GENERATED","seed_a_trades":len(a),"seed_b_trades":len(b)}))
if __name__=="__main__":main()
