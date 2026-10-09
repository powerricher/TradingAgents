"""PRISM v4.25: Opportunity Cost Audit (no hindsight trading).
Compare held position vs signals rejected due to capital lock-up, using same
future close dates as diagnostic labels. Do not promote switching based on this.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--scored",default="results/prism-v420/scored_signals.csv")
 p.add_argument("--ledger",default="results/prism-v424/ledger.csv")
 p.add_argument("--out",default="results/prism-v425")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 signals=pd.read_csv(a.scored)
 ledger=pd.read_csv(a.ledger)
 held=ledger[ledger.strategy=="VCP_CONTROL"].copy()
 if held.empty:raise RuntimeError("No control trades; abort")
 for d in (signals,held):
  for c in ("entry_date","exit_date"):d[c]=pd.to_datetime(d[c]).dt.normalize()
 signals["date"]=pd.to_datetime(signals.date).dt.normalize()
 if signals[["ticker","entry_date","rank_VCP_ALL"]].isna().any().any():raise RuntimeError("Incomplete candidate data")
 prices={}
 for ticker,g in signals.groupby("ticker"):prices[ticker]=load_prices(ticker,g.date.min())
 held=held.sort_values("entry_date")
 rows=[];missing=[]
 # Candidates are generated after signal-day close and enter next open.
 # At candidate entry open, current position may already have an MA30 exit.
 for candidate in signals.itertuples(index=False):
  when=candidate.entry_date
  active=held[(held.entry_date<=when)&(held.exit_date>when)]
  if active.empty:continue
  h=active.iloc[0]
  if h.ticker==candidate.ticker:continue
  old=prices[h.ticker];new=prices[candidate.ticker]
  if when not in old.index or when not in new.index:
   missing.append((candidate.ticker,str(when.date())));continue
  old_entry=float(old.loc[when,"Open"])
  new_entry=float(new.loc[when,"Open"])
  if old_entry<=0 or new_entry<=0:raise RuntimeError("Invalid OHLC")
  item={"signal_date":str(candidate.date.date()),"entry_date":str(when.date()),
   "held_ticker":h.ticker,"candidate_ticker":candidate.ticker,
   "candidate_rank":float(candidate.rank_VCP_ALL),"held_exit_date":str(h.exit_date.date())}
  for n in (5,10,20):
   # Compare same future session close, only if both stocks have n sessions.
   oi=old.index.get_loc(when);ni=new.index.get_loc(when)
   if oi+n>=len(old) or ni+n>=len(new):
    item[f"candidate_d{n}"]=np.nan;item[f"held_d{n}"]=np.nan;item[f"switch_advantage_d{n}"]=np.nan;continue
   end_new=new.index[ni+n];end_old=old.index[oi+n]
   # Must compare at the same calendar date, not independently indexed sessions.
   if end_new!=end_old:
    item[f"candidate_d{n}"]=np.nan;item[f"held_d{n}"]=np.nan;item[f"switch_advantage_d{n}"]=np.nan;continue
   cr=float(new.Close.iloc[ni+n])/new_entry-1
   hr=float(old.Close.iloc[oi+n])/old_entry-1
   item[f"candidate_d{n}"]=cr;item[f"held_d{n}"]=hr
   item[f"switch_advantage_d{n}"]=cr-hr-.001
  rows.append(item)
 if missing:raise RuntimeError("Missing daily price observations "+str(missing[:5]))
 audit=pd.DataFrame(rows)
 if audit.empty:raise RuntimeError("No overlapping distinct ticker signals")
 summary={"version":"v4.25","status":"RETROSPECTIVE OPPORTUNITY AUDIT ONLY","control_trades":len(held),
  "overlapping_candidates":len(audit),"unique_competition_dates":int(audit.entry_date.nunique()),
  "horizons":{},"limitations":[
  "The candidate and held comparison is an ex-post fixed-horizon price diagnostic, not an executable switch portfolio.",
  "Both returns are measured from the same candidate entry open to the same future close; the held stock may have exited under MA30 before the horizon.",
  "Subtracts a flat 10bps switching penalty for illustration, not a full execution simulation.",
  "Overlapping candidate-day observations are dependent and cannot be added as portfolio profit.",
  "No contemporaneous held-position score is computed; no switch decision engine has been trained.",
  "Same historical 80-stock universe and repeated model development; no independent OOS.",
  "Existing portfolio controls and exit assumptions remain unchanged."]}
 for n in (5,10,20):
  x=audit[f"switch_advantage_d{n}"].dropna()
  summary["horizons"][str(n)]={"evaluable":len(x),"candidate_better_after_cost":int((x>0).sum()),
    "better_share":float((x>0).mean()) if len(x) else None,
    "mean_ex_post_advantage":float(x.mean()) if len(x) else None,
    "median_ex_post_advantage":float(x.median()) if len(x) else None}
 audit.to_csv(out/"overlapping_candidates.csv",index=False)
 (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 lines="".join(f"<tr><td>D+{n}</td><td>{v['evaluable']}</td><td>{v['better_share']:.1%}</td><td>{v['mean_ex_post_advantage']:.2%}</td></tr>" for n,v in summary["horizons"].items() if v["evaluable"])
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.25</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;padding:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.25 · Opportunity Cost Audit</h1><p>Retrospective, non-tradable counterfactuals. Do not interpret as realized profits.</p><table><tr><th>Horizon</th><th>Observations</th><th>Candidate better</th><th>Mean advantage</th></tr>"+lines+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","horizons":summary["horizons"]},ensure_ascii=False))
if __name__=="__main__":main()
