"""PRISM v4.28: audit actual entry competition, missed winners, and FAIL5 exits.
No hindsight signals are fed back into trading. Audit only.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd

def metrics(x):
 x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
 return {"n":int(len(x)),"mean":float(x.mean()) if len(x) else None,"median":float(np.median(x)) if len(x) else None}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--scored",default="results/prism-v427/scored_signals.csv")
 p.add_argument("--candidate-exits",default="results/prism-v427/candidate_exits.csv")
 p.add_argument("--ledger",default="results/prism-v427/ledger.csv")
 p.add_argument("--out",default="results/prism-v428")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 scored=pd.read_csv(a.scored)
 exits=pd.read_csv(a.candidate_exits)
 ledger=pd.read_csv(a.ledger)
 base=exits[exits.variant.eq("VCP_MA30")].copy()
 fail=exits[exits.variant.eq("VCP_FAIL5")].copy()
 held=ledger[ledger.variant.eq("VCP_MA30")].copy()
 held_fail=ledger[ledger.variant.eq("VCP_FAIL5")].copy()
 if not len(base) or not len(held):raise RuntimeError("Missing baseline")
 if len(base)!=len(scored) or len(fail)!=len(scored):raise RuntimeError("Candidate coverage mismatch")
 for frame in (scored,base,fail,held,held_fail):
  for c in ("entry_date","exit_date"):
   if c in frame:frame[c]=pd.to_datetime(frame[c]).dt.normalize()
 if "date" in scored:scored.date=pd.to_datetime(scored.date).dt.normalize()
 base["candidate_net_return"]=(base.exit_price/base.entry_price)*(1-.001)-1
 fail["candidate_net_return"]=(fail.exit_price/fail.entry_price)*(1-.001)-1
 base=base.merge(scored[["ticker","entry_date","rank_VCP_ALL","rank_WINNER","p_big_winner"]],on=["ticker","entry_date"],validate="one_to_one")
 held_keys=held[["ticker","entry_date"]].drop_duplicates()
 base=base.merge(held_keys.assign(executed=True),on=["ticker","entry_date"],how="left",validate="one_to_one")
 base["executed"]=base.executed.fillna(False).astype(bool)
 if base.executed.sum()!=len(held):raise RuntimeError("Baseline executed trade join mismatch")
 # Same-day competition when portfolio is free. Do not mix in signals blocked
 # by an already open position; those are a separate capital-lockup class.
 comparisons=[];rank_days=[]
 for dt,g in base.groupby("entry_date"):
  if len(g)<2:continue
  selected=g[g.executed]
  if len(selected)==1:
   winner=selected.iloc[0]
   for r in g[~g.executed].itertuples(index=False):
    comparisons.append({"entry_date":str(dt.date()),"chosen":winner.ticker,"alternative":r.ticker,
      "chosen_rank":float(winner.rank_VCP_ALL),"alternative_rank":float(r.rank_VCP_ALL),
      "chosen_return":float(winner.candidate_net_return),"alternative_return":float(r.candidate_net_return),
      "ex_post_advantage":float(r.candidate_net_return-winner.candidate_net_return),
      "alternative_big_winner":bool(r.candidate_net_return>=.15)})
   rank_days.append({"entry_date":str(dt.date()),"candidates":len(g),
     "chosen_is_best_ex_post":bool(winner.candidate_net_return>=g.candidate_net_return.max()-1e-12),
     "top_rank_winner":bool(winner.candidate_net_return>=.15),
     "missed_big_winners":int(((g.candidate_net_return>=.15)&(~g.executed)).sum()),
     "spearman":float(g.rank_VCP_ALL.corr(g.candidate_net_return,method="spearman")) if g.rank_VCP_ALL.nunique()>1 and g.candidate_net_return.nunique()>1 else None})
 # Compare completed trades across exit policies by original candidate key.
 # This is an ex-post candidate-level contrast, not causal portfolio attribution.
 joined=base[["ticker","entry_date","candidate_net_return","executed"]].merge(
  fail[["ticker","entry_date","exit_date","candidate_net_return","reason"]].rename(columns={
   "candidate_net_return":"fail5_net_return","exit_date":"fail5_exit_date","reason":"fail5_reason"}),
  on=["ticker","entry_date"],validate="one_to_one")
 joined["delta_fail5_vs_ma30"]=joined.fail5_net_return-joined.candidate_net_return
 changed=joined[joined.fail5_exit_date.ne(base.set_index(["ticker","entry_date"]).loc[
  pd.MultiIndex.from_frame(joined[["ticker","entry_date"]]),"exit_date"].to_numpy())].copy()
 # Explicitly identify actually executed original trades; never infer that
 # alternative strategy would execute the same sequence after early exits.
 original=joined[joined.executed]
 changed_actual=changed[changed.executed]
 comp=pd.DataFrame(comparisons)
 daily=pd.DataFrame(rank_days)
 report={"version":"v4.28","status":"DIAGNOSTIC ONLY / NOT INDEPENDENT OOS",
 "baseline_executed":len(held),"candidate_signals":len(base),
 "same_day_competition":{"days_with_executed_competition":len(daily),
  "alternative_pairs":len(comp),
  "chosen_best_share":float(daily.chosen_is_best_ex_post.mean()) if len(daily) else None,
  "missed_15pct_winner_pairs":int(comp.alternative_big_winner.sum()) if len(comp) else 0,
  "positive_alternative_advantage_share":float((comp.ex_post_advantage>0).mean()) if len(comp) else None,
  "alternative_advantage":metrics(comp.ex_post_advantage) if len(comp) else metrics([]),
  "daily_spearman":metrics(daily.spearman) if len(daily) else metrics([])},
 "fail5_diagnostics":{"candidate_exits_changed":len(changed),
  "candidate_delta":metrics(changed.delta_fail5_vs_ma30),
  "original_executed_exits_changed":len(changed_actual),
  "original_executed_delta":metrics(changed_actual.delta_fail5_vs_ma30),
  "original_executed_saved_loss_count":int((changed_actual.delta_fail5_vs_ma30>0).sum()),
  "original_executed_missed_gain_count":int((changed_actual.delta_fail5_vs_ma30<0).sum()),
  "actual_fail5_portfolio_trades":len(held_fail)},
 "limitations":[
  "Alternative candidate returns are each candidate's hypothetical MA30 trade return, not same-horizon paired returns.",
  "Same-day competition includes only dates where baseline actually entered a trade and another candidate competed.",
  "Signals blocked while capital was already occupied are NOT included in same-day ranking metrics; see v4.25 for separate lockup audit.",
  "Counterfactual candidates overlap in time and cannot be summed as attainable profit.",
  "FAIL5 candidate exit differences do not represent full-portfolio counterfactual P&L because earlier exits change later entries.",
  "No future outcome is used to create or modify a live trading signal in this audit.",
  "The 2022-2026 sample was repeatedly studied and has universe survivorship bias; independent forward testing is required."]}
 comp.to_csv(out/"missed_winner_pairs.csv",index=False)
 daily.to_csv(out/"competition_days.csv",index=False)
 joined.to_csv(out/"fail5_candidate_attribution.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.28</title><style>body{background:#0b1729;color:#edf7ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.28 · Selection Error Intelligence</h1><p>Historical diagnostic only; not an executable strategy.</p><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","competition_days":len(daily),"fail5_changed":len(changed)}))
if __name__=="__main__":main()
