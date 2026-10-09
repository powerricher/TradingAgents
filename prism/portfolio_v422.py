"""PRISM v4.22 adaptive cash allocation; no leverage.
Compare fixed sleeve 1/3 and dynamic cash-only 2/3/4 position limits.
No rebalancing of open holdings; no same-day proceeds reinvestment.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices
from .portfolio_v421 import replay as fixed_replay
INITIAL=10_000_000.;COST=.001
MODELS={"RIDGE":"rank_RIDGE","VCP_ALL":"rank_VCP_ALL","VCP_VOLATILITY":"rank_VCP_VOLATILITY"}
def dynamic(records,prices,limit,weighted):
 entries={pd.Timestamp(day):g.sort_values(["rank","ticker"],ascending=[False,True]).to_dict("records") for day,g in records.groupby("entry_date")}
 first=min(entries);last=max(pd.Timestamp(x) for x in records.exit_date)
 days=sorted(d for d in set().union(*(set(x.index) for x in prices.values())) if first<=d<=last)
 cash=INITIAL;positions=[];ledger=[];curve=[];skipped=0
 for day in days:
  # Exits at open; cannot use proceeds for new buys until following session.
  sold=[]
  remaining=[]
  for p in positions:
   if pd.Timestamp(p["exit_date"])==day:
    proceeds=p["shares"]*p["exit_price"]*(1-COST/2)
    cash+=proceeds
    ledger.append({"ticker":p["ticker"],"entry_date":p["entry_date"],"exit_date":str(day.date()),"entry_capital":p["entry_capital"],"proceeds":proceeds,"net_return":proceeds/p["entry_capital"]-1,"reason":p["reason"]})
    sold.append(p["ticker"])
   else:remaining.append(p)
  positions=remaining
  pool=entries.get(day,[])
  occupied={p["ticker"] for p in positions}
  available=[p for p in pool if p["ticker"] not in occupied and p["ticker"] not in sold]
  capacity=max(0,limit-len(positions))
  # Respect actual cash before exits today to avoid same-session re-use.
  cash_available=cash-sum(x["proceeds"] for x in ledger if x["exit_date"]==str(day.date()))
  chosen=available[:capacity] if cash_available>0 else []
  if chosen:
   # Equal allocations or positive shifted ranking weights.
   ranks=np.array([float(x["rank"]) for x in chosen])
   if weighted:
    z=ranks-ranks.min()
    weights=(z+0.01)/(z+0.01).sum()
   else:weights=np.full(len(chosen),1/len(chosen))
   # Target account equity / max positions, capped by available cash.
   equity_open=cash
   for p in positions:
    d=prices[p["ticker"]]
    equity_open+=p["shares"]*float(d.loc[day,"Open"])
   budget=min(cash_available,equity_open/limit*len(chosen))
   for item,w in zip(chosen,weights):
    allocation=float(budget*w)
    if allocation<=0:continue
    entry=float(item["entry_price"])
    if entry<=0 or not np.isfinite(entry):raise ValueError("Invalid entry")
    positions.append({**item,"shares":allocation*(1-COST/2)/entry,"entry_capital":allocation})
    cash-=allocation
  skipped+=len(pool)-len(chosen)
  equity=cash
  for p in positions:
   d=prices[p["ticker"]]
   if day not in d.index:raise RuntimeError("Missing close "+p["ticker"]+" "+str(day))
   equity+=p["shares"]*float(d.loc[day,"Close"])
  curve.append({"date":str(day.date()),"equity":float(equity),"cash":float(cash),"positions":len(positions)})
 if positions:raise RuntimeError("Open positions at horizon")
 vals=np.array([r["equity"] for r in curve]);peaks=np.maximum.accumulate(np.r_[INITIAL,vals])[1:]
 return {"final_krw":float(cash),"compound_return":float(cash/INITIAL-1),"daily_close_mdd":float(np.min(vals/peaks-1)),"trades":len(ledger),"skipped":skipped,
 "average_cash_fraction":float(np.mean([r["cash"]/r["equity"] for r in curve]))},ledger,curve
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--scored",default="results/prism-v420/scored_signals.csv");ap.add_argument("--out",default="results/prism-v422");a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored)
 df.entry_date=pd.to_datetime(df.entry_date);df.exit_date=pd.to_datetime(df.exit_date)
 if df.empty:raise RuntimeError("Empty frozen signals")
 prices={t:load_prices(t,g.date.min()) for t,g in df.groupby("ticker")}
 report={"version":"v4.22","status":"EXPLORATORY / NOT INDEPENDENT OOS","strategies":{},"assumptions":[
 "Same frozen TECH80 candidates and MA30 exits as v4.20/4.21.",
 "Dynamic position count 2, 3, or 4; actual cash only; no margin.",
 "No rebalancing or liquidation of held names.",
 "Entry orders at next-session open; same-session exit proceeds not reused.",
 "Cash budget is at most equity divided by position limit per new position.",
 "Weighted strategy splits only the day's buy budget among simultaneous signals; it does not overweight a single signal when only one is present.",
 "Same historical period repeatedly researched; survivorship and model-selection biases remain.",
 "Daily-close MDD omits intraday troughs."]}
 all_ledger=[];all_curve=[]
 for model,col in MODELS.items():
  g=df[["ticker","entry_date","exit_date","entry_price","exit_price","reason",col]].rename(columns={col:"rank"})
  for n in (1,3):
   perf,led,curve=fixed_replay(g,prices,n)
   key=f"{model}_FIXED_{n}S";report["strategies"][key]=perf
   for r in led:r["strategy"]=key
   for r in curve:r["strategy"]=key
   all_ledger+=led;all_curve+=curve
  for n in (2,3,4):
   for weighted in (False,True):
    key=f"{model}_DYNAMIC_{n}S_"+("WEIGHTED" if weighted else "EQUAL")
    perf,led,curve=dynamic(g,prices,n,weighted)
    report["strategies"][key]=perf
    for r in led:r["strategy"]=key
    for r in curve:r["strategy"]=key
    all_ledger+=led;all_curve+=curve
 pd.DataFrame(all_ledger).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(all_curve).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 rows="".join(f"<tr><td>{html.escape(k)}</td><td>{v['compound_return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v.get('executed',v.get('trades'))}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.22</title><style>body{background:#0b1729;color:#edf7ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM v4.22 — Adaptive Capital Allocation</h1><p>Historical research, not OOS. 10M KRW initial, fixed MA30 exit.</p><table><tr><th>Variant</th><th>Return</th><th>Daily MDD</th><th>Trades</th></tr>"+rows+"</table>")
 print(json.dumps({"status":"generated","strategies":len(report["strategies"])}))
if __name__=="__main__":main()
