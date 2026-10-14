"""PRISM v4.68: fixed Seed C signals, conservative daily-OHLC exits.
No intraday ordering inference; if stop and target both touch, stop first.
Research-only, no signal tuning and no independent historical OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
from .recovery_portfolio_v462_fix import portfolio_same_day
START=10_000_000.
EXITS={"TP5_SL3":(.05,.03),"TP6_SL3":(.06,.03),"TP8_SL4":(.08,.04)}
HOLDS=(1,3,5)
def exit_path(d,entry,entry_price,tp,sl,hold):
 ix=d.index.get_loc(pd.Timestamp(entry));target=entry_price*(1+tp);stop=entry_price*(1-sl)
 for j in range(ix, min(ix+hold,len(d)-1)+1):
  op,hi,lo=map(float,(d.Open.iloc[j],d.High.iloc[j],d.Low.iloc[j]))
  if j>ix and op<=stop:return str(d.index[j].date()),op,"gap_stop"
  if j>ix and op>=target:return str(d.index[j].date()),op,"gap_target"
  # Conservative assumption when both levels touched on the same daily bar.
  if lo<=stop:return str(d.index[j].date()),stop*.998,"stop_touch"
  if hi>=target:return str(d.index[j].date()),target*.998,"target_touch"
  if j-ix>=hold:return str(d.index[j].date()),float(d.Close.iloc[j]),"time_close"
 return None
def main():
 out=Path("results/prism-v468");out.mkdir(parents=True,exist_ok=True)
 signals=pd.read_csv("results/prism-v467/signal_forward_returns.csv")
 required={"ticker","detector","signal_date","entry_date","entry_open","signal_rvol","prior_dollar_volume","prior_atr_pct"}
 if signals.empty or not required.issubset(signals):raise RuntimeError("Missing frozen Seed C signals")
 prices={};errors=[]
 for t in sorted(signals.ticker.unique()):
  try:
   d=yf.download(t,start="2021-01-01",end="2026-10-07",auto_adjust=True,progress=False,threads=False)
   if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
   d=d.dropna(subset=["Open","High","Low","Close"]).copy()
   d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
   if d.empty:raise ValueError("Empty OHLC")
   prices[t]=d
  except Exception as e:errors.append({"ticker":t,"error":str(e)})
 if errors:raise RuntimeError("Missing price histories: "+str(errors[:10]))
 allrows=[];missing=[];audit=[]
 for r in signals.itertuples(index=False):
  d=prices[r.ticker];day=pd.Timestamp(r.entry_date)
  if day not in d.index:
   missing.append({"ticker":r.ticker,"entry_date":r.entry_date,"reason":"MISSING_ENTRY"});continue
  price=float(d.loc[day,"Open"])
  if abs(price/float(r.entry_open)-1)>0.001:
   missing.append({"ticker":r.ticker,"entry_date":r.entry_date,"reason":"PRICE_SNAPSHOT_MISMATCH"});continue
  for name,(tp,sl) in EXITS.items():
   for hold in HOLDS:
    result=exit_path(d,day,price,tp,sl,hold)
    if result is None:
     missing.append({"ticker":r.ticker,"entry_date":r.entry_date,"reason":"UNRESOLVED_EXIT"});continue
    xd,xp,reason=result
    allrows.append({"ticker":r.ticker,"detector":r.detector,"signal_date":r.signal_date,
     "entry_date":r.entry_date,"entry_price":price,"exit_date":xd,"exit_price":xp,
     "reason":reason,"exit_rule":name,"hold":hold,
     "edge":float(r.signal_rvol),"gross_return":xp/price-1})
 df=pd.DataFrame(allrows)
 df.to_csv(out/"candidate_exit_scenarios.csv",index=False)
 pd.DataFrame(missing).to_csv(out/"unresolved.csv",index=False)
 if missing:raise RuntimeError("Unresolved or price-mismatched candidates; refuse comparison")
 report={"version":"v4.68","status":"EXPLORATORY DAILY_OHLC EXECUTION_STRESS_NOT_OOS",
 "signals":len(signals),"strategies":{},"limitations":[
 "Historical v4.67 signals are frozen; all exits use the same next-open entry.",
 "If stop and target both touch the same bar, stop is assumed first.",
 "Entry-day stop/target processing supported by corrected v4.62 portfolio engine.",
 "Time exit is modeled at holding-day close; execution at exact close is not assured.",
 "Gap opens can be worse than stop or better than target; modeled without intraday sequencing.",
 "Round-trip portfolio cost 10bps; stop/target touch additional 20bps haircut.",
 "One position per Seed C; signals ranked by RVOL, not predictive calibration.",
 "D+5 historical candidate discovery already reused for research, NOT untouched OOS.",
 "Current-listed sample omits delisted names and is not point-in-time.",
 "Daily-close MDD omits intraday troughs; no alerts or orders."]}
 ledgers=[];curves=[]
 for detector in sorted(df.detector.unique()):
  for rule in EXITS:
   for hold in HOLDS:
    z=df[(df.detector==detector)&(df.exit_rule==rule)&(df.hold==hold)]
    if len(z)!=len(signals[signals.detector==detector]):
     raise RuntimeError("Candidate count mismatch "+detector+" "+rule)
    name=f"{detector}_{rule}_H{hold}"
    records=z[["ticker","entry_date","entry_price","exit_date","exit_price","edge","reason"]].to_dict("records")
    perf,ledger,curve=portfolio_same_day(records,prices,name)
    if perf["trades"]<=0:raise RuntimeError("No trades "+name)
    report["strategies"][name]={**perf,"eligible":len(records)}
    for x in ledger:x["strategy"]=name
    for x in curve:x["strategy"]=name
    ledgers+=ledger;curves+=curve
 if len(report["strategies"])!=27:raise RuntimeError("Expected exactly 27 strategy combinations")
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
 lines="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td><td>{v['same_day_exits']}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.68</title><style>body{background:#10192c;color:#eaf2ff;font:14px system-ui;margin:25px}h1{color:#5ce6b9}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #38485e}</style><h1>PRISM Seed C · Exit Frontier</h1><p>Conservative OHLC stop-first simulation; historical research only</p><table><tr><th>Strategy</th><th>Return</th><th>MDD</th><th>Trades</th><th>Same-day exits</th></tr>"+lines+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"VALID","strategies":len(report["strategies"]),"signals":len(signals)}))
if __name__=="__main__":main()
