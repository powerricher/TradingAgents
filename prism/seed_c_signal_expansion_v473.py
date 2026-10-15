"""PRISM v4.73 C1 signal expansion and quality audit.
Frozen 1000-symbol snapshot; historical research, not OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
from .seed_c_discovery_v467 import indicators
from .seed_c_execution_v468 import exit_path
from .seed_c_multi_position_v471 import simulate
from .atr_frontier_v433 import CUTOFF
MODES=("BASE","COMPRESSION_80","RVOL_15","BREAKOUT_995")
START=10_000_000.
def main():
 out=Path("results/prism-v473");out.mkdir(parents=True,exist_ok=True)
 names=pd.read_csv("results/prism-v467/universe_snapshot.csv").ticker.astype(str).tolist()
 frozen=pd.read_csv("results/prism-v468/candidate_exit_scenarios.csv")
 frozen=frozen[(frozen.detector=="C1_VOL_EXPANSION")&(frozen.exit_rule=="TP8_SL4")&(frozen.hold==5)]
 ref=pd.read_csv("results/prism-v471/daily_equity.csv")
 ref=ref[ref.strategy=="FOUR_25"].sort_values("date")
 if len(names)!=1000 or frozen.empty or ref.empty:raise RuntimeError("Missing baseline inputs")
 prices={};errors=[];records=[]
 for batch_start in range(0,len(names),25):
  batch=names[batch_start:batch_start+25]
  try:data=yf.download(batch,start="2021-01-01",end="2026-10-07",auto_adjust=True,progress=False,threads=False,group_by="ticker")
  except Exception as exc:
   errors.extend({"ticker":t,"error":str(exc)} for t in batch);continue
  for t in batch:
   try:
    d=data[t] if isinstance(data.columns,pd.MultiIndex) else data
    d=d.dropna(subset=["Open","High","Low","Close","Volume"]).copy()
    if len(d)<80:continue
    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    prices[t]=d
    c=d.Close.astype(float);h=d.High.astype(float);l=d.Low.astype(float);v=d.Volume.astype(float)
    prev=c.shift(1);ret=c.pct_change()
    dollar=(c*v).shift(1).rolling(20,min_periods=20).mean()
    tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    atrpct=tr.rolling(14,min_periods=14).mean()/c
    rvol=v/v.shift(1).rolling(20,min_periods=20).mean()
    compression=ret.shift(1).rolling(5).std()/ret.shift(1).rolling(20).std()
    hi20=h.shift(1).rolling(20).max()
    eligible=(prev>=5)&(dollar>=20_000_000)&(atrpct.shift(1)>=.04)
    common=(ret>.03)&eligible
    masks={
     "BASE":common&(compression<.65)&(c>hi20)&(rvol>=1.8),
     "COMPRESSION_80":common&(compression<.80)&(c>hi20)&(rvol>=1.8),
     "RVOL_15":common&(compression<.65)&(c>hi20)&(rvol>=1.5),
     "BREAKOUT_995":common&(compression<.65)&(c>hi20*.995)&(rvol>=1.8)}
    for mode,mask in masks.items():
     valid=mask.fillna(False)&(d.index>=pd.Timestamp("2022-01-01"))
     for i in np.flatnonzero(valid.to_numpy()):
      if i+6>=len(d):continue
      entry=d.index[i+1];ep=float(d.Open.iloc[i+1])
      if not np.isfinite(ep) or ep<=0:continue
      z=exit_path(d,entry,ep,.08,.04,5)
      if z is None:continue
      xd,xp,reason=z
      records.append({"ticker":t,"mode":mode,"signal_date":str(d.index[i].date()),
       "entry_date":str(entry.date()),"entry_price":ep,"exit_date":xd,
       "exit_price":xp,"edge":float(rvol.iloc[i]),"reason":reason,
       "signal_rvol":float(rvol.iloc[i]),"prior_atr_pct":float(atrpct.iloc[i-1]),
       "breakout_strength":float(c.iloc[i]/hi20.iloc[i]-1),
       "gross_return":float(xp/ep-1)})
   except Exception as exc:errors.append({"ticker":t,"error":str(exc)})
  print("Processed",min(batch_start+25,len(names)),"/",len(names),flush=True)
 df=pd.DataFrame(records)
 if df.empty:raise RuntimeError("No candidate signals")
 df.to_csv(out/"candidate_exits.csv",index=False)
 (out/"coverage_errors.json").write_text(json.dumps(errors,indent=2))
 # Exact baseline identity and price match against the prior frozen study.
 old=frozen[["ticker","entry_date","entry_price","exit_date","exit_price"]].sort_values(["ticker","entry_date"]).reset_index(drop=True)
 new=df[df["mode"]=="BASE"][["ticker","entry_date","entry_price","exit_date","exit_price"]].sort_values(["ticker","entry_date"]).reset_index(drop=True)
 if len(old)!=len(new) or not old[["ticker","entry_date","exit_date"]].equals(new[["ticker","entry_date","exit_date"]]):
  raise RuntimeError("Baseline signal identities diverged from v4.68")
 if not np.allclose(old[["entry_price","exit_price"]],new[["entry_price","exit_price"]],rtol=.001,atol=.001):
  raise RuntimeError("Baseline price snapshot differs")
 calendar=[pd.Timestamp(x) for x in pd.to_datetime(ref.date).dt.normalize().unique()]
 report={"version":"v4.73","status":"HISTORICAL C1 SIGNAL EXPANSION NOT OOS",
  "coverage":{"requested":1000,"available":len(prices),"errors":len(errors)},
  "strategies":{},"limitations":[
   "C1 signal thresholds relaxed one at a time; exit TP8 SL4 H5 and FOUR_25 fixed.",
   "Original BASE identities and OHLC must reproduce v4.68 within 0.1%.",
   "Selection remains RVOL; new candidate ranking is descriptive only.",
   "Current-listed universe excludes delisted names, not point-in-time.",
   "Daily OHLC stop-first exits; no intraday ordering or execution assurance.",
   "Repeated historical exploration is not untouched out-of-sample.",
   "No live trading, alerts or automatic strategy promotion."]}
 ledger_all=[];curve_all=[]
 for mode in MODES:
  z=df[df["mode"]==mode].copy()
  rows=z[["ticker","entry_date","exit_date","entry_price","exit_price","edge","reason"]].to_dict("records")
  perf,ledger,curve=simulate(rows,calendar,prices,4,.25)
  v=np.asarray([x["equity"] for x in curve])
  report["strategies"][mode]={**perf,"candidate_count":len(z),
   "mean_invested":float(np.mean([1-x["cash"]/x["equity"] for x in curve])),
   "signal_days":int(z.entry_date.nunique()),
   "mean_candidate_return":float(z.gross_return.mean())}
  for r in ledger:r["mode"]=mode
  for r in curve:r["mode"]=mode
  ledger_all+=ledger;curve_all+=curve
 target=float(ref.equity.iloc[-1])
 if abs(report["strategies"]["BASE"]["final_krw"]-target)>max(1.,target*.00001):
  raise RuntimeError("Portfolio baseline does not reproduce v4.71")
 pd.DataFrame(ledger_all).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curve_all).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
 lines="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['mdd']:.2%}</td><td>{v['trades']}</td><td>{v['mean_invested']:.1%}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.73</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.73 · Signal Expansion</h1><p>Frozen C1 baseline, 3 isolated threshold expansions.</p><table><tr><th>Mode</th><th>Return</th><th>MDD</th><th>Trades</th><th>Avg invested</th></tr>"+lines+"</table>",encoding="utf-8")
 print(json.dumps({"status":"VALID","variants":len(report["strategies"])}))
if __name__=="__main__":main()
