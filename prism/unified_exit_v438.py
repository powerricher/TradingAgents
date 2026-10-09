"""PRISM v4.38: isolate exit-engine differences in regime adaptive strategy.
Frozen VCP80 entries. Use the EXACT CONFIRM30_1 engine for bull exits,
including its ATR stop and 90-session limit. Research only.
"""
import argparse,json,html
from pathlib import Path
import pandas as pd
from .trend_v48 import load_prices,portfolio
from .trend_v49 import exit_trade as confirmed_exit
from .regime_adaptive_v436 import regimes,adaptive_exit
from .atr_frontier_v433 import exit_atr,CUTOFF

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--scored",default="results/prism-v420/scored_signals.csv");ap.add_argument("--out",default="results/prism-v438");a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored)
 for col in ("date","entry_date","exit_date"):df[col]=pd.to_datetime(df[col]).dt.normalize()
 if df.empty or df.rank_VCP_ALL.isna().any():raise RuntimeError("No frozen VCP80 inputs")
 prices={t:load_prices(t,g.date.min()).loc[lambda x:x.index<=CUTOFF] for t,g in df.groupby("ticker")}
 market=regimes(df.entry_date.min())
 report={"version":"v4.38","status":"HISTORICAL EXIT ENGINE AUDIT / NOT INDEPENDENT OOS",
 "strategies":{},"differences":{},
 "notes":["Bull entries in UNIFIED_FIXED use the exact v4.9 CONFIRM30_1 exit_trade, including ATR3 stop and 90-session max hold.",
 "Non-bull entries use monotone ATR4, matching v4.36 fixed exit.",
 "OLD_FIXED reproduces v4.36 simplified bull MA30 exit and non-bull ATR4.",
 "All candidates and VCP ranks identical, one full-capital position, same transaction cost.",
 "This is an 80-stock exit-engine audit; 100-stock VCP expansion remains a separate experiment.",
 "Terminal valuations are synthetic; repeated historical optimization and survivorship bias remain.",
 "Daily close MDD excludes intraday lows."]}
 alltr=[];allled=[];allcurves=[];by_variant={}
 for mode in ("MA30_ORIGINAL","ATR4_ORIGINAL","OLD_FIXED","UNIFIED_FIXED"):
  rows=[]
  for r in df.itertuples(index=False):
   d=prices[r.ticker];state=market.loc[r.entry_date]
   if state=="UNKNOWN":raise RuntimeError("Unknown regime")
   if mode=="MA30_ORIGINAL":xd,px,reason=r.exit_date,float(r.exit_price),r.reason
   elif mode=="ATR4_ORIGINAL":xd,px,reason=exit_atr(d,r.entry_date,4)
   elif mode=="OLD_FIXED":xd,px,reason=adaptive_exit(d,r.entry_date,market,"FIXED")
   elif state=="BULL":
    signal_ix=d.index.get_loc(r.entry_date)-1
    if signal_ix<0:raise RuntimeError("No preceding signal date")
    z=confirmed_exit(d,d.index[signal_ix],"CONFIRM30_1")
    if z is None:raise RuntimeError("Unresolved unified bull MA30 "+r.ticker)
    e,x,entry,px,reason=z
    if d.index[e]!=r.entry_date:raise RuntimeError("Unified entry mismatch "+r.ticker)
    xd=d.index[x]
   else:xd,px,reason=exit_atr(d,r.entry_date,4)
   if xd<r.entry_date:raise RuntimeError("Invalid exit chronology")
   rows.append({"ticker":r.ticker,"mode":mode,"entry_date":str(r.entry_date.date()),"exit_date":str(xd.date()),
    "entry_price":float(r.entry_price),"exit_price":float(px),"edge":float(r.rank_VCP_ALL),"reason":reason,"regime":state})
  perf,ledger,curve=portfolio(rows,prices,mode,rotation=False)
  if not ledger:raise RuntimeError("Zero-trade "+mode)
  lookup={(x["ticker"],x["entry_date"]):x for x in rows}
  perf["terminal_executed"]=sum(lookup[(x["ticker"],x["entry_date"])]["reason"]=="terminal_mark_to_market" for x in ledger)
  report["strategies"][mode]=perf;by_variant[mode]=pd.DataFrame(rows)
  for x in rows:x["variant"]=mode
  for x in ledger:x["variant"]=mode
  for x in curve:x["variant"]=mode
  alltr+=rows;allled+=ledger;allcurves+=curve
 old=by_variant["OLD_FIXED"].set_index(["ticker","entry_date"])
 new=by_variant["UNIFIED_FIXED"].set_index(["ticker","entry_date"])
 if not old.index.equals(new.index):raise RuntimeError("Candidate identity mismatch")
 diff=old[["exit_date","exit_price","regime"]].join(new[["exit_date","exit_price"]],lsuffix="_old",rsuffix="_unified")
 changed=(diff.exit_date_old!=diff.exit_date_unified)|(abs(diff.exit_price_old-diff.exit_price_unified)>1e-8)
 report["differences"]={"candidate_exits_changed":int(changed.sum()),"bull_changed":int((changed&(diff.regime=="BULL")).sum()),
  "nonbull_changed":int((changed&(diff.regime!="BULL")).sum())}
 if report["differences"]["nonbull_changed"]!=0:raise RuntimeError("Unexpected non-bull exit changes")
 diff[changed].to_csv(out/"changed_exits.csv")
 pd.DataFrame(alltr).to_csv(out/"candidate_exits.csv",index=False)
 pd.DataFrame(allled).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(allcurves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.38</title><style>body{background:#0b1729;color:#edf7ff;font:15px system-ui;padding:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.38 · Unified Exit Audit</h1><p>Frozen VCP80; independent OOS not established.</p><table><tr><th>Strategy</th><th>Return</th><th>MDD</th><th>Trades</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","strategies":{k:v["return"] for k,v in report["strategies"].items()},"differences":report["differences"]},ensure_ascii=False))
if __name__=="__main__":main()
