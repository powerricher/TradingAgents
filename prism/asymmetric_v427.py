"""PRISM v4.27: winner-first chronological ranking and fast-loss exit ablation.
No outcome leakage into signal-time features; exit rules use daily OHLC with
next-open execution for end-of-day decisions and gap-aware stops.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .ranking_v417 import FEATURES
from .trend_v48 import load_prices,portfolio,STOP_SLIPPAGE
FEATS=FEATURES+["contraction_ratio","range_contraction","volume_dryup"]
def cut_exit(d,entry_date,original_exit,entry_price,mode):
 e=d.index.get_loc(entry_date);x=d.index.get_loc(original_exit)
 if x<=e:raise ValueError("Bad holding window")
 if mode=="MA30":return original_exit,None,None
 # Stop is known at entry, applied from next session to avoid unknown
 # entry-session intraday ordering. Gap fills at actual open.
 atr=(pd.concat([(d.High-d.Low).abs(),(d.High-d.Close.shift()).abs(),(d.Low-d.Close.shift()).abs()],axis=1).max(axis=1)).rolling(14).mean()
 stop=entry_price*.95 if mode=="CUT5" else entry_price-2*float(atr.iloc[e-1])
 if not np.isfinite(stop):return original_exit,None,None
 for j in range(e+1,x+1):
  op=float(d.Open.iloc[j]);low=float(d.Low.iloc[j])
  if op<=stop:return d.index[j],op,"gap_cut"
  if low<=stop:return d.index[j],stop*(1-STOP_SLIPPAGE),"stop_touch_stress"
  if mode=="FAIL5" and j==e+5 and float(d.Close.iloc[j])<=entry_price and j+1<=x:
   return d.index[j+1],float(d.Open.iloc[j+1]),"failed_to_advance"
 return original_exit,None,None
def main():
 p=argparse.ArgumentParser();p.add_argument("--scored",default="results/prism-v420/scored_signals.csv");p.add_argument("--out",default="results/prism-v427");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored)
 for k in ("date","entry_date","exit_date"):df[k]=pd.to_datetime(df[k])
 df=df.sort_values(["date","ticker"]).reset_index(drop=True)
 prices={t:load_prices(t,g.date.min()) for t,g in df.groupby("ticker")}
 df["p_big_winner"]=np.nan
 for month,g in df.groupby(df.date.dt.to_period("M"),sort=True):
  tr=df[df.exit_date<month.start_time-pd.Timedelta(days=5)]
  if len(tr)<50 or (tr.ret>=.15).nunique()<2:
   df.loc[g.index,"p_big_winner"]=.5
   continue
  model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000))
  model.fit(tr[FEATS],(tr.ret>=.15).astype(int))
  df.loc[g.index,"p_big_winner"]=model.predict_proba(g[FEATS])[:,1]
 if df.p_big_winner.isna().any():raise RuntimeError("Unscored winners")
 df["rank_WINNER"]=df.rank_VCP_ALL+.03*(df.p_big_winner-.5)
 variants={"VCP_MA30":("rank_VCP_ALL","MA30"),"WINNER_MA30":("rank_WINNER","MA30"),
  "VCP_CUT5":("rank_VCP_ALL","CUT5"),"WINNER_CUT5":("rank_WINNER","CUT5"),
  "VCP_FAIL5":("rank_VCP_ALL","FAIL5"),"WINNER_FAIL5":("rank_WINNER","FAIL5"),
  "VCP_ATR2":("rank_VCP_ALL","ATR2"),"WINNER_ATR2":("rank_WINNER","ATR2")}
 report={"version":"v4.27","status":"HISTORICAL RESEARCH / NOT INDEPENDENT OOS","strategies":{},
 "notes":["Winner label is historical MA30 net return >=15%; trained only on previously closed trades.",
 "Winner probability is a ranking adjustment, not a guarantee or hard exclusion.",
 "Exit stops are applied only from the session after next-open entry to avoid same-day OHLC ordering ambiguity.",
 "Fixed -5% or 2ATR stop is initialized at entry, not optimized during the trade.",
 "Gap exits use next available open; stop touches assume adverse 20bps slippage.",
 "FAIL5 exits next session open if day-5 close does not exceed entry; if MA30 exit occurs earlier, keep MA30 exit.",
 "Same retrospectively selected 80-stock sample has been researched repeatedly; no independent OOS.",
 "Daily-close MDD excludes intraday troughs."]}
 ledgers=[];curves=[];trades=[]
 for name,(rank,mode) in variants.items():
  rows=[]
  for r in df.itertuples(index=False):
   d=prices[r.ticker]
   if mode=="MA30":xd,px,reason=r.exit_date,None,None
   else:xd,px,reason=cut_exit(d,r.entry_date,r.exit_date,float(r.entry_price),mode)
   if px is None:px=float(r.exit_price);reason=r.reason
   rows.append({"ticker":r.ticker,"mode":name,"entry_date":str(r.entry_date.date()),"exit_date":str(xd.date()),"entry_price":float(r.entry_price),"exit_price":px,"edge":float(getattr(r,rank)),"reason":reason})
  perf,ledger,curve=portfolio(rows,prices,name,rotation=False)
  if perf["trades"]==0:raise RuntimeError("Zero trades")
  report["strategies"][name]=perf
  for x in rows:x["variant"]=name
  for x in ledger:x["variant"]=name
  for x in curve:x["variant"]=name
  trades+=rows;ledgers+=ledger;curves+=curve
 df.to_csv(out/"scored_signals.csv",index=False)
 pd.DataFrame(trades).to_csv(out/"candidate_exits.csv",index=False)
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.27</title><style>body{background:#0b1729;color:#edf7ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.27 · Asymmetric Winner & Fast Loss</h1><p>TECH80 · one slot · 10M KRW · historical research only</p><table><tr><th>Strategy</th><th>Compound return</th><th>Daily MDD</th><th>Trades</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","strategies":{k:v["return"] for k,v in report["strategies"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
