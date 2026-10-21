"""PRISM v4.79 Seed C: configurable 200-2500-name discovery research.
Current exchange listing is NOT a point-in-time historical universe.
Signals at completed close; next-session open entry; forward returns diagnostic only.
"""
import io,json,html,hashlib,time,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd,requests,yfinance as yf
from .universe_v464 import UNIVERSE as CORE200
END="2026-10-07"
HORIZONS=(1,3,5)
def universe():
 s=requests.Session()
 urls=[
 ("NASDAQ","https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"),
 ("OTHER","https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")]
 tickers=set();errors=[]
 for market,url in urls:
  try:
   r=s.get(url,timeout=25);r.raise_for_status()
   table=pd.read_csv(io.StringIO(r.text),sep="|",dtype=str)
   if market=="NASDAQ":
    table=table[(table["Test Issue"]=="N")&(table["ETF"]=="N")]
    names=table["Security Name"].fillna("")
    symbols=table["Symbol"]
   else:
    table=table[(table["Test Issue"]=="N")&(table["ETF"]=="N")&(table["Exchange"].isin(["N","A"]))]
    names=table["Security Name"].fillna("")
    symbols=table["ACT Symbol"]
   mask=~names.str.contains("warrant|preferred|depositary|unit|right|notes due|bond|etf|etn|fund|trust",case=False,regex=True)
   for t in symbols[mask].dropna():
    if t.isascii() and t.replace(".","").replace("-","").isalnum() and len(t)<=6:
     tickers.add(t.replace(".","-"))
  except Exception as e:errors.append({"source":market,"error":str(e)})
 if errors:raise RuntimeError("Exchange directory unavailable: "+json.dumps(errors))
 # Deterministic current-listing sample; historical survivorship bias remains.
 # This is a research sample, NOT the top 1000 liquid names.
 pool=sorted(tickers-set(CORE200),key=lambda t:hashlib.sha256(t.encode()).hexdigest())
 target=int(os.getenv('PRISM_UNIVERSE_SIZE','1000'))
 if target<200 or target>2500:raise RuntimeError('Universe size must be 200..2500')
 if len(pool)<target-len(CORE200):raise RuntimeError('Insufficient exchange-listed additions')
 return list(CORE200)+pool[:target-len(CORE200)],len(tickers)
def indicators(d):
 c=d.Close.astype(float);h=d.High.astype(float);l=d.Low.astype(float);v=d.Volume.astype(float)
 prev=c.shift(1);dollar=(c*v).shift(1).rolling(20,min_periods=20).mean()
 atr=((h-l).combine((h-prev).abs(),max)).combine((l-prev).abs(),max).rolling(14,min_periods=14).mean()
 atrpct=atr/c
 ret=c.pct_change()
 vol20=v.shift(1).rolling(20,min_periods=20).mean()
 rvol=v/vol20
 compression=ret.shift(1).rolling(5).std()/ret.shift(1).rolling(20).std()
 hi20=h.shift(1).rolling(20).max()
 # Independent detectors. No event-news assertion.
 c1=(compression<.65)&(c>hi20)&(rvol>=1.8)&(ret>.03)
 gap=d.Open/prev-1
 c2=(gap>=.07)&(ret>=.07)&(rvol>=2.5)&(c>=l+.7*(h-l))
 c3=(ret.shift(1)>=.08)&(rvol.shift(1)>=2)&(c>prev)&(c>=l+.6*(h-l))
 eligible=(prev>=5)&(dollar>=20_000_000)&(atrpct.shift(1)>=.04)
 return {"C1_VOL_EXPANSION":c1&eligible,"C2_GAP_PIVOT_PROXY":c2&eligible,
         "C3_SECOND_DAY":c3&eligible},dollar,atrpct,rvol
def main():
 out=Path("results/prism-v479");out.mkdir(parents=True,exist_ok=True)
 names,total=universe()
 pd.DataFrame({"ticker":names}).to_csv(out/"universe_snapshot.csv",index=False)
 signals=[];errors=[];loaded=0
 for batchstart in range(0,len(names),25):
  batch=names[batchstart:batchstart+25]
  try:
   data=yf.download(batch,start="2021-01-01",end=END,auto_adjust=True,progress=False,threads=False,group_by="ticker")
  except Exception as e:
   errors.extend({"ticker":t,"error":"BATCH_DOWNLOAD "+str(e)} for t in batch);continue
  for t in batch:
   try:
    d=data[t] if isinstance(data.columns,pd.MultiIndex) else data
    d=d.dropna(subset=["Open","High","Low","Close","Volume"]).copy()
    if len(d)<80:raise ValueError("Insufficient history")
    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    loaded+=1
    masks,dollar,atrpct,rvol=indicators(d)
    for name,mask in masks.items():
     valid=mask.fillna(False)&(d.index>=pd.Timestamp("2022-01-01"))
     # Signals require full D+5 follow-up; no artificial terminal mark.
     for i in np.flatnonzero(valid.to_numpy()):
      if i+6>=len(d):continue
      ep=float(d.Open.iloc[i+1])
      if not np.isfinite(ep) or ep<=0:continue
      row={"ticker":t,"detector":name,"signal_date":str(d.index[i].date()),
       "entry_date":str(d.index[i+1].date()),"entry_open":ep,
       "signal_close":float(d.Close.iloc[i]),"signal_rvol":float(rvol.iloc[i]),
       "prior_dollar_volume":float(dollar.iloc[i]),"prior_atr_pct":float(atrpct.iloc[i-1])}
      for n in HORIZONS:
       sub=d.iloc[i+1:i+2+n]
       row[f"ret_{n}"]=float(d.Close.iloc[i+1+n]/ep-1)
       row[f"mae_{n}"]=float(sub.Low.min()/ep-1)
       row[f"mfe_{n}"]=float(sub.High.max()/ep-1)
      signals.append(row)
   except Exception as e:errors.append({"ticker":t,"error":str(e)})
  print("processed",min(batchstart+25,len(names)),"of",len(names),flush=True)
 df=pd.DataFrame(signals)
 if df.empty:raise RuntimeError("No eligible signals; inspect coverage")
 df.to_csv(out/"signal_forward_returns.csv",index=False)
 (out/"coverage_errors.json").write_text(json.dumps(errors,indent=2))
 summary={"version":"v4.79","status":"SIGNAL_DISCOVERY_ONLY_NOT_PORTFOLIO_BACKTEST",
 "listing_snapshot_utc":datetime.now(timezone.utc).isoformat(),"listed_pool":total,
 "requested":len(names),"loaded":loaded,"errors":len(errors),"signals":len(df),"detectors":{},
 "limitations":[
 "Current exchange listings sampled by deterministic hash; NOT 2022-26 point-in-time universe.",
 "200 core names plus 800 pseudo-random names, NOT top 1000 by liquidity.",
 "Survivorship, delistings, short histories and corporate actions bias historical outcomes.",
 "C2 uses gap and volume as an event proxy, NOT verified earnings/news catalyst.",
 "Signal at close, next open entry; daily OHLC only; no intraday confirmation.",
 "D+1/3/5 returns and MAE/MFE are independent candidate outcomes, not executable stop fills.",
 "No portfolio compounding, fees, capital occupancy, stop loss or profit target yet.",
 "No trading, alert, or automatic strategy promotion."]}
 for name,g in df.groupby("detector"):
  summary["detectors"][name]={"signals":len(g),"tickers":int(g.ticker.nunique()),
   **{f"mean_return_{n}":float(g[f"ret_{n}"].mean()) for n in HORIZONS},
   **{f"median_return_{n}":float(g[f"ret_{n}"].median()) for n in HORIZONS},
   "win_rate_5d":float((g.ret_5>0).mean()),
   "hit_plus5_5d":float((g.mfe_5>=.05).mean()),
   "touch_minus3_5d":float((g.mae_5<=-.03).mean())}
 (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 rows="".join(f"<tr><td>{html.escape(k)}</td><td>{v['signals']}</td><td>{v['mean_return_5']:.2%}</td><td>{v['win_rate_5d']:.1%}</td><td>{v['hit_plus5_5d']:.1%}</td><td>{v['touch_minus3_5d']:.1%}</td></tr>" for k,v in summary["detectors"].items())
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM Seed C v4.67</title><style>body{background:#10192c;color:#eaf2ff;font:15px system-ui;margin:30px}h1{color:#5ce6b9}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #38485e}</style><h1>PRISM Seed C · High Velocity Research</h1><p>Historical signal discovery only. No executable P&amp;L yet.</p><p>Requested 1000 / Loaded "+str(loaded)+" / Signals "+str(len(df))+"</p><table><tr><th>Detector</th><th>Signals</th><th>Mean D+5</th><th>Win D+5</th><th>MFE +5%</th><th>MAE -3%</th></tr>"+rows+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"GENERATED","loaded":loaded,"signals":len(df)}))
if __name__=="__main__":main()
