"""On-demand TECH150 opportunity HTML scanner. No alerts or orders."""
import json,html
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from .universe_v445 import UNIVERSE
from .trend_v48 import load_prices
from .trend_v47 import atr
from .technical_quality_v450 import features,quality
def main():
 out=Path("results/prism-v458");out.mkdir(parents=True,exist_ok=True)
 rows=[];errors=[]
 for ticker in UNIVERSE:
  try:
   d=load_prices(ticker,"2025-01-01")
   if len(d)<85:raise ValueError("Insufficient bars")
   c=d.Close.astype(float);h=d.High.astype(float);l=d.Low.astype(float);v=d.Volume.astype(float)
   vol20=float(v.iloc[-21:-1].mean())
   high20=float(h.iloc[-21:-1].max())
   if vol20<=0:raise ValueError("Invalid volume")
   ratio=float(v.iloc[-1]/vol20)
   r=h-l
   contraction=float(r.iloc[-10:].mean()/r.iloc[-40:].mean())
   breakout=contraction<.75 and float(c.iloc[-1])>high20 and ratio>1.3
   recent=float(h.iloc[-20:].max()/l.iloc[-20:].min()-1)
   prior=float(h.iloc[-60:-20].max()/l.iloc[-60:-20].min()-1)
   dry=float(v.iloc[-5:].mean()/vol20)
   vcp=prior>0 and recent/prior<.7 and dry<.85 and float(c.iloc[-1])>float(c.iloc[-20:].mean())
   tmp=d.copy();tmp.loc[d.index[-1]+pd.Timedelta(days=1)]=d.iloc[-1]
   f=features(tmp,tmp.index[-1])
   av=atr(d).iloc[-1]
   rows.append({"ticker":ticker,"date":str(d.index[-1].date()),"close":float(c.iloc[-1]),
    "volume_ratio":ratio,"quality":float(quality(f)) if f else None,
    "breakout_level":high20,"atr4_reference":float(c.iloc[-1]-4*av) if pd.notna(av) else None,
    "signal":"BREAKOUT" if breakout else ("VCP_WATCH" if vcp else "NO_SIGNAL")})
  except Exception as exc:errors.append({"ticker":ticker,"error":str(exc)})
 if not rows:raise RuntimeError("No usable price data")
 df=pd.DataFrame(rows);latest=df.date.max()
 df.loc[df.date!=latest,"signal"]="STALE"
 order={"BREAKOUT":0,"VCP_WATCH":1,"NO_SIGNAL":2,"STALE":3}
 df["priority"]=df.signal.map(order)
 df=df.sort_values(["priority","quality","ticker"],ascending=[True,False,True])
 df.to_csv(out/"screened_universe.csv",index=False)
 selected=df[df.signal.isin(["BREAKOUT","VCP_WATCH"])]
 selected.to_csv(out/"candidates.csv",index=False)
 (out/"data_errors.json").write_text(json.dumps(errors,indent=2))
 summary={"version":"v4.58","date":latest,"generated_utc":datetime.now(timezone.utc).isoformat(),
  "scanned":len(df),"breakout":int((df.signal=="BREAKOUT").sum()),
  "vcp_watch":int((df.signal=="VCP_WATCH").sum()),"stale":int((df.signal=="STALE").sum()),
  "errors":len(errors),"notes":["Seed A VCP is a screening proxy, not the trained historical ranking.",
  "ATR4 is a dynamic exit reference, not a guaranteed initial stop.",
  "Quality is uncalibrated; no investment orders or alerts."]}
 (out/"summary.json").write_text(json.dumps(summary,indent=2))
 tr=[]
 for r in df.itertuples(index=False):
  atrref="—" if pd.isna(r.atr4_reference) else f"{r.atr4_reference:.2f}"
  tr.append(f"<tr><td>{html.escape(r.ticker)}</td><td>{r.signal}</td><td>{r.date}</td><td>{r.close:.2f}</td><td>{r.volume_ratio:.2f}x</td><td>{r.quality:.1f}</td><td>{r.breakout_level:.2f}</td><td>{atrref}</td></tr>")
 page="<html lang='ko'><meta charset='utf-8'><title>PRISM Opportunity Report</title><style>body{background:#101b2d;color:#e8f1ff;font:14px system-ui;margin:30px}h1{color:#64dfbd}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #35445b;text-align:right}td:first-child,th:first-child{text-align:left}</style><h1>PRISM v4.58 Opportunity Scanner</h1><p>Market date: "+latest+" | On-demand TECH150 scan | Seed A VCP proxy / ATR4, Seed B Breakout / MA30</p><p>Breakout: "+str(summary["breakout"])+" | VCP watch: "+str(summary["vcp_watch"])+" | Scanned: "+str(len(df))+"/150</p><table><tr><th>Ticker</th><th>Signal</th><th>Date</th><th>Close</th><th>Vol/20D</th><th>Quality</th><th>20D High</th><th>ATR4 Reference</th></tr>"+"".join(tr)+"</table><p>Research-only technical screening; no automatic trades. ATR4 reference is not a fixed stop.</p></html>"
 (out/"report.html").write_text(page,encoding="utf-8")
 print(json.dumps(summary))
if __name__=="__main__":main()
