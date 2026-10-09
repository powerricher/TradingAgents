"""PRISM v4.35: QQQ market regime attribution on frozen MA30/ATR4 equity curves.
Daily regimes use only PRIOR session QQQ closes, preventing same-day hindsight.
Historical diagnostic, not a tradable strategy or causal alpha estimate.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf

def qqq_prices(start,end):
 d=yf.download("QQQ",start=start,end=end,auto_adjust=True,progress=False)
 if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
 if d.empty or "Close" not in d:raise RuntimeError("QQQ price download failed")
 d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
 return d.Close.astype(float).sort_index()
def main():
 p=argparse.ArgumentParser();p.add_argument("--equity",default="results/prism-v434/daily_equity.csv");p.add_argument("--out",default="results/prism-v435");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.equity)
 if not {"date","equity","strategy"}.issubset(df):raise ValueError("Missing equity fields")
 df.date=pd.to_datetime(df.date).dt.normalize()
 df=df[df.strategy.isin(["MA30","ATR4"])].copy()
 if df.empty:raise RuntimeError("No benchmark strategies")
 first=df.date.min();last=df.date.max()
 q=qqq_prices((first-pd.Timedelta(days=440)).strftime("%Y-%m-%d"),(last+pd.Timedelta(days=5)).strftime("%Y-%m-%d"))
 f=pd.DataFrame({"qqq_close":q})
 f["ma200"]=q.rolling(200,min_periods=200).mean()
 f["ret60"]=q.pct_change(60)
 # Regime observed at previous close, before today's portfolio return.
 f["regime"]=np.select([(q>f.ma200)&(f.ret60>.05),(q<f.ma200)&(f.ret60<-.05)],["BULL","BEAR"],default="SIDEWAYS")
 f.loc[f.ma200.isna()|f.ret60.isna(),"regime"]="UNKNOWN"
 f["regime_prior"]=f.regime.shift(1)
 f["qqq_daily_return"]=q.pct_change()
 report={"version":"v4.35","status":"HISTORICAL ATTRIBUTION / NOT INDEPENDENT OOS","strategies":{},
 "regime_definition":"Prior close QQQ > MA200 and prior 60-session return >5%: bull; below MA200 and <-5%: bear; otherwise sideways.",
 "limitations":["Regimes lag one session to avoid same-day classification lookahead.",
 "This is descriptive attribution, not a causally identified alpha estimate.",
 "Cash exposure cannot be separated without daily position-level data; zero-return days are not automatically cash days.",
 "Same retrospectively selected universe and historical period were used for model development.",
 "ATR4 final terminal mark is unrealized; legacy equity engine treats terminal liquidation synthetically.",
 "Annual and regime compounded subseries are not directly additive; regime P&L contributions are in KRW."]}
 rows=[]
 for name,g in df.groupby("strategy"):
  g=g.sort_values("date").drop_duplicates("date")
  g=g.merge(f[["regime_prior","qqq_daily_return","qqq_close"]],left_on="date",right_index=True,how="left")
  if g[["regime_prior","qqq_daily_return","qqq_close"]].isna().any().any():raise RuntimeError("Missing QQQ regime for "+name)
  g["pnl_krw"]=g.equity.diff().fillna(g.equity-10_000_000.)
  g["daily_return"]=g.equity.pct_change().fillna(g.equity/10_000_000.-1)
  g["year"]=g.date.dt.year
  regimes={}
  for regime,z in g.groupby("regime_prior"):
   rr=z.daily_return.to_numpy()
   qr=z.qqq_daily_return.to_numpy()
   regimes[regime]={"sessions":len(z),"pnl_contribution_krw":float(z.pnl_krw.sum()),
    "strategy_compound_on_regime_days":float(np.prod(1+rr)-1),
    "qqq_compound_on_same_days":float(np.prod(1+qr)-1),
    "strategy_positive_day_share":float((rr>0).mean())}
  years={}
  for year,z in g.groupby("year"):
   prior=10_000_000. if z.index[0]==g.index[0] else float(g.loc[g.date<z.date.min(),"equity"].iloc[-1])
   years[str(year)]={"strategy_return":float(z.equity.iloc[-1]/prior-1),
    "qqq_return":float(z.qqq_close.iloc[-1]/q.loc[q.index<z.date.min()].iloc[-1]-1)}
  report["strategies"][name]={"regimes":regimes,"years":years,"total_pnl_krw":float(g.pnl_krw.sum())}
  g["strategy"]=name;rows.append(g)
 pd.concat(rows,ignore_index=True).to_csv(out/"daily_regime_attribution.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.35</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.35 · Market Regime Attribution</h1><p>QQQ bull/bear/sideways, prior-close labels. Historical attribution only.</p><pre>"+html.escape(json.dumps(report["strategies"],ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","strategies":list(report["strategies"])}))
if __name__=="__main__":main()
