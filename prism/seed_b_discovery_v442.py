"""PRISM v4.42 — independent Seed B signal discovery (TECH80).
Three predeclared price-action detectors; MA30 exits. Historical research only.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v413 import UNIVERSE
from .trend_v48 import load_prices,portfolio
from .trend_v49 import exit_trade
from .regime_adaptive_v436 import regimes
from .atr_frontier_v433 import CUTOFF

INITIAL=10_000_000.
def discover(d,q,ticker):
 c=d.Close.astype(float);h=d.High.astype(float);l=d.Low.astype(float);v=d.Volume.astype(float)
 r20=c.pct_change(20);qr=q.pct_change(20).reindex(d.index)
 high20=h.shift(1).rolling(20,min_periods=20).max()
 low20=l.shift(1).rolling(20,min_periods=20).min()
 vol20=v.shift(1).rolling(20,min_periods=20).mean()
 range10=(h-l).rolling(10,min_periods=10).mean()
 range40=(h-l).rolling(40,min_periods=40).mean()
 rs=r20-qr
 rs_signal=(qr<-.03)&(r20>.03)&(rs>.08)&(c>c.rolling(20,min_periods=20).mean())
 # Prior session broke prior support; today's close recaptures the level.
 prev_support=l.shift(2).rolling(20,min_periods=20).min()
 reversal=(l.shift(1)<prev_support)&(c>prev_support)&(c>c.shift(1))&(v>vol20)
 breakout=(range10/range40<.75)&(c>high20)&(v>1.3*vol20)
 # BCH: near the 20/30-session high after a stable, defended base.
 hi20=h.rolling(20,min_periods=20).max()
 hi30=h.rolling(30,min_periods=30).max()
 proximity=((c/hi20>=.97)&(c<=hi20))|((c/hi30>=.97)&(c<=hi30))
 base_high=h.rolling(25,min_periods=25).max()
 base_low=l.rolling(25,min_periods=25).min()
 tight=(base_high/base_low-1<=.18)
 previous_low=l.shift(10).rolling(10,min_periods=10).min()
 recent_low=l.rolling(10,min_periods=10).min()
 defended=recent_low>=previous_low*.97
 avg10=c.rolling(10,min_periods=10).mean()
 compression=c.pct_change().rolling(10,min_periods=10).std()<=c.pct_change().rolling(30,min_periods=30).std()
 bch=proximity&tight&defended&(c>=avg10)&compression
 bch_score=(c/base_high)+.2*(recent_low/previous_low-1)
 result=[]
 for label,mask,score in [
  ("RS_DIVERGENCE",rs_signal,rs),
  ("FAILED_BREAKDOWN",reversal,(c/prev_support-1)),
  ("VCP_BREAKOUT",breakout,(c/high20-1)),
  ("BASE_CONSOLIDATION_HIGH",bch,bch_score)]:
  ok=mask.fillna(False)&(d.index>=pd.Timestamp("2022-01-01"))&(d.index<=CUTOFF-pd.Timedelta(days=220))
  for dt in d.index[ok]:
   if not np.isfinite(score.loc[dt]):continue
   result.append({"ticker":ticker,"date":dt,"detector":label,"edge":float(score.loc[dt])})
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",default="results/prism-v442");ap.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv");ap.add_argument("--control",default="results/prism-v420/scored_signals.csv");a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 prices={t:load_prices(t,"2021-01-01").loc[lambda x:x.index<=CUTOFF] for t in UNIVERSE}
 q=load_prices("QQQ","2021-01-01").loc[lambda x:x.index<=CUTOFF].Close
 signals=pd.DataFrame([r for ticker,d in prices.items() for r in discover(d,q,ticker)])
 if signals.empty:raise RuntimeError("No independently discovered signals")
 signals=signals.sort_values(["date","ticker","detector"]).drop_duplicates(["date","ticker","detector"])
 market=regimes(signals.date.min())
 control=pd.read_csv(a.control)
 control["date"]=pd.to_datetime(control.date).dt.normalize()
 base_keys=set(zip(control.ticker,control.date))
 signals["in_vcp"]= [(t,d) in base_keys for t,d in zip(signals.ticker,signals.date)]
 signals["regime"]=signals.date.map(market)
 if signals.regime.isna().any():raise RuntimeError("Missing QQQ regime")
 # Select one best independent candidate per signal date and detector.
 candidates=[];issues=[]
 for r in signals.itertuples(index=False):
  d=prices[r.ticker]
  try:z=exit_trade(d,r.date,"CONFIRM30_1")
  except (ValueError,KeyError) as exc:z=None
  if z is None:
   issues.append({"ticker":r.ticker,"date":str(r.date.date()),"detector":r.detector})
   continue
  e,x,entry,exit_price,reason=z
  candidates.append({"ticker":r.ticker,"signal_date":str(r.date.date()),"entry_date":str(d.index[e].date()),
   "exit_date":str(d.index[x].date()),"entry_price":float(entry),"exit_price":float(exit_price),
   "edge":float(r.edge),"reason":reason,"detector":r.detector,"regime":r.regime,"in_vcp":bool(r.in_vcp)})
 if issues:raise RuntimeError("Incomplete exit coverage: "+str(issues[:8])+" count "+str(len(issues)))
 trades=pd.DataFrame(candidates)
 seed_a=pd.read_csv(a.seed_a)
 seed_a=seed_a[seed_a.variant=="ATR4_CONTROL"].copy()
 seed_a.date=pd.to_datetime(seed_a.date).dt.normalize()
 aa=seed_a.set_index("date").equity
 report={"version":"v4.42","status":"HISTORICAL DISCOVERY RESEARCH / NOT INDEPENDENT OOS","strategies":{},
  "discovery":{"signals":len(signals),"outside_vcp":int((~signals.in_vcp).sum()),
   "by_detector":signals.detector.value_counts().to_dict(),
   "by_regime":signals.regime.value_counts().to_dict()},
  "limitations":["New Seed B candidates are independent of VCP eligibility, but drawn from the same retrospective TECH80 universe.",
  "Signal thresholds are hypotheses informed by prior historical research; no independent OOS.",
  "Exit coverage requires every signal to have a completed MA30 trade; signals within 220 calendar days of cutoff excluded.",
  "Seed A ATR4 is frozen and includes terminal synthetic valuation.",
  "Seed B has no shorting, leverage, sector rotation or earnings event timestamps.",
  "Scores are fixed rules, not trained D+5/D+10/D+20 forecasts.",
  "BCH uses high proximity, 25-session range <=18%, defended lows, MA10 and volatility compression.",
  "Leading dates without account valuations are treated as initial cash; internal gaps fail.",
  "Daily-close MDD excludes intraday drawdowns."]}
 ledgers=[];curves=[];combined=[]
 for mode in ("RS_DIVERGENCE","FAILED_BREAKDOWN","VCP_BREAKOUT","BASE_CONSOLIDATION_HIGH","ALL_DISCOVERY"):
  z=trades if mode=="ALL_DISCOVERY" else trades[trades.detector==mode]
  if z.empty:raise RuntimeError("No candidate trades "+mode)
  # Within one date, cross-detector scores are not calibrated. The ALL mode
  # gives fixed priority to RS, then reversal, then breakout as a hypothesis.
  rows=[]
  for r in z.itertuples(index=False):
   priority={"RS_DIVERGENCE":4,"FAILED_BREAKDOWN":3,"VCP_BREAKOUT":2,"BASE_CONSOLIDATION_HIGH":1}[r.detector]
   rows.append({"ticker":r.ticker,"mode":mode,"entry_date":r.entry_date,"exit_date":r.exit_date,
    "entry_price":r.entry_price,"exit_price":r.exit_price,"edge":float(priority*100+r.edge) if mode=="ALL_DISCOVERY" else r.edge,"reason":r.reason})
  perf,ledger,curve=portfolio(rows,prices,mode,rotation=False)
  b=pd.DataFrame(curve);b.date=pd.to_datetime(b.date).dt.normalize()
  bb=b.set_index("date").equity
  calendar=aa.index.union(bb.index).sort_values()
  xa=aa.reindex(calendar);xb=bb.reindex(calendar)
  xa.loc[calendar<aa.index.min()]=INITIAL
  xb.loc[calendar<bb.index.min()]=INITIAL
  if xa.isna().any() or xb.isna().any():
   raise RuntimeError("Internal equity calendar gaps "+mode)
  total=xa+xb;peak=np.maximum.accumulate(np.r_[2*INITIAL,total.to_numpy()])[1:]
  report["strategies"][mode]={"seed_b":perf,"combined_return":float(total.iloc[-1]/(2*INITIAL)-1),
   "combined_mdd":float(np.min(total.to_numpy()/peak-1)),"candidate_count":len(z)}
  for r in ledger:r["variant"]=mode
  for r in curve:r["variant"]=mode
  ledgers+=ledger;curves+=curve
  combined.extend({"date":str(day.date()),"variant":mode,"seed_a_equity":float(ae),
   "seed_b_equity":float(be),"combined_equity":float(ae+be)} for day,ae,be in zip(total.index,xa,xb))
 signals.to_csv(out/"discovered_signals.csv",index=False)
 trades.to_csv(out/"candidate_exits.csv",index=False)
 pd.DataFrame(ledgers).to_csv(out/"seed_b_ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"seed_b_daily_equity.csv",index=False)
 pd.DataFrame(combined).to_csv(out/"combined_daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['seed_b']['return']:.2%}</td><td>{v['combined_return']:.2%}</td><td>{v['combined_mdd']:.2%}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.42</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;padding:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.42 · Independent Seed B Discovery</h1><p>Same TECH80 universe, independent signal discovery; historical research only</p><table><tr><th>Seed B detector</th><th>Seed B return</th><th>Combined return</th><th>Combined MDD</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","discovery":report["discovery"],"strategies":{k:v["combined_return"] for k,v in report["strategies"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
