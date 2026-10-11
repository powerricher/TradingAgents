"""PRISM v4.53 — Long-pattern post-entry paths and exit-only comparison.
Frozen v4.52 candidates; no hindsight exit optimization or entry changes.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v445 import UNIVERSE,CORE100
from .universe_v413 import UNIVERSE as CORE80
from .trend_v48 import load_prices,portfolio
from .trend_v49 import exit_trade
from .atr_frontier_v433 import exit_atr,CUTOFF
INITIAL=10_000_000.
HORIZONS=(20,60,120)
MODES=("MA30_ORIGINAL","MA30_CONFIRM2","ATR3","ATR4")
def post_entry(d,entry):
 idx=d.index.get_loc(pd.Timestamp(entry))
 opening=float(d.Open.iloc[idx])
 row={}
 for n in HORIZONS:
  if idx+n>=len(d):
   for k in ("return","mae","mfe"):row[f"{k}_{n}"]=np.nan
   continue
  sub=d.iloc[idx:idx+n+1]
  row[f"return_{n}"]=float(d.Close.iloc[idx+n]/opening-1)
  row[f"mae_{n}"]=float(sub.Low.min()/opening-1)
  row[f"mfe_{n}"]=float(sub.High.max()/opening-1)
 return row
def exit_variant(d,r,mode):
 if mode=="MA30_ORIGINAL":
  return str(r.exit_date),float(r.exit_price),str(r.reason)
 if mode=="MA30_CONFIRM2":
  z=exit_trade(d,pd.Timestamp(r.signal_date),"CONFIRM30_2")
  if z is None:return None
  e,x,ep,px,reason=z
  if str(d.index[e].date())!=str(r.entry_date)[:10]:
   raise RuntimeError("Entry mismatch "+str(r.ticker))
  return str(d.index[x].date()),float(px),reason
 xd,px,reason=exit_atr(d,pd.Timestamp(r.entry_date),int(mode[-1]))
 return str(xd.date()),float(px),reason
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--candidates",default="results/prism-v452/candidate_exits.csv")
 ap.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv")
 ap.add_argument("--out",default="results/prism-v453")
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 source=pd.read_csv(a.candidates)
 needed={"ticker","signal_date","entry_date","exit_date","entry_price","exit_price","edge","reason","pattern"}
 if not needed.issubset(source):raise RuntimeError("Missing frozen v4.52 candidate fields")
 if source.empty:raise RuntimeError("Empty v4.52 candidates")
 prices={};errors=[]
 for ticker in sorted(source.ticker.unique()):
  try:prices[ticker]=load_prices(ticker,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
  except Exception as e:errors.append({"ticker":ticker,"error":str(e)})
 if errors:raise RuntimeError("Missing historical prices; cannot compare identical candidates "+str(errors))
 paths=[];variants=[];unresolved=[]
 for r in source.itertuples(index=False):
  d=prices[r.ticker]
  if pd.Timestamp(r.entry_date) not in d.index:raise RuntimeError("Missing entry "+str(r.ticker))
  paths.append({"ticker":r.ticker,"pattern":r.pattern,"entry_date":r.entry_date,
   "original_exit_date":r.exit_date,"original_exit_return":float(r.exit_price/r.entry_price-1),
   **post_entry(d,r.entry_date)})
  for mode in MODES:
   result=exit_variant(d,r,mode)
   if result is None:
    unresolved.append({"ticker":r.ticker,"entry_date":r.entry_date,"mode":mode});continue
   xd,px,reason=result
   variants.append({"ticker":r.ticker,"pattern":r.pattern,"entry_date":r.entry_date,
    "exit_date":xd,"entry_price":float(r.entry_price),"exit_price":px,
    "edge":float(r.edge),"reason":reason,"mode":mode,
    "return_gross":float(px/r.entry_price-1),
    "terminal_mark":reason=="terminal_mark_to_market"})
 pd.DataFrame(paths).to_csv(out/"post_entry_paths.csv",index=False)
 pd.DataFrame(variants).to_csv(out/"exit_candidate_comparison.csv",index=False)
 pd.DataFrame(unresolved,columns=["ticker","entry_date","mode"]).to_csv(out/"unresolved_exits.csv",index=False)
 pathdf=pd.DataFrame(paths)
 statistics=[]
 for pattern,g in pathdf.groupby("pattern"):
  entry={"pattern":pattern,"candidates":len(g),"original_mean_return":float(g.original_exit_return.mean())}
  for n in HORIZONS:
   valid=g.dropna(subset=[f"return_{n}"])
   entry[f"count_{n}"]=len(valid)
   entry[f"mean_return_{n}"]=float(valid[f"return_{n}"].mean()) if len(valid) else None
   entry[f"median_return_{n}"]=float(valid[f"return_{n}"].median()) if len(valid) else None
   entry[f"positive_fraction_{n}"]=float((valid[f"return_{n}"]>0).mean()) if len(valid) else None
   entry[f"mean_mae_{n}"]=float(valid[f"mae_{n}"].mean()) if len(valid) else None
   entry[f"mean_mfe_{n}"]=float(valid[f"mfe_{n}"].mean()) if len(valid) else None
  statistics.append(entry)
 pd.DataFrame(statistics).to_csv(out/"pattern_forward_summary.csv",index=False)
 seed=pd.read_csv(a.seed_a)
 seed=seed[seed.variant=="ATR4_CONTROL"].copy()
 seed.date=pd.to_datetime(seed.date).dt.normalize()
 aa=seed.set_index("date").equity.sort_index()
 if aa.empty or aa.index.duplicated().any():raise RuntimeError("Invalid frozen Seed A")
 groups={"TECH80":set(CORE80),"TECH100":set(CORE100),"TECH150":set(UNIVERSE)}
 variants_df=pd.DataFrame(variants)
 report={"version":"v4.53","status":"HISTORICAL EXIT-ONLY COMPARISON / NOT INDEPENDENT OOS",
  "candidate_count":len(source),"unresolved_exits":len(unresolved),"pattern_forward":statistics,
  "strategies":{},"limitations":[
   "All entries are frozen v4.52 candidates; only exit policy changes.",
   "D+20/60/120 outcomes are retrospective labels and never used for trading decisions.",
   "ATR3/4 can terminal-mark at cutoff; such values are not realized exits.",
   "Exit candidates without resolved exits are excluded; check counts before interpreting.",
   "Portfolio uses one full-capital position and no rotation; Seed A TECH80 ATR4 frozen.",
   "The same historical sample was repeatedly researched; not untouched OOS.",
   "OHLC auto-adjusted; stop fills are modeled, not guaranteed; daily MDD excludes intraday lows.",
   "Comparisons use 150-session pattern detector; formation sensitivity not tested."]}
 ledger_all=[];curve_all=[]
 for group,allowed in groups.items():
  for pattern in ("ASCENDING_TRIANGLE","DESCENDING_TRIANGLE_REVERSAL","FALLING_WEDGE","ALL_LONG_PATTERNS"):
   for mode in MODES:
    z=variants_df[(variants_df['mode']==mode)&variants_df.ticker.isin(allowed)]
    if pattern!="ALL_LONG_PATTERNS":z=z[z.pattern==pattern]
    name=f"{group}_{pattern}_{mode}"
    if z.empty:
     report["strategies"][name]={"status":"NO_SIGNALS"};continue
    records=z[["ticker","entry_date","exit_date","entry_price","exit_price","edge","reason"]].to_dict("records")
    try:perf,ledger,curve=portfolio(records,prices,name,rotation=False)
    except Exception as e:
     report["strategies"][name]={"status":"INVALID","error":str(e)};continue
    b=pd.DataFrame(curve);b.date=pd.to_datetime(b.date).dt.normalize()
    bb=b.set_index("date").equity.sort_index()
    cal=aa.index.union(bb.index).sort_values()
    xa=aa.reindex(cal);xb=bb.reindex(cal)
    xa.loc[cal<aa.index.min()]=INITIAL;xb.loc[cal<bb.index.min()]=INITIAL
    if xa.isna().any() or xb.isna().any():raise RuntimeError("Internal valuation gap")
    total=xa+xb;peak=np.maximum.accumulate(np.r_[2*INITIAL,total.to_numpy()])[1:]
    keys={(r["ticker"],r["entry_date"]):r for r in records}
    terminal=sum(keys[(r["ticker"],r["entry_date"])]["reason"]=="terminal_mark_to_market" for r in ledger)
    report["strategies"][name]={"seed_b":perf,"combined_return":float(total.iloc[-1]/(2*INITIAL)-1),
     "combined_mdd":float(np.min(total.to_numpy()/peak-1)),
     "terminal_marks_executed":terminal,"candidate_count":len(records)}
    for row in ledger:row["strategy"]=name
    for row in curve:row["strategy"]=name
    ledger_all+=ledger;curve_all+=curve
 pd.DataFrame(ledger_all).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curve_all).to_csv(out/"seed_b_daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.53</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.53 · Long Pattern Outcomes and Exit Policies</h1><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","candidate_count":len(source),"strategies":len(report["strategies"])}))
if __name__=="__main__":main()
