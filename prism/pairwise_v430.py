"""PRISM v4.30: VCP 100-stock universe and chronological pairwise ranking.
No future trade outcome used before both compared trades have exited.
Historical research, NOT independent out-of-sample validation.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .trend_v48 import load_prices,portfolio
from .price_action_v419 import feature_frame

FEATURES=["edge","breakout_distance20","breakout_distance60","rs20",
 "trend_persistence","contraction_ratio","range_contraction","volume_dryup"]
def prepare(signals,trades,prices):
 s=signals[["ticker","date","edge"]].copy()
 s.date=pd.to_datetime(s.date).dt.normalize()
 t=trades[["ticker","signal_date","entry_date","exit_date","entry_price","exit_price","reason","ret"]].copy()
 t.signal_date=pd.to_datetime(t.signal_date).dt.normalize()
 t.exit_date=pd.to_datetime(t.exit_date).dt.normalize()
 d=s.merge(t,left_on=["ticker","date"],right_on=["ticker","signal_date"],validate="one_to_one")
 if len(d)!=len(s):raise RuntimeError("Incomplete signal trade join")
 fs=[]
 for ticker,g in d.groupby("ticker"):
  f=feature_frame(prices[ticker]).reset_index()
  f=f.rename(columns={f.columns[0]:"date"})
  f.date=pd.to_datetime(f.date).dt.normalize()
  f["ticker"]=ticker
  fs.append(f)
 d=d.merge(pd.concat(fs,ignore_index=True),on=["ticker","date"],how="left",validate="one_to_one")
 if d[FEATURES+["ret"]].isna().any().any():raise RuntimeError("Incomplete signal-time features")
 return d.sort_values(["date","ticker"]).reset_index(drop=True)
def evaluate(d,prices,label):
 d=d.copy()
 for name in ("VCP","PAIRWISE"):d["rank_"+name]=np.nan
 trained=0;pair_months=0
 for month,g in d.groupby(d.date.dt.to_period("M"),sort=True):
  train=d[d.exit_date<month.start_time-pd.Timedelta(days=5)]
  if len(train)<50:
   for name in ("VCP","PAIRWISE"):d.loc[g.index,"rank_"+name]=g.edge
   continue
  reg=make_pipeline(StandardScaler(),Ridge(alpha=100.))
  reg.fit(train[FEATURES],train.ret)
  vcp=reg.predict(g[FEATURES])
  d.loc[g.index,"rank_VCP"]=vcp
  # Same-day historical comparisons only, both exits known before month.
  pairs=[];outcomes=[]
  for _,z in train.groupby("date"):
   if len(z)<2:continue
   arr=z[FEATURES].to_numpy(dtype=float);ret=z.ret.to_numpy()
   for i in range(len(z)):
    for j in range(i+1,len(z)):
     if ret[i]==ret[j]:continue
     delta=arr[i]-arr[j];win=int(ret[i]>ret[j])
     pairs.extend([delta,-delta]);outcomes.extend([win,1-win])
  if len(pairs)<30 or len(set(outcomes))<2:
   d.loc[g.index,"rank_PAIRWISE"]=vcp
   continue
  clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500))
  clf.fit(np.asarray(pairs),np.asarray(outcomes))
  # Pairwise score is mean predicted head-to-head win probability within
  # the day's CURRENT candidates. Single candidate uses VCP score.
  for day,z in g.groupby("date"):
   if len(z)==1:
    d.loc[z.index,"rank_PAIRWISE"]=vcp[g.index.get_indexer(z.index)[0]]
    continue
   arr=z[FEATURES].to_numpy(dtype=float)
   delta=(arr[:,None,:]-arr[None,:,:]).reshape(-1,len(FEATURES))
   probs=clf.predict_proba(delta)[:,1].reshape(len(z),len(z))
   np.fill_diagonal(probs,np.nan)
   d.loc[z.index,"rank_PAIRWISE"]=np.nanmean(probs,axis=1)
  trained+=len(train);pair_months+=1
 if d[["rank_VCP","rank_PAIRWISE"]].isna().any().any():raise RuntimeError("Unranked signals")
 result={};ledgers=[];curves=[]
 for model in ("VCP","PAIRWISE"):
  rows=[{"ticker":r.ticker,"mode":model,"entry_date":str(pd.Timestamp(r.entry_date).date()),
    "exit_date":str(pd.Timestamp(r.exit_date).date()),"entry_price":float(r.entry_price),
    "exit_price":float(r.exit_price),"edge":float(getattr(r,"rank_"+model)),"reason":r.reason}
    for r in d.itertuples(index=False)]
  perf,ledger,curve=portfolio(rows,prices,model,rotation=False)
  if perf["trades"]==0:raise RuntimeError("Zero trade portfolio")
  result[model]=perf
  for x in ledger:x["variant"]=label+"_"+model
  for x in curve:x["variant"]=label+"_"+model
  ledgers+=ledger;curves+=curve
 return result,d,ledgers,curves,pair_months
def main():
 p=argparse.ArgumentParser();p.add_argument("--v429",default="results/prism-v429");p.add_argument("--v420",default="results/prism-v420/scored_signals.csv");p.add_argument("--out",default="results/prism-v430");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 root=Path(a.v429)
 t=pd.read_csv(root/"trades.csv")
 all_results={};all_scored=[];all_ledgers=[];all_curves=[]
 # Compare 80-trained on 80/100 and 100-trained on 100.
 for train,universe in [("FROZEN80_TRAIN","CORE80"),("FROZEN80_TRAIN","TECH100"),("REFIT100_TRAIN","TECH100")]:
  key=train+"_"+universe
  s=pd.read_csv(root/(key+"_signals.csv"))
  sub=t[t.experiment.eq(key)]
  if len(sub)!=len(s):raise RuntimeError("Incomplete candidate exits "+key)
  prices={ticker:load_prices(ticker,g.date.min()) for ticker,g in s.groupby("ticker")}
  d=prepare(s,sub,prices)
  result,scored,ledger,curve,n=evaluate(d,prices,key)
  all_results[key]={"models":result,"pairwise_training_months":n,"signals":len(d)}
  scored["experiment"]=key;all_scored.append(scored)
  all_ledgers+=ledger;all_curves+=curve
 # Preserve historical 80-stock VCP reference from v4.20 as an explicitly
 # separate comparison, not falsely identical to the re-trained 80 model.
 reference={"status":"v4.20 historical VCP80 benchmark is external; no v4.20 artifact required", "historical_compound_return":1.8995, "comparison":"NON_IDENTICAL_MODEL"}
 report={"version":"v4.30","status":"RESEARCH / NOT INDEPENDENT OOS","experiments":all_results,
 "reference":reference,"limitations":[
 "VCP100 is a chronological ridge ranker on price-action/VCP features, not an exact reproduction of v4.20 feature set.",
 "80-stock original VCP 189.95% is a separate historical benchmark; model features and training samples differ.",
 "Pairwise training uses only comparisons from the same historical signal date, with BOTH trade exits completed before training month minus 5 days.",
 "Pairwise model uses candidate features at signal close, with next-session-open entry; outcomes never used to rank that same date.",
 "Historical 100-stock universe is retrospectively selected and has survivorship bias.",
 "Same 2022-2026 sample repeatedly researched; no independent OOS. Hindsight ceiling is not attainable profit.",
 "Single-slot full-capital MA30 exits and fixed transaction assumptions; daily-close MDD omits intraday troughs."]}
 pd.concat(all_scored,ignore_index=True).to_csv(out/"scored_signals.csv",index=False)
 pd.DataFrame(all_ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(all_curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 rows="".join(f"<tr><td>{html.escape(k+' '+m)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td></tr>" for k,z in all_results.items() for m,v in z["models"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.30</title><style>body{background:#0b1729;color:#edf7ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.30 · VCP100 & Pairwise</h1><p>Historical research only. Baseline v4.20 VCP80 is not identical to these refitted models.</p><table><tr><th>Model</th><th>Return</th><th>Daily MDD</th><th>Trades</th></tr>"+rows+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","experiments":{k:{m:round(v["return"],4) for m,v in z["models"].items()} for k,z in all_results.items()}},ensure_ascii=False))
if __name__=="__main__":main()
