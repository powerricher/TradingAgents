"""PRISM v4.11 multi-horizon entry research.
Strictly time-separated 5/10/20-session labels; no API key.
Candidate research only; no portfolio return claims.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from .risk_gate import _prep,_X
from .rolling_v42 import _add_regime,_XR
from .rolling_v41 import FOLDS
from .fast_lab import load_history
from .universe_v45 import UNIVERSE
HORIZONS=(5,10,20)
def prepare():
    panel=_add_regime(_prep(UNIVERSE,"2021-01-01","2026-10-06"))
    if panel.empty:raise RuntimeError("Empty feature panel")
    panel["date"]=pd.to_datetime(panel.date)
    for h in HORIZONS:panel[f"fwd{h}"]=np.nan
    for ticker in panel.ticker.unique():
        d=load_history(ticker,"2021-01-01","2026-10-06").sort_index()
        dates=pd.to_datetime(d.index).tz_localize(None)
        close=pd.Series(np.asarray(d.Close,dtype=float),index=dates)
        for h in HORIZONS:
            future=close.shift(-h)/close-1
            mask=panel.ticker.eq(ticker)
            panel.loc[mask,f"fwd{h}"]=future.reindex(panel.loc[mask,"date"]).to_numpy()
    return panel
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="results/prism-v411")
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    panel=prepare()
    selected=[];folds=[]
    # Purge 35 calendar days before test begins to prevent training labels
    # from including prices from test window (20 sessions + holidays).
    for i,(train_start,train_end,test_start,test_end) in enumerate(FOLDS,1):
        start=pd.Timestamp(test_start)
        train_cut=min(pd.Timestamp(train_end),start-pd.Timedelta(days=35))
        tr=panel[(panel.date>=pd.Timestamp(train_start))&(panel.date<=train_cut)].copy()
        te=panel[(panel.date>=start)&(panel.date<=pd.Timestamp(test_end))].copy()
        if len(tr)<200 or te.empty:raise RuntimeError(f"Insufficient fold {i}")
        downside=LogisticRegression(max_iter=4000,class_weight="balanced").fit(_X(tr),tr.down)
        te["p_down"]=downside.predict_proba(_X(te))[:,1]
        te["risk_pass"]=(te.p_down<=.45)&(te.qqq_vs_ma20>-.03)&(te.qqq_ret5>-.06)
        for h in HORIZONS:
            training=tr.dropna(subset=[f"fwd{h}"])
            model=HistGradientBoostingRegressor(loss="absolute_error",learning_rate=.045,max_iter=180,max_leaf_nodes=15,l2_regularization=2,min_samples_leaf=45,random_state=42)
            model.fit(_XR(training),training[f"fwd{h}"])
            tr_pred=model.predict(_XR(training))
            # Conservative pooled training-only residual penalty, not a confidence interval.
            residual=float(np.sqrt(np.mean((training[f"fwd{h}"].to_numpy()-tr_pred)**2)))
            te[f"pred{h}"]=model.predict(_XR(te))
            te[f"edge{h}"]=te[f"pred{h}"]-.35*residual
        # Fixed equal-weight horizon blend; no tuning on validation.
        te["rank_edge"]=(te.edge5+te.edge10+te.edge20)/3
        chunks=[]
        for dt,g in te[te.risk_pass].groupby("date"):
            n=max(1,int(np.ceil(len(g)*.10)))
            s=g.nlargest(n,"rank_edge")
            s=s[s.rank_edge>0]
            if len(s):chunks.append(s)
        chosen=pd.concat(chunks) if chunks else te.iloc[:0]
        chosen=chosen.copy();chosen["fold"]=i
        cols=["ticker","date","fold","rank_edge","p_down","pred5","pred10","pred20","edge5","edge10","edge20","fwd5","fwd10","fwd20"]
        selected.append(chosen[cols])
        folds.append({"fold":i,"train_cutoff":str(train_cut.date()),"test_start":test_start,"test_end":test_end,"eligible":int(te.risk_pass.sum()),"selected":len(chosen),
                      "mean_realized":{str(h):float(chosen[f"fwd{h}"].mean()) if chosen[f"fwd{h}"].notna().any() else None for h in HORIZONS}})
    result=pd.concat(selected,ignore_index=True)
    report={"version":"v4.11","status":"RESEARCH ONLY / NO PORTFOLIO VALIDATION","selection_count":len(result),"folds":folds,"horizons":list(HORIZONS),
            "rules":{"risk_gate":"frozen v4.2","ranking":"equal-weight edge5/10/20, top 10% daily, positive only","purge":"35 calendar days","exit":"MA30 fixed for future separate replay"},
            "limitations":["Historical 50-stock universe selected with hindsight","Training features come from existing v4.2 panel; label integrity requires independent audit","Horizon return labels are close-to-future-close, not next-open entry-to-exit realized returns","No MA30 portfolio replay performed in this version","2026-10-06 horizon labels unavailable and are null, never imputed","Model comparisons on 2022-2026 sample are exploratory; future OOS required"]}
    result.to_csv(out/"selected_signals.csv",index=False)
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    rows="".join(f"<tr><td>{f['fold']}</td><td>{f['test_start']}–{f['test_end']}</td><td>{f['selected']}</td><td>{f['mean_realized']['5']}</td><td>{f['mean_realized']['10']}</td><td>{f['mean_realized']['20']}</td></tr>" for f in folds)
    (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.11</title><style>body{background:#0c1629;color:#e8f5ff;font:15px system-ui;padding:28px}h1{color:#5eead4}td,th{padding:12px;border-bottom:1px solid #345}table{border-collapse:collapse;width:100%}</style><h1>PRISM v4.11 · Multi-Horizon Entry Research</h1><p>Frozen MA30 exit; 5/10/20 session entry forecast; 35-day label purge. No portfolio result yet.</p><table><tr><th>Fold</th><th>Period</th><th>Selected</th><th>Realized D5</th><th>D10</th><th>D20</th></tr>"+rows+"</table><p>Read summary.json for limitations.</p>",encoding="utf-8")
    print(json.dumps({"status":"complete","selected":len(result),"folds":len(folds)}))
if __name__=="__main__":main()
