"""PRISM v4.76: fixed regime-gated expansion challenger, historical diagnostic only."""
import json, html
from pathlib import Path
import numpy as np
import pandas as pd
from .regime_adaptive_v436 import regimes
from .seed_c_multi_position_v471 import simulate
from .trend_v48 import load_prices
from .atr_frontier_v433 import CUTOFF

START=10_000_000.
def main():
    out=Path("results/prism-v476")
    out.mkdir(parents=True,exist_ok=True)
    candidates=pd.read_csv("results/prism-v474/candidate_quality_audit.csv")
    reference=pd.read_csv("results/prism-v474/daily_equity.csv")
    prior=json.loads(Path("results/prism-v474/summary.json").read_text())
    market=regimes(pd.Timestamp("2022-01-01"))
    candidates["entry_date"]=pd.to_datetime(candidates.entry_date).dt.strftime("%Y-%m-%d")
    candidates["is_base"]=candidates.is_base.astype(bool)
    candidates["regime"]=pd.to_datetime(candidates.entry_date).map(market)
    if candidates.regime.isna().any() or (candidates.regime=="UNKNOWN").any():
        raise RuntimeError("Missing prior-close market regime")
    # Frozen, explicit hypothesis: always preserve BASE; allow new compression
    # signals only outside BEAR. No fitted threshold or future outcomes.
    candidates["eligible"]=candidates.is_base | (candidates.regime!="BEAR")
    selected=candidates[candidates.eligible].copy()
    selected["rank_score"]=1000*selected.is_base.astype(int)+selected.quality_score
    selected.to_csv(out/"candidate_decisions.csv",index=False)
    baseline=reference[reference.strategy=="BASE_RVOL"].sort_values("date")
    if baseline.empty:raise RuntimeError("Missing baseline")
    dates=[pd.Timestamp(x) for x in pd.to_datetime(baseline.date).dt.normalize().unique()]
    prices={t:load_prices(t,"2021-01-01").loc[lambda x:x.index<=CUTOFF] for t in candidates.ticker.unique()}
    report={"version":"v4.76","status":"HISTORICAL_CHALLENGER_NOT_OOS",
            "rule":"Preserve all BASE; exclude expansion-only entries when prior-close QQQ regime is BEAR",
            "regime_counts":candidates.groupby(["regime","is_base","eligible"]).size().rename("n").reset_index().to_dict("records"),
            "strategies":{},"limitations":["Historical, repeatedly researched data; not independent OOS.",
            "Prior-close regime mapping required; market labels must be verified against implementation.",
            "Same-day stop-first daily OHLC fills and current-listed universe remain.",
            "No fees or slippage unless incorporated by upstream simulator.",
            "Portfolio simulation is authoritative, not standalone candidate gross returns.",
            "No live trading or automatic promotion."]}
    ledgers=[];curves=[]
    policies={"BASE_RVOL":candidates[candidates.is_base].assign(rank_score=lambda x:x.edge),
              "EXPANDED_BASE_PRIORITY":candidates.assign(rank_score=1000*candidates.is_base.astype(int)+candidates.quality_score),
              "V476_REGIME_GATE":selected}
    for name,z in policies.items():
        rows=z[["ticker","entry_date","exit_date","entry_price","exit_price","reason","rank_score"]].rename(columns={"rank_score":"edge"}).to_dict("records")
        perf,ledger,curve=simulate(rows,dates,prices,4,.25)
        report["strategies"][name]={**perf,"candidate_count":len(z)}
        for row in ledger:row["strategy"]=name
        for row in curve:row["strategy"]=name
        ledgers.extend(ledger);curves.extend(curve)
        if name in prior["strategies"]:
            old=prior["strategies"][name]
            if abs(perf["final_krw"]-old["final_krw"])>max(1,old["final_krw"]*1e-5) or perf["trades"]!=old["trades"]:
                raise RuntimeError("Reference replay mismatch "+name)
    pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
    trs="".join("<tr><td>%s</td><td>%.2f%%</td><td>%.2f%%</td><td>%s</td></tr>"%(html.escape(k),100*v["return"],100*v["mdd"],v["trades"]) for k,v in report["strategies"].items())
    (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.76</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}td,th{padding:12px;border-bottom:1px solid #456}table{border-collapse:collapse}</style><h1>PRISM v4.76 Regime Gate</h1><p>Historical challenger; not independent OOS.</p><table><tr><th>Strategy</th><th>Return</th><th>MDD</th><th>Trades</th></tr>"+trs+"</table></html>",encoding="utf-8")
    print(json.dumps({"status":"V476_COMPLETE","strategies":report["strategies"]},default=str))
if __name__=="__main__":main()
