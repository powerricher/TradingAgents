"""PRISM v4.10 — trend exit robustness diagnostics.
No new alpha model, no independent holdout claims.
"""
import argparse,json,html
from pathlib import Path
import pandas as pd
import numpy as np
from . import trend_v49
from .trend_v48 import load_prices,portfolio
MODES={"MA25":(25,1,3,90),"MA30":(30,1,3,90),"MA35":(35,1,3,90),"MA40":(40,1,3,90)}
def metrics(ledger,starting=10_000_000):
    if not ledger:return {"trades":0,"return":None}
    last=float(ledger[-1]["capital_after"])
    return {"trades":len(ledger),"return":last/starting-1}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--signals",default="results/prism-v45/report.json")
    p.add_argument("--out",default="results/prism-v410")
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    selected=pd.DataFrame(json.loads(Path(a.signals).read_text())["selected_rows"])
    if selected.empty:raise ValueError("No frozen signals")
    # The v4.9 function reads this module-level configuration.
    trend_v49.MODES.update(MODES)
    prices={};rows=[];issues=[]
    for ticker,g in selected.groupby("ticker"):
        try:
            d=load_prices(ticker,g.date.min());prices[ticker]=d
            for s in g.itertuples(index=False):
                dt=pd.Timestamp(s.date).normalize()
                if dt not in d.index:
                    issues.append([ticker,str(dt.date()),"missing signal date"]);continue
                for mode in MODES:
                    z=trend_v49.exit_trade(d,dt,mode)
                    if z is None:
                        issues.append([ticker,str(dt.date()),mode+" unresolved"]);continue
                    e,x,entry,exit_price,reason=z
                    rows.append({"ticker":ticker,"mode":mode,"signal_date":str(dt.date()),"entry_date":str(d.index[e].date()),"exit_date":str(d.index[x].date()),"entry_price":entry,"exit_price":exit_price,"ret":exit_price/entry-1,"edge":float(s.edge),"reason":reason,"holding_sessions":x-e})
        except Exception as exc:
            issues.append([ticker,"download",str(exc)])
    trades=pd.DataFrame(rows)
    if trades.empty:raise RuntimeError("No trades generated")
    report={"version":"v4.10","status":"EXPLORATORY; NOT INDEPENDENT OOS","signals":len(selected),"issues":issues,"strategies":{},"limitations":["All MA choices inspected on historical sample already used to develop MA30; 2025-2026 segment is NOT independent OOS.","Historical tech universe selected with hindsight; survivorship bias.","Frozen entry predictor is D+1, not a multi-horizon model.","Daily-close MDD misses intraday losses.","Portfolio simulation uses fixed entry selection and one full-capital position."]}
    ledgers=[];curves=[]
    for mode in MODES:
        sub=trades[trades["mode"]==mode]
        if len(sub)!=len(selected):
            report["strategies"][mode]={"status":"INCOMPLETE","n":len(sub)}
            continue
        try:
            result,ledger,curve=portfolio(sub.to_dict("records"),prices,mode,rotation=False)
            # Ledger is the actual sequence of executed positions, not all candidate signals.
            executed=pd.DataFrame(ledger)
            result["calendar_year_closed_trades"]={str(y):int(n) for y,n in executed.groupby(executed.exit_date.str[:4]).size().items()}
            result["yearly_realized_pnl_krw"]={}
            prev=10_000_000.
            for row in ledger:
                year=row["exit_date"][:4]
                result["yearly_realized_pnl_krw"][year]=result["yearly_realized_pnl_krw"].get(year,0)+row["capital_after"]-prev
                prev=row["capital_after"]
            result["entry_ticker_counts"]=executed.ticker.value_counts().to_dict()
            result["cost_stress_note"]="Base cost 10bps and stop-touch stress 20bps; further sensitivity requires full rerun."
            report["strategies"][mode]=result
            for row in ledger:row["strategy"]=mode
            for row in curve:row["strategy"]=mode
            ledgers.extend(ledger);curves.extend(curve)
        except Exception as exc:
            report["strategies"][mode]={"status":"INVALID","error":str(exc)}
    trades.to_csv(out/"trades.csv",index=False)
    pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    trs="".join("<tr><td>"+k+"</td><td>"+(f"{v['return']:.2%}" if "return" in v else "INVALID")+"</td><td>"+(f"{v['daily_close_mdd']:.2%}" if "daily_close_mdd" in v else "—")+"</td><td>"+str(v.get("trades","—"))+"</td></tr>" for k,v in report["strategies"].items())
    (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.10</title><style>body{background:#0c1728;color:#e9f3ff;font:15px system-ui;margin:28px}h1{color:#63dfbd}table{width:100%;border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.10 · MA Robustness</h1><p>Research only — NOT independent OOS. Frozen 253 signals, 10M KRW, 10bps costs + stop stress.</p><table><tr><th>Exit</th><th>Compound return</th><th>Daily-close MDD</th><th>Trades</th></tr>"+trs+"</table><p>See summary.json for yearly realized P&L, ticker concentration, missing signals and limitations.</p>",encoding="utf-8")
    print(json.dumps({"strategies":report["strategies"],"issues":len(issues)},ensure_ascii=False))
if __name__=="__main__":main()
