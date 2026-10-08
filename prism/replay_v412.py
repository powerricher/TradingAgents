"""v4.12: replay v4.11 frozen entry signals using the fixed MA30 exit.
No refit or parameter selection from realized portfolio performance.
"""
import argparse,json,html
from pathlib import Path
import pandas as pd
from .trend_v48 import load_prices,portfolio
from .trend_v49 import exit_trade

def replay(signals, out, label):
    rows=[];issues=[];prices={}
    for ticker,g in signals.groupby("ticker"):
        try:
            d=load_prices(ticker,g.date.min());prices[ticker]=d
            for s in g.itertuples(index=False):
                dt=pd.Timestamp(s.date).normalize()
                if dt not in d.index:
                    issues.append([ticker,str(dt.date()),"missing signal date"]);continue
                z=exit_trade(d,dt,"CONFIRM30_1")
                if z is None:
                    issues.append([ticker,str(dt.date()),"unresolved"]);continue
                e,x,entry,exit_price,reason=z
                rows.append(dict(ticker=ticker,mode="MA30",signal_date=str(dt.date()),entry_date=str(d.index[e].date()),exit_date=str(d.index[x].date()),entry_price=entry,exit_price=exit_price,edge=float(s.edge),reason=reason,ret=exit_price/entry-1))
        except Exception as exc:issues.append([ticker,"download",str(exc)])
    if not rows:return {"status":"INVALID","issues":issues},[],[],rows
    if len(rows)!=len(signals):
        return {"status":"INCOMPLETE","signals":len(signals),"completed":len(rows),"issues":issues},[],[],rows
    try:
        result,ledger,curve=portfolio(rows,prices,"MA30",rotation=False)
        return {"status":"COMPLETE",**result,"signals":len(signals),"issues":issues},ledger,curve,rows
    except Exception as exc:
        return {"status":"INVALID","error":str(exc),"issues":issues},[],[],rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--challenger",default="results/prism-v411/selected_signals.csv")
    ap.add_argument("--baseline",default="results/prism-v45/report.json")
    ap.add_argument("--out",default="results/prism-v412")
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    challenger=pd.read_csv(a.challenger)
    challenger["edge"]=challenger["rank_edge"]
    baseline=pd.DataFrame(json.loads(Path(a.baseline).read_text())["selected_rows"])
    report={"version":"v4.12","status":"RESEARCH / NOT INDEPENDENT OOS","comparisons":{},"limitations":["Same historical sample used to develop MA30; no independent OOS validation","Historical 50-stock universe may have survivorship bias","Close-based signal; entry next session open","Daily-close MDD excludes intraday lows","Stop touch uses assumed 20bps adverse slippage","v4.11 signal labels are close-to-future-close; portfolio returns use actual next-open entries"]}
    all_ledgers=[];all_curves=[];all_trades=[]
    for label,signals in [("D1_BASELINE",baseline),("MULTI_HORIZON",challenger)]:
        result,ledger,curve,rows=replay(signals,out,label)
        report["comparisons"][label]=result
        for r in ledger:r["variant"]=label
        for r in curve:r["variant"]=label
        for r in rows:r["variant"]=label
        all_ledgers.extend(ledger);all_curves.extend(curve);all_trades.extend(rows)
    pd.DataFrame(all_trades).to_csv(out/"trades.csv",index=False)
    pd.DataFrame(all_ledgers).to_csv(out/"ledger.csv",index=False)
    pd.DataFrame(all_curves).to_csv(out/"daily_equity.csv",index=False)
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    trs="".join("<tr><td>"+html.escape(k)+"</td><td>"+(f"{v['return']:.2%}" if "return" in v else v["status"])+"</td><td>"+(f"{v['daily_close_mdd']:.2%}" if "daily_close_mdd" in v else "—")+"</td><td>"+str(v.get("trades","—"))+"</td></tr>" for k,v in report["comparisons"].items())
    (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.12</title><style>body{background:#0c1629;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#6ee7c2}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.12 — D1 vs Multi Horizon / Fixed MA30</h1><p>Initial KRW 10M, 10bps round-trip, stop stress 20bps. Exploratory.</p><table><tr><th>Entry</th><th>Compound Return</th><th>Daily-close MDD</th><th>Executed trades</th></tr>"+trs+"</table><p>Check summary.json for limitations.</p>",encoding="utf-8")
    print(json.dumps(report["comparisons"],ensure_ascii=False))
if __name__=="__main__":main()
