"""PRISM v4.71 — C1 frozen signals, multi-position risk audit.
One portfolio, cash ledger, next-open entries, same-day stop processing.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import COST,load_prices
from .atr_frontier_v433 import CUTOFF
START=10_000_000.
NAME="C1_VOL_EXPANSION_TP8_SL4_H5"
CONFIGS={"FULL_1":(1,1.),"SINGLE_25":(1,.25),"SINGLE_50":(1,.50),
         "FOUR_25":(4,.25),"TWO_50":(2,.50)}
def simulate(rows,calendar,prices,slots,weight):
    byday={}
    for r in rows:byday.setdefault(r["entry_date"],[]).append(r)
    for day in byday:byday[day].sort(key=lambda r:(-float(r["edge"]),r["ticker"]))
    cash=START;positions=[];ledger=[];curve=[];skipped=0
    for dt in calendar:
        day=dt.strftime("%Y-%m-%d")
        sold_today=False
        # Sell existing positions before opening new ones; no replacement on an exit day.
        for p in list(positions):
            if p["exit_date"]==day:
                proceeds=p["shares"]*p["exit_price"]*(1-COST/2)
                cash+=proceeds
                ledger.append({"ticker":p["ticker"],"entry_date":p["entry_date"],"exit_date":day,
                  "entry_notional":p["notional"],"proceeds":proceeds,
                  "pnl_krw":proceeds-p["notional"],"reason":p["reason"]})
                positions.remove(p);sold_today=True
        today=byday.get(day,[])
        selected=0
        if not sold_today:
            # Mark existing positions using current close for capital sizing.
            # New entries are sized on prior-close equity instead to avoid lookahead.
            previous=calendar[calendar.index(dt)-1] if calendar.index(dt)>0 else None
            prior_equity=cash
            for p in positions:
                d=prices[p["ticker"]]
                if previous is None or previous not in d.index:raise RuntimeError("Missing prior close")
                prior_equity+=p["shares"]*float(d.loc[previous,"Close"])
            target=prior_equity*weight
            held={p["ticker"] for p in positions}
            for r in today:
                if len(positions)>=slots:break
                if r["ticker"] in held:continue
                # Equal fixed equity sleeve per position; unused cash remains idle.
                budget=min(target,cash)
                if budget<target*.999:break
                shares=budget*(1-COST/2)/float(r["entry_price"])
                cash-=budget
                p={**r,"shares":shares,"notional":budget}
                positions.append(p);held.add(r["ticker"]);selected+=1
                if r["exit_date"]==day:
                    proceeds=shares*float(r["exit_price"])*(1-COST/2)
                    cash+=proceeds
                    ledger.append({"ticker":r["ticker"],"entry_date":day,"exit_date":day,
                     "entry_notional":budget,"proceeds":proceeds,
                     "pnl_krw":proceeds-budget,"reason":r["reason"]})
                    positions.remove(p)
                    # Do not recycle same-day proceeds into another entry.
                    break
        skipped+=len(today)-selected
        equity=cash
        for p in positions:
            d=prices[p["ticker"]]
            if dt not in d.index:raise RuntimeError("Missing valuation close "+p["ticker"]+" "+day)
            equity+=p["shares"]*float(d.loc[dt,"Close"])
        if cash < -1e-6:raise RuntimeError("Negative cash")
        curve.append({"date":day,"equity":float(equity),"cash":float(cash),"positions":len(positions)})
    if positions:raise RuntimeError("Unclosed positions: "+str([p["ticker"] for p in positions]))
    v=np.array([x["equity"] for x in curve]);peak=np.maximum.accumulate(np.r_[START,v])[1:]
    return {"return":float(v[-1]/START-1),"mdd":float(np.min(v/peak-1)),
      "trades":len(ledger),"same_day_exits":sum(x["entry_date"]==x["exit_date"] for x in ledger),
      "skipped":skipped,"final_krw":float(v[-1]),
      "average_positions":float(np.mean([x["positions"] for x in curve]))},ledger,curve
def main():
    out=Path("results/prism-v471");out.mkdir(parents=True,exist_ok=True)
    z=pd.read_csv("results/prism-v468/candidate_exit_scenarios.csv")
    z=z[(z.detector=="C1_VOL_EXPANSION")&(z.exit_rule=="TP8_SL4")&(z.hold==5)].copy()
    ref=pd.read_csv("results/prism-v468/daily_equity.csv")
    ref=ref[ref.strategy==NAME].sort_values("date")
    original=pd.read_csv("results/prism-v468/ledger.csv")
    original=original[original.strategy==NAME]
    if z.empty or ref.empty or original.empty:raise RuntimeError("Missing v4.68 frozen baseline")
    if z.duplicated(["ticker","entry_date"]).any():raise RuntimeError("Duplicate candidate")
    prices={t:load_prices(t,"2021-01-01").loc[lambda x:x.index<=CUTOFF] for t in z.ticker.unique()}
    for r in z.itertuples(index=False):
        d=prices[r.ticker];day=pd.Timestamp(r.entry_date)
        if day not in d.index or abs(float(d.loc[day,"Open"])/float(r.entry_price)-1)>.001:
            raise RuntimeError("Price snapshot mismatch "+r.ticker+" "+str(day))
    calendar=[pd.Timestamp(x) for x in pd.to_datetime(ref.date).dt.normalize().unique()]
    records=z[["ticker","entry_date","exit_date","entry_price","exit_price","edge","reason"]].to_dict("records")
    report={"version":"v4.71","status":"HISTORICAL MULTIPOSITION RESEARCH NOT OOS",
      "strategies":{},"limitations":[
      "Frozen C1 TP8 SL4 H5 candidates; no entry signal optimization.",
      "Equal capital sleeves based on previous-close equity; cash must cover full sleeve.",
      "Positions exited at current session before new entries; no new entry on exit days.",
      "Entry-day exits are processed immediately, with no same-day capital recycling.",
      "Daily OHLC stop-first convention inherited from v4.68; no minute-level sequencing.",
      "Close-to-close MDD excludes intraday troughs and commissions/stop fills are modeled.",
      "Historical present-day universe excludes delisted names; not untouched OOS.",
      "Diversification by number of names does not guarantee sector or factor diversification.",
      "No live orders, notifications, or automatic promotion."]}
    ledgers=[];curves=[]
    for name,(slots,weight) in CONFIGS.items():
        perf,ledger,curve=simulate(records,calendar,prices,slots,weight)
        report["strategies"][name]=perf
        for r in ledger:r["strategy"]=name
        for r in curve:r["strategy"]=name
        ledgers.extend(ledger);curves.extend(curve)
    base=report["strategies"]["FULL_1"]
    target=float(ref.equity.iloc[-1])
    if abs(base["final_krw"]-target)>max(1.,target*.00001) or base["trades"]!=len(original):
        raise RuntimeError("Full capital baseline mismatch with v4.68")
    if len(report["strategies"])!=5:raise RuntimeError("Missing portfolio variant")
    pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['mdd']:.2%}</td><td>{v['trades']}</td><td>{v['same_day_exits']}</td></tr>" for k,v in report["strategies"].items())
    (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.71</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.71 · Seed C Multi-Position Audit</h1><p>Historical only. Baseline reconciliation required.</p><table><tr><th>Policy</th><th>Return</th><th>MDD</th><th>Trades</th><th>Same-day</th></tr>"+lines+"</table>",encoding="utf-8")
    print(json.dumps({"status":"VALID","strategies":len(report["strategies"])}))
if __name__=="__main__":main()
