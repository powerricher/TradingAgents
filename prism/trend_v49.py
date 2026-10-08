"""PRISM v4.9: confirm trend breaks instead of selling on first MA20 breach.
Frozen v4.5 signals, same next-open entry, no rotation. Research only.
"""
import argparse,html,json
from pathlib import Path
import numpy as np
import pandas as pd
from .trend_v47 import atr
from .trend_v48 import load_prices,exit_trade as baseline_exit,portfolio,STOP_SLIPPAGE

# Predeclared variants: original E60 and 3 slower trend-confirmation exits.
MODES={"E60_BASE":(20,1,3,60),
       "CONFIRM20_2":(20,2,3,90),
       "CONFIRM30_1":(30,1,3,90),
       "CONFIRM30_2":(30,2,4,120)}

def exit_trade(d,dt,mode):
    if mode=="E60_BASE":
        return baseline_exit(d,dt,"E60")
    window,confirm,k,maxdays=MODES[mode]
    e=d.index.get_loc(dt)+1
    if e>=len(d):return None
    entry=float(d.Open.iloc[e])
    if not np.isfinite(entry) or entry<=0:return None
    volatility=atr(d)
    peak=entry
    stop=-np.inf
    below_count=0
    for j in range(e,min(len(d),e+maxdays+1)):
        op,lo,cl=map(float,(d.Open.iloc[j],d.Low.iloc[j],d.Close.iloc[j]))
        # The stop was determined at the PREVIOUS session's close.
        if j>e and np.isfinite(stop):
            if op<=stop:return e,j,entry,op,"gap_stop"
            if lo<=stop:return e,j,entry,stop*(1-STOP_SLIPPAGE),"stop_touch_stress"
        if j-e>=maxdays:
            return e,j,entry,op,"time_exit_next_open"
        ma=float(d.Close.iloc[max(0,j-window+1):j+1].mean())
        below_count=below_count+1 if cl<ma else 0
        if below_count>=confirm and j+1<len(d):
            return e,j+1,entry,float(d.Open.iloc[j+1]),"confirmed_ma_break_next_open"
        peak=max(peak,cl)
        if pd.notna(volatility.iloc[j]):
            stop=max(stop,peak-k*float(volatility.iloc[j]))
    return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--signals",default="results/prism-v45/report.json")
    p.add_argument("--out",default="results/prism-v49")
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    src=json.loads(Path(a.signals).read_text())
    selected=pd.DataFrame(src["selected_rows"])
    if selected.empty:raise ValueError("No frozen entry signals")
    prices={};rows=[];issues=[]
    for ticker,g in selected.groupby("ticker"):
        try:
            d=load_prices(ticker,g.date.min());prices[ticker]=d
            for s in g.itertuples(index=False):
                dt=pd.Timestamp(s.date).normalize()
                if dt not in d.index:
                    issues.append([ticker,str(dt.date()),"missing_signal"]);continue
                for mode in MODES:
                    z=exit_trade(d,dt,mode)
                    if z is None:
                        issues.append([ticker,str(dt.date()),mode+" incomplete"]);continue
                    e,x,entry,exit_price,reason=z
                    rows.append(dict(ticker=ticker,mode=mode,signal_date=str(dt.date()),entry_date=str(d.index[e].date()),exit_date=str(d.index[x].date()),entry_price=entry,exit_price=exit_price,edge=float(s.edge),reason=reason,ret=exit_price/entry-1,holding_sessions=x-e))
        except Exception as exc:
            issues.append([ticker,"download",str(exc)])
    trades=pd.DataFrame(rows)
    if trades.empty:raise RuntimeError("No trades")
    report={"version":"v4.9","status":"CHALLENGER / NOT VALIDATED","signals":len(selected),
            "completed_by_mode":trades.groupby("mode").size().to_dict(),
            "issues":issues,"strategies":{},"limitations":[
            "Same 2022-2026 historical sample used to design these exits; independent OOS needed",
            "Historical 50-stock universe not point-in-time; survivorship bias",
            "D+1 frozen entry model, not multi-horizon trained",
            "Daily-close MDD excludes intraday drawdown",
            "ATR stop-touch haircut is a hypothetical 20bps, not executable fill guarantee",
            "Portfolio is one-position full capital, no rotation; costs included by v4.8 engine"]}
    ledgers=[];curves=[]
    for mode in MODES:
        sub=trades[trades["mode"]==mode].to_dict("records")
        if len(sub)!=len(selected):
            report["strategies"][mode]={"status":"INCOMPLETE","n":len(sub)}
            continue
        try:
            result,ledger,curve=portfolio(sub,prices,mode,rotation=False)
            result["avg_holding_sessions_all_signals"]=float(trades.loc[trades["mode"]==mode,"holding_sessions"].mean())
            report["strategies"][mode]=result
            for x in ledger:x["strategy"]=mode
            for x in curve:x["strategy"]=mode
            ledgers.extend(ledger);curves.extend(curve)
        except Exception as exc:
            report["strategies"][mode]={"status":"INVALID","error":str(exc)}
    trades.to_csv(out/"trades.csv",index=False)
    pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    trs="".join("<tr><td>"+html.escape(k)+"</td><td>"+(f"{v['return']:.2%}" if "return" in v else "INVALID")+"</td><td>"+(f"{v['daily_close_mdd']:.2%}" if "daily_close_mdd" in v else "—")+"</td><td>"+str(v.get("trades","—"))+"</td></tr>" for k,v in report["strategies"].items())
    page="""<!doctype html><meta charset="utf-8"><title>PRISM v4.9</title><style>body{background:#091524;color:#e9f3ff;font:15px system-ui;padding:24px}h1{color:#5eeac0}table{border-collapse:collapse;width:100%}td,th{padding:13px;border-bottom:1px solid #345}</style><h1>PRISM v4.9 · Confirmed Trend Exits</h1><p>2022–2026 | 50 technology stocks | frozen signals | 10M KRW initial | 10bps round trip | 20bps stop-touch stress</p><table><tr><th>Exit</th><th>Compound return</th><th>Daily-close MDD</th><th>Executed</th></tr>"""+trs+"""</table><p>Exploratory, not independent OOS. See summary.json for incomplete samples and assumptions.</p>"""
    (out/"report.html").write_text(page,encoding="utf-8")
    print(json.dumps({"status":"generated","strategies":report["strategies"],"issues":len(issues)},ensure_ascii=False))
if __name__=="__main__":main()
