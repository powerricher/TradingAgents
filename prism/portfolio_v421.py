"""PRISM v4.21: one-slot vs three independent compounding sleeves.
Signal-time rankings are frozen from v4.20. Same MA30 exits and costs.
Each sleeve starts with 10M/slots; no leverage or transfer between sleeves.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices
MODES={"RIDGE":"rank_RIDGE","VCP_ALL":"rank_VCP_ALL","VCP_VOLATILITY":"rank_VCP_VOLATILITY"}
CAPITAL=10_000_000.
COST=0.001
def replay(g,prices,slots):
    # Deterministic allocation. At most one new position per slot per session.
    # Positions whose exit date is today release funds, but no same-day re-entry.
    balances=[CAPITAL/slots]*slots
    positions=[None]*slots
    closed=[];curve=[];skipped=0;accepted=0
    entries={}
    for d,q in g.groupby("entry_date"):
        entries[pd.Timestamp(d)]=q.sort_values(["rank","ticker"],ascending=[False,True]).to_dict("records")
    days=sorted(set().union(*(set(d.index) for d in prices.values())))
    first=min(entries);last=max(pd.Timestamp(x) for x in g.exit_date)
    days=[d for d in days if first<=d<=last]
    for day in days:
        sold=set()
        for i,pos in enumerate(positions):
            if pos and pd.Timestamp(pos["exit_date"])==day:
                proceeds=pos["shares"]*pos["exit_price"]*(1-COST/2)
                balances[i]=proceeds
                closed.append({"slot":i+1,"ticker":pos["ticker"],"entry_date":pos["entry_date"],"exit_date":str(day.date()),"capital_after":proceeds,"entry_capital":pos["entry_capital"],"net_return":proceeds/pos["entry_capital"]-1,"reason":pos["reason"]})
                positions[i]=None;sold.add(i)
        pool=entries.get(day,[])
        occupied={p["ticker"] for p in positions if p}
        free=[i for i,p in enumerate(positions) if p is None and i not in sold]
        taken=0
        for cand in pool:
            if not free:break
            if cand["ticker"] in occupied:continue
            i=free.pop(0)
            entry=float(cand["entry_price"])
            if entry<=0 or not np.isfinite(entry):raise ValueError("Bad entry")
            cash=balances[i]
            positions[i]={**cand,"shares":cash*(1-COST/2)/entry,"entry_capital":cash}
            balances[i]=0.;occupied.add(cand["ticker"]);taken+=1;accepted+=1
        skipped+=len(pool)-taken
        equity=0.
        for i,pos in enumerate(positions):
            if pos is None:equity+=balances[i];continue
            d=prices[pos["ticker"]]
            if day not in d.index:
                # NYSE and NASDAQ should share the trading calendar.
                raise RuntimeError(f"Missing mark {pos['ticker']} {day}")
            equity+=pos["shares"]*float(d.loc[day,"Close"])
        curve.append({"date":str(day.date()),"equity":equity})
    if any(positions):raise RuntimeError("Open position at horizon")
    final=sum(balances)
    vals=np.array([r["equity"] for r in curve])
    peaks=np.maximum.accumulate(np.r_[CAPITAL,vals])[1:]
    return {"final_krw":float(final),"compound_return":float(final/CAPITAL-1),"daily_close_mdd":float(np.min(vals/peaks-1)),
      "executed":len(closed),"accepted_signals":accepted,"skipped_signals":skipped,"slots":slots},closed,curve

def main():
    p=argparse.ArgumentParser();p.add_argument("--scored",default="results/prism-v420/scored_signals.csv");p.add_argument("--out",default="results/prism-v421");a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.scored)
    df["entry_date"]=pd.to_datetime(df.entry_date)
    df["exit_date"]=pd.to_datetime(df.exit_date)
    if df.empty or df[["ticker","entry_date","exit_date","entry_price","exit_price"]+list(MODES.values())].isna().any().any():
        raise RuntimeError("Missing candidate records")
    prices={}
    for ticker,g in df.groupby("ticker"):prices[ticker]=load_prices(ticker,g.date.min())
    result={"version":"v4.21","status":"RESEARCH / NOT INDEPENDENT OOS","initial_krw":CAPITAL,"variants":{},
      "assumptions":["One-slot control and 3 equal initial capital sleeves, each compounds independently.",
      "At most 3 concurrent distinct tickers, no borrowing and no capital transfers.",
      "One new position per sleeve per session; no re-entry on the same day as a sale.",
      "Daily close marked-to-market MDD excludes intraday troughs.",
      "Frozen signals and fixed MA30 exits; round-trip cost 10bps, stop touch stress already embedded in exit price.",
      "Retrospective fixed 80-stock universe and repeated historical optimization prevent OOS claims."]}
    alltr=[];allcurves=[]
    for name,col in MODES.items():
        records=df[["ticker","entry_date","exit_date","entry_price","exit_price","reason",col]].rename(columns={col:"rank"}).copy()
        for slots in (1,3):
            key=f"{name}_{slots}S"
            perf,ledger,curve=replay(records,prices,slots)
            result["variants"][key]=perf
            for x in ledger:x["variant"]=key
            for x in curve:x["variant"]=key
            alltr+=ledger;allcurves+=curve
    pd.DataFrame(alltr).to_csv(out/"ledger.csv",index=False)
    pd.DataFrame(allcurves).to_csv(out/"daily_equity.csv",index=False)
    (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['compound_return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['executed']}</td><td>{v['skipped_signals']}</td></tr>" for k,v in result["variants"].items())
    (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.21</title><style>body{background:#0b1729;color:#edf7ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.21 · 1 Slot vs 3 Slots</h1><p>10M KRW initial · equal sleeves · frozen MA30 exit · research only</p><table><tr><th>Strategy</th><th>Compound Return</th><th>Daily MDD</th><th>Trades</th><th>Skipped</th></tr>"+trs+"</table></html>",encoding="utf-8")
    print(json.dumps({"status":"generated","variants":result["variants"]},ensure_ascii=False))
if __name__=="__main__":main()
