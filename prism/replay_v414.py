"""PRISM v4.14: replay 80-stock D+1 variants with identical fixed MA30 exits.
Research only; same 80-stock fitted model for CORE50, TECH80, QUALITY.
"""
import argparse,json,html
from pathlib import Path
import pandas as pd
from .replay_v412 import replay

VARIANTS=("CORE50","TECH80","TECH80_QUALITY")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--signals",default="results/prism-v413/signals.csv")
    p.add_argument("--out",default="results/prism-v414")
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    signals=pd.read_csv(a.signals,dtype={"ticker":str,"date":str,"variant":str})
    required={"ticker","date","variant","edge"}
    if not required.issubset(signals.columns):
        raise ValueError("Missing columns: "+str(required-set(signals.columns)))
    report={"version":"v4.14","status":"RESEARCH / NOT INDEPENDENT OOS",
      "initial_capital_krw":10000000,"exit":"fixed MA30 CONFIRM30_1",
      "round_trip_cost_bps":10,"stop_touch_stress_bps":20,
      "variants":{},"limitations":[
        "CORE50 and TECH80 both use the 80-stock trained estimator, NOT original 50-stock v4.12 baseline",
        "Same 2022-2026 sample repeatedly used for research; no independent validation",
        "80-stock membership selected retrospectively; survivorship and listing-date bias",
        "Daily-close MDD excludes intraday troughs",
        "Close-based signals enter next session open",
        "Quality thresholds are exploratory; 40 signals may be insufficient",
        "No guarantee of future profitability"]}
    ledgers=[];curves=[];trades=[]
    for name in VARIANTS:
        group=signals.loc[signals.variant.eq(name)].copy()
        result,ledger,curve,rows=replay(group,out,name)
        report["variants"][name]=result
        for x in ledger:x["variant"]=name
        for x in curve:x["variant"]=name
        for x in rows:x["variant"]=name
        ledgers.extend(ledger);curves.extend(curve);trades.extend(rows)
    pd.DataFrame(trades).to_csv(out/"trades.csv",index=False)
    pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    rows="".join("<tr><td>"+html.escape(k)+"</td><td>"+str(v.get("signals","—"))+"</td><td>"+(f"{v['return']:.2%}" if "return" in v else html.escape(v["status"]))+"</td><td>"+(f"{v['daily_close_mdd']:.2%}" if "daily_close_mdd" in v else "—")+"</td><td>"+str(v.get("trades","—"))+"</td></tr>" for k,v in report["variants"].items())
    (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.14</title><style>body{background:#0c1728;color:#e9f5ff;font:15px system-ui;padding:25px}h1{color:#5eead4}table{width:100%;border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.14 · 50 vs 80 vs Quality</h1><p>Fixed MA30 exits · 10M KRW · 10bps round trip · 20bps stop-touch stress</p><table><tr><th>Variant</th><th>Signals</th><th>Compound return</th><th>Daily-close MDD</th><th>Executed</th></tr>"+rows+"</table><p>CORE50 uses the 80-stock trained model. This is NOT the original 50-stock baseline. Research only.</p>",encoding="utf-8")
    print(json.dumps({"status":"generated","variants":{k:{x:v.get(x) for x in ("status","signals","return","daily_close_mdd","trades")} for k,v in report["variants"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
