"""PRISM v4.6 portfolio and signal diagnostics; consumes v4.5 frozen signals/trades."""
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v45 import UNIVERSE
def calc(trades,mode,capital=10_000_000,cost=.001):
 g=trades[trades["mode"]==mode].sort_values(["entry_date","edge","ticker"],ascending=[True,False,True]).copy()
 cash=float(capital);last_exit="";ledger=[];skipped=0;peak=cash;mdd=0.
 for r in g.itertuples(index=False):
  if r.entry_date<=last_exit:skipped+=1;continue
  before=cash;net=float(r.ret)-cost;cash*=1+net;peak=max(peak,cash);mdd=min(mdd,cash/peak-1)
  ledger.append({"ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,"gross_return":float(r.ret),"net_return":net,"edge":float(r.edge),"capital_before":before,"capital_after":cash,"pnl":cash-before})
  last_exit=r.exit_date
 return {"mode":mode,"trades":len(ledger),"skipped":skipped,"ending_capital":cash,"net_pnl":cash-capital,"total_return":cash/capital-1,"closed_trade_mdd":mdd},ledger
def stats(g,cost):
 r=g.ret.to_numpy(dtype=float)-cost
 pos=r[r>0].sum();neg=-r[r<0].sum()
 return {"n":len(r),"mean_net":float(r.mean()) if len(r) else None,"median_net":float(np.median(r)) if len(r) else None,"win_net":float(np.mean(r>0)) if len(r) else None,"pf_net":float(pos/neg) if neg>0 else None,"worst_net":float(r.min()) if len(r) else None}
def main():
 p=argparse.ArgumentParser();p.add_argument("--signals",default="results/prism-v45/report.json");p.add_argument("--trades",default="results/prism-v45-execution/trades.csv");p.add_argument("--out",default="results/prism-v46");p.add_argument("--cost-bps",type=float,default=10.);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 src=json.loads(Path(a.signals).read_text());selected=pd.DataFrame(src["selected_rows"]);trades=pd.read_csv(a.trades,dtype={"ticker":str,"entry_date":str,"exit_date":str,"mode":str})
 required={"ticker","entry_date","exit_date","mode","ret","edge"}
 if not required.issubset(trades.columns):raise ValueError("Missing trade columns: "+str(required-set(trades.columns)))
 if len(trades)!=len(selected)*3:raise ValueError("Incomplete A/B/C trades: "+str(len(trades)))
 cost=a.cost_bps/10000.;report={"version":"v4.6","cost_bps_round_trip":a.cost_bps,"signals":len(selected),"universe":len(UNIVERSE),"strategy":{}}
 ledgers=[]
 for mode in "ABC":
  g=trades[trades["mode"]==mode];portfolio,ledger=calc(trades,mode,cost=cost)
  if len(g)!=len(selected):raise ValueError("Missing mode "+mode)
  report["strategy"][mode]={"trade_statistics":stats(g,cost),"portfolio":portfolio,"annual":{str(y):stats(q,cost) for y,q in g.groupby(g.entry_date.str[:4])}}
  for item in ledger:item["mode"]=mode
  ledgers+=ledger
 report["selected_tickers"]=sorted(selected.ticker.unique().tolist())
 report["never_selected"]=sorted(set(UNIVERSE)-set(report["selected_tickers"]))
 report["signals_by_halfyear"]={str(k):int(v) for k,v in selected.groupby(selected.date.str[:4]+"-H"+np.where(selected.date.str[5:7].astype(int)<=6,"1","2")).size().items()}
 report["folds"]=src["meta"].get("folds",[])
 pd.DataFrame(ledgers).to_csv(out/"portfolio_ledger.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
 import html
 rows="".join("<tr><td>"+m+"</td><td>"+str(round(report["strategy"][m]["portfolio"]["ending_capital"]))+"</td><td>"+f'{report["strategy"][m]["portfolio"]["total_return"]:.2%}'+"</td><td>"+f'{report["strategy"][m]["portfolio"]["closed_trade_mdd"]:.2%}'+"</td><td>"+str(report["strategy"][m]["portfolio"]["trades"])+"</td><td>"+f'{report["strategy"][m]["trade_statistics"]["pf_net"]:.2f}'+"</td></tr>" for m in "ABC")
 page="""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRISM v4.6</title><style>body{background:#0b1323;color:#e8f3ff;font:15px system-ui;margin:32px;max-width:1000px}h1{color:#6de2c3}table{width:100%;border-collapse:collapse;background:#15243a}td,th{padding:12px;border-bottom:1px solid #31465c;text-align:right}td:first-child,th:first-child{text-align:left}.muted{color:#a9bed3}</style><h1>PRISM v4.6 · Portfolio Diagnostics</h1><p class="muted">1,000만원 최초 자본 · 단일 종목 전액 재투자 · 왕복 10bps · 매도일 신규 진입 금지</p><table><tr><th>전략</th><th>최종자산 (원)</th><th>누적수익</th><th>종료시점 MDD*</th><th>실행 거래</th><th>순 PF</th></tr>"""+rows+"""</table><h2>선택되지 않은 종목</h2><p>"""+html.escape(", ".join(report["never_selected"]))+"""</p><h2>반기별 신호</h2><pre>"""+html.escape(json.dumps(report["signals_by_halfyear"],indent=2))+"""</pre><p class="muted">*보유 중 평가손실 제외. 현재 50종목 고정 Universe는 역사적 생존편향 가능성이 있으며 신호 생성에 사용된 종가 기준 D+1 모델은 아직 D+3/D+5로 재학습되지 않았습니다. 2026 H2 신호가 0인 원인은 전체 후보 단계 로그 없이 단정할 수 없습니다.</p></html>"""
 (out/"report.html").write_text(page,encoding="utf-8")
 print(json.dumps({"status":"ok","portfolio":{m:report["strategy"][m]["portfolio"] for m in "ABC"},"never_selected":report["never_selected"]},ensure_ascii=False))
if __name__=="__main__":main()
