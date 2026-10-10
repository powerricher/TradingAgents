"""PRISM v4.47: SEC filing discovery and point-in-time catalyst evidence audit.
Filing presence is NOT a positive catalyst or a financial impact score.
"""
import argparse,json,time,html,os
from pathlib import Path
from datetime import timedelta
import pandas as pd,requests

FORMS={"8-K","10-Q","10-K","8-K/A","10-Q/A","10-K/A"}
def sec_json(session,url):
 r=session.get(url,timeout=25);r.raise_for_status();return r.json()
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--candidates",default="results/prism-v445/candidate_exits.csv")
 p.add_argument("--ledger",default="results/prism-v445/ledger.csv")
 p.add_argument("--out",default="results/prism-v447")
 p.add_argument("--max-tickers",type=int,default=25)
 p.add_argument("--offline",action="store_true")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 c=pd.read_csv(a.candidates);l=pd.read_csv(a.ledger)
 for d in (c,l):d["entry_date"]=pd.to_datetime(d.entry_date).dt.normalize()
 if "detector" not in c or "experiment" not in l:raise RuntimeError("Missing v4.45 source fields")
 # Target executed trades and same-entry-day competitors. Prefer conflict days.
 selected=l[["ticker","entry_date"]].drop_duplicates()
 active_dates=set(selected.entry_date)
 targets=c[c.entry_date.isin(active_dates)].copy()
 targets["selected"]=pd.MultiIndex.from_frame(targets[["ticker","entry_date"]]).isin(
  pd.MultiIndex.from_frame(selected[["ticker","entry_date"]]))
 collision_days=targets.groupby("entry_date").detector.nunique()
 targets["collision"]=targets.entry_date.map(collision_days)>1
 targets=targets.sort_values(["collision","selected","entry_date"],ascending=[False,False,True])
 tickers=targets.ticker.drop_duplicates().head(a.max_tickers).tolist()
 targets=targets[targets.ticker.isin(tickers)].copy()
 targets["signal_cutoff_utc"]=targets.entry_date.dt.strftime("%Y-%m-%d")+"T00:00:00Z"
 targets["window_start"]= (targets.entry_date-pd.Timedelta(days=60)).dt.strftime("%Y-%m-%d")
 targets.to_csv(out/"research_targets.csv",index=False)
 report={"version":"v4.47","status":"PIT_EVIDENCE_DISCOVERY_ONLY","target_rows":len(targets),
  "target_tickers":len(tickers),"sources":{},"filing_matches":0,"catalyst_scored":0,
  "limitations":[
   "SEC filing metadata is evidence of disclosure, NOT evidence of a positive or negative catalyst.",
   "Only filings with acceptance timestamps strictly before entry date 00:00 UTC are linked; conservative, may exclude known prior-close news.",
   "60-day lookback; source discovery is limited to top selected/conflict tickers, not all TECH150.",
   "SEC recent submissions may omit older filings; historical archive pagination is not implemented.",
   "A completed job without SEC filing matches is not evidence that no historical catalysts existed.",
   "Press releases, news and IR timestamps require separate verified evidence; none are fabricated.",
   "SEC availability failures are reported; missing does not mean no catalyst.",
   "No historical catalyst score or investment return improvement is claimed.",
   "Universe is retrospective; future information leakage and survivorship need separate audits."]}
 evidence=[];errors=[]
 if not a.offline and not os.getenv("SEC_USER_AGENT"):
  errors.append({"source":"SEC_CONFIG","error":"SEC_USER_AGENT secret not configured; no SEC fetch attempted"})
 if not a.offline and os.getenv("SEC_USER_AGENT"):
  s=requests.Session()
  s.headers.update({"User-Agent":os.environ["SEC_USER_AGENT"],"Accept-Encoding":"gzip, deflate","Host":"www.sec.gov"})
  try:
   ticker_map=sec_json(s,"https://www.sec.gov/files/company_tickers.json")
   mapping={v["ticker"].upper():int(v["cik_str"]) for v in ticker_map.values()}
  except Exception as e:
   mapping={};errors.append({"source":"SEC_TICKER_MAP","error":str(e)})
  for ticker in tickers:
   cik=mapping.get(ticker)
   if not cik:
    errors.append({"ticker":ticker,"error":"CIK mapping unavailable"});continue
   try:
    s.headers.pop("Host",None)
    obj=sec_json(s,f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    recent=obj["filings"]["recent"]
    for i,form in enumerate(recent["form"]):
     if form not in FORMS:continue
     acceptance=recent.get("acceptanceDateTime",[])
     if i>=len(acceptance) or not acceptance[i]:continue
     ts=pd.to_datetime(acceptance[i],utc=True,errors="coerce")
     if pd.isna(ts):continue
     subset=targets[targets.ticker==ticker]
     for row in subset.itertuples(index=False):
      cutoff=pd.Timestamp(row.entry_date).tz_localize("UTC")
      if not (cutoff-pd.Timedelta(days=60)<=ts<cutoff):continue
      accession=recent["accessionNumber"][i]
      accession_plain=accession.replace("-","")
      url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_plain}/{recent['primaryDocument'][i]}"
      evidence.append({"ticker":ticker,"entry_date":str(row.entry_date.date()),
       "detector":row.detector,"form":form,"acceptance_utc":ts.isoformat(),
       "accession":accession,"source_url":url,"classification":"UNREVIEWED_DISCLOSURE",
       "catalyst_direction":"UNKNOWN","impact_score":None})
   except Exception as e:errors.append({"ticker":ticker,"error":str(e)})
   time.sleep(.2)
 report["filing_matches"]=len(evidence)
 report["sources"]={"sec_errors":len(errors),"sec_tickers_with_errors":len(set(x.get("ticker","") for x in errors))}
 pd.DataFrame(evidence,columns=["ticker","entry_date","detector","form","acceptance_utc","accession","source_url","classification","catalyst_direction","impact_score"]).to_csv(out/"sec_filing_evidence.csv",index=False)
 (out/"source_errors.json").write_text(json.dumps(errors,indent=2,ensure_ascii=False))
 pd.DataFrame(columns=["ticker","entry_date","event_type","first_public_utc","source_url","verified_by","direction","expected_impact","evidence_status"]).to_csv(out/"catalyst_review_template.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.47</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.47 · Historical Catalyst Evidence</h1><p>SEC metadata only; no catalyst inference or historical alpha claim.</p><pre>"+html.escape(json.dumps(report,indent=2,ensure_ascii=False))+"</pre>",encoding="utf-8")
 print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
