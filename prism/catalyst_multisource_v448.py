"""PRISM v4.48: independent multi-source historical catalyst evidence collector.
SEC submissions, GDELT news discovery, curated IR evidence. No inferred alpha.
"""
import argparse,json,os,time,html
from pathlib import Path
import pandas as pd,requests

FORMS={"8-K","10-Q","10-K","8-K/A","10-Q/A","10-K/A"}
COLS=["ticker","signal_date","source","source_url","headline","event_type","first_public_utc","verified_pit","evidence_status","accession"]
def fetch(session,url,params=None):
 r=session.get(url,params=params,timeout=30)
 r.raise_for_status()
 return r.json()
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--targets",default="results/prism-v447/research_targets.csv")
 p.add_argument("--out",default="results/prism-v448")
 p.add_argument("--ir-input",default="data/prism/ir_evidence_pit.csv")
 p.add_argument("--cik-map",default="data/prism/ticker_cik.csv")
 p.add_argument("--max-tickers",type=int,default=25)
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 t=pd.read_csv(a.targets)
 if not {"ticker","entry_date"}.issubset(t):raise RuntimeError("Missing targets")
 t.entry_date=pd.to_datetime(t.entry_date).dt.normalize()
 if "signal_date" in t:
  t.signal_date=pd.to_datetime(t.signal_date).dt.normalize()
 else:
  # Conservative previous calendar day; does NOT assert true signal timestamp.
  t["signal_date"]=t.entry_date-pd.Timedelta(days=1)
 t=t[t.ticker.isin(t.ticker.drop_duplicates().head(a.max_tickers))].copy()
 if t.empty:raise RuntimeError("Empty target set")
 # 00:00 UTC of entry date is conservative; any same-day disclosure is excluded.
 t["cutoff_utc"]=pd.to_datetime(t.entry_date,utc=True)
 t["start_utc"]=t.cutoff_utc-pd.Timedelta(days=60)
 evidence=[];errors=[];source_stats={}
 session=requests.Session()
 session.headers.update({"User-Agent":os.getenv("SEC_USER_AGENT","PRISM research contact research@example.com")})
 # SEC mapping can be supplied from a previously validated file.
 mapping={}
 if Path(a.cik_map).exists():
  m=pd.read_csv(a.cik_map)
  if not {"ticker","cik"}.issubset(m):raise RuntimeError("Invalid CIK map columns")
  mapping={str(r.ticker).upper():int(r.cik) for r in m.itertuples(index=False)}
 else:
  try:
   j=fetch(session,"https://www.sec.gov/files/company_tickers.json")
   mapping={v["ticker"].upper():int(v["cik_str"]) for v in j.values()}
  except Exception as e:errors.append({"source":"SEC_CIK_MAPPING","error":str(e)})
 source_stats["sec_mapping_count"]=len(mapping)
 for ticker,g in t.groupby("ticker",sort=True):
  cik=mapping.get(ticker)
  if cik is None:
   errors.append({"source":"SEC","ticker":ticker,"error":"CIK_NOT_AVAILABLE"});continue
  try:
   j=fetch(session,f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
   filings=j.get("filings",{})
   blocks=[filings.get("recent",{})]
   # SEC archive pagination: older filing blocks from submissions.files.
   for meta in filings.get("files",[]):
    first=pd.Timestamp(meta["filingFrom"])
    last=pd.Timestamp(meta["filingTo"])
    if last<g.start_utc.min().tz_localize(None) or first>=g.cutoff_utc.max().tz_localize(None):continue
    try:blocks.append(fetch(session,"https://data.sec.gov/submissions/"+meta["name"]))
    except Exception as e:errors.append({"source":"SEC_ARCHIVE","ticker":ticker,"file":meta["name"],"error":str(e)})
    time.sleep(.15)
   seen=set()
   for block in blocks:
    for i,form in enumerate(block.get("form",[])):
     if form not in FORMS:continue
     accession=block["accessionNumber"][i]
     if accession in seen:continue
     seen.add(accession)
     stamps=block.get("acceptanceDateTime",[])
     # Filing date alone is NOT a verified intraday timestamp.
     stamp=stamps[i] if i<len(stamps) else None
     if not stamp:continue
     ts=pd.to_datetime(stamp,utc=True,errors="coerce")
     if pd.isna(ts):continue
     doc=block["primaryDocument"][i]
     url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-','')}/{doc}"
     for row in g.itertuples(index=False):
      if row.start_utc<=ts<row.cutoff_utc:
       evidence.append({"ticker":ticker,"signal_date":str(row.signal_date.date()),
        "source":"SEC","source_url":url,"headline":form,"event_type":"UNREVIEWED_FILING",
        "first_public_utc":ts.isoformat(),"verified_pit":True,
        "evidence_status":"METADATA_VERIFIED_CONTENT_UNREVIEWED","accession":accession})
  except Exception as e:errors.append({"source":"SEC_SUBMISSIONS","ticker":ticker,"error":str(e)})
  time.sleep(.2)
 # GDELT DOC search is discovery only: published date is not necessarily
 # the first public release of the underlying corporate information.
 for ticker,g in t.groupby("ticker",sort=True):
  # Bound requests by target day, rather than querying an unrestricted archive.
  for day in sorted(g.entry_date.unique())[:3]:
   dt=pd.Timestamp(day)
   try:
    j=fetch(session,"https://api.gdeltproject.org/api/v2/doc/doc",{
     "query":'"'+ticker+'" (earnings OR guidance OR acquisition OR contract)',
     "mode":"artlist","format":"json","maxrecords":25,"startdatetime":(dt-pd.Timedelta(days=60)).strftime("%Y%m%d%H%M%S"),
     "enddatetime":dt.strftime("%Y%m%d%H%M%S")})
    for article in j.get("articles",[]):
     url=article.get("url")
     stamp=article.get("seendate")
     if not url or not stamp:continue
     try:ts=pd.to_datetime(stamp,utc=True)
     except Exception:continue
     cutoff=dt.tz_localize("UTC")
     if not (cutoff-pd.Timedelta(days=60)<=ts<cutoff):continue
     evidence.append({"ticker":ticker,"signal_date":str((dt-pd.Timedelta(days=1)).date()),
      "source":"GDELT","source_url":url,"headline":article.get("title",""),
      "event_type":"UNREVIEWED_NEWS","first_public_utc":ts.isoformat(),
      "verified_pit":False,"evidence_status":"NEWS_INDEX_TIME_NOT_FIRST_PUBLIC","accession":""})
   except Exception as e:errors.append({"source":"GDELT","ticker":ticker,"date":str(dt.date()),"error":str(e)})
   time.sleep(.25)
 # IR data must be supplied with explicit verified first-public timestamps.
 ir=Path(a.ir_input)
 if ir.exists():
  f=pd.read_csv(ir)
  required={"ticker","signal_date","first_public_utc","source_url","headline"}
  if not required.issubset(f):raise RuntimeError("IR input missing "+str(required-set(f)))
  for row in f.itertuples(index=False):
   ts=pd.to_datetime(row.first_public_utc,utc=True,errors="coerce")
   day=pd.Timestamp(row.signal_date).tz_localize("UTC")
   if pd.isna(ts) or not day-pd.Timedelta(days=60)<=ts<day+pd.Timedelta(days=1):
    errors.append({"source":"IR","ticker":row.ticker,"error":"INVALID_PIT_WINDOW"});continue
   evidence.append({"ticker":row.ticker,"signal_date":str(row.signal_date),"source":"CURATED_IR",
    "source_url":row.source_url,"headline":row.headline,"event_type":"UNREVIEWED_IR",
    "first_public_utc":ts.isoformat(),"verified_pit":False,
    "evidence_status":"CURATED_REQUIRES_SOURCE_VERIFICATION","accession":""})
 frame=pd.DataFrame(evidence,columns=COLS).drop_duplicates(["ticker","signal_date","source_url"])
 frame.to_csv(out/"catalyst_evidence.csv",index=False)
 (out/"source_errors.json").write_text(json.dumps(errors,ensure_ascii=False,indent=2))
 pd.DataFrame(columns=["ticker","signal_date","first_public_utc","source_url","headline"]).to_csv(out/"ir_input_template.csv",index=False)
 report={"version":"v4.48","status":"EVIDENCE_DISCOVERY_NOT_CATALYST_SCORING",
  "targets":len(t),"tickers":int(t.ticker.nunique()),"evidence":len(frame),
  "source_counts":frame.source.value_counts().to_dict() if len(frame) else {},
  "verified_metadata":int(frame.verified_pit.sum()) if len(frame) else 0,
  "source_errors":len(errors),"catalyst_scored":0,
  "limitations":["SEC filing acceptance time is verified metadata, not positive catalyst classification.",
   "GDELT seendate is not first-public time; GDELT evidence is never automatically PIT-verified.",
   "Historical IR is curated input only, not automatically crawled.",
   "v4.47 targets may lack true signal timestamps; conservative entry-day midnight cutoff applied.",
   "Search is limited to at most 3 entry dates per ticker and 25 GDELT records per request.",
   "Missing evidence is SOURCE_UNKNOWN, not NO_CATALYST.",
   "No alpha, returns, catalyst polarity or priced-in estimates are computed."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.48</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.48 · Multi-Source Catalyst Evidence</h1><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
