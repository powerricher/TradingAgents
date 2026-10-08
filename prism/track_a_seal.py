"""Track A v2: validate external pre-entry signal snapshot and seal with SHA256.
Do not create a synthetic live prediction.
"""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
REQUIRED={"market_date","ticker","signal_utc","model_version","edge","ridge_rank"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--out",default="results/prism-track-a-v2");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 raw=Path(a.input).read_bytes();df=pd.read_csv(a.input)
 if not REQUIRED.issubset(df.columns):raise ValueError("Missing signal columns: "+str(REQUIRED-set(df.columns)))
 if df.empty:raise ValueError("Refuse to seal empty live snapshot")
 if df.duplicated(["market_date","ticker"]).any():raise ValueError("Duplicate signal keys")
 if df[list(REQUIRED)].isna().any().any():raise ValueError("Missing required values")
 # Check UTC capture timestamps and signal market date; no after-the-fact snapshots.
 stamp=pd.to_datetime(df.signal_utc,utc=True,errors="raise")
 now=pd.Timestamp.now(tz="UTC")
 if (stamp>now).any():raise ValueError("Future signal timestamps")
 if (now-stamp).max()>pd.Timedelta(hours=48):raise ValueError("Stale signal snapshot; refusing forward label")
 fingerprint=hashlib.sha256(raw).hexdigest()
 (out/"snapshot.csv").write_bytes(raw)
 manifest={"status":"SEALED_USER_SUPPLIED_SIGNALS_NOT_MODEL_GENERATED","sha256":fingerprint,"rows":len(df),"sealed_utc":str(now),"input_file":Path(a.input).name,
 "limitations":["Source timestamps are self-reported and not independently attested.","To establish true pre-trade evidence, store signed snapshots in an append-only external record before next open.","No trade execution or outcome calculation is performed."]}
 (out/"seal.json").write_text(json.dumps(manifest,indent=2))
 print(json.dumps(manifest))
if __name__=="__main__":main()
