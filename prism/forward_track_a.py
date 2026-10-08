"""Track A: immutable forward snapshot manifest and paper-trade journal template.
No claim of live performance until post-snapshot outcomes exist.
"""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from .universe_v413 import UNIVERSE
def main():
 p=argparse.ArgumentParser();p.add_argument("--out",default="results/prism-track-a");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 now=datetime.now(timezone.utc).isoformat()
 # This is a launch manifest, not a fabricated live prediction.
 config={"track":"A","model":"v4.17 ridge","created_utc":now,"universe":UNIVERSE,
 "entry":"next US trading session open after frozen close signal",
 "exit":"MA30 / ATR as in v4.17","round_trip_bps":10,
 "signal_status":"NOT GENERATED; requires a live point-in-time scoring adapter",
 "rules":["Never backfill or infer a prediction snapshot","Record source timestamp and SHA256 before next-session entry","Never overwrite frozen forecasts","Only score settled exits","Do not count historical replay as forward performance"]}
 payload=json.dumps(config,sort_keys=True).encode()
 config["manifest_sha256"]=hashlib.sha256(payload).hexdigest()
 (out/"manifest.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
 cols=["signal_utc","market_date","ticker","model_version","edge","ridge_rank","p_down","entry_open","exit_date","exit_price","net_return","status","source_sha256"]
 pd.DataFrame(columns=cols).to_csv(out/"forward_journal_template.csv",index=False)
 (out/"README.md").write_text("# Track A — forward validation\n\nThis is a versioned launch manifest and empty journal, **not a live signal generator**.\nDo not treat it as forward test performance. A point-in-time live scoring adapter and persistent append-only storage must be built before live operation.\n")
 print(json.dumps({"status":"manifest_only","universe":len(UNIVERSE),"created_utc":now}))
if __name__=="__main__":main()
