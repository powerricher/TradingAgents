"""PRISM v4.64: TECH200 expansion, fixed Seed A/B entry rules.
New 50 are an ex-ante research proposal, not point-in-time historical constituents.
"""
from .universe_v445 import UNIVERSE as CORE150
CANDIDATE_ADDITIONS={
 "HEALTHCARE":"JNJ UNH ABBV MRK TMO DHR ISRG BSX MDT BDX".split(),
 "FINANCIALS":"JPM BAC WFC GS MS BLK SCHW PNC USB COF".split(),
 "CONSUMER":"WMT COST HD LOW MCD SBUX NKE TGT ROST TJX".split(),
 "INDUSTRIALS":"CAT DE EMR GE HON LMT NOC GD UPS FDX".split(),
 "ENERGY_MATERIALS":"XOM CVX COP SLB EOG OXY NEM FCX NUE VLO".split()
}
PROPOSED=[t for group in CANDIDATE_ADDITIONS.values() for t in group]
NEW50=[t for t in PROPOSED if t not in CORE150]
# Fail loudly rather than silently substituting overlapping names.
assert len(PROPOSED)==50 and len(NEW50)==50,(len(PROPOSED),len(NEW50),sorted(set(PROPOSED)&set(CORE150)))
UNIVERSE=CORE150+NEW50
assert len(UNIVERSE)==200 and len(set(UNIVERSE))==200
