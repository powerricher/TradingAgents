"""PRISM v4.29 technology 100 research universe, retrospectively defined."""
from .universe_v413 import UNIVERSE as CORE80
ADDITIONS={
 "SEMICONDUCTOR_PACKAGING":"AMKR ASX UCTT ICHR VECO OLED GFS MTSI RMBS SITM PI".split(),
 "OPTICAL_NETWORK":"VIAV CALX HLIT NOK ERIC".split(),
 "POWER_ENERGY":"FSLR ENPH SEDG FLNC".split(),
}
NEW20=[t for group in ADDITIONS.values() for t in group]
UNIVERSE=CORE80+NEW20
assert len(CORE80)==80 and len(NEW20)==20 and len(UNIVERSE)==100 and len(set(UNIVERSE))==100
