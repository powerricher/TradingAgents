"""PRISM v4.45: retrospective 150-stock research universe.
Do not interpret membership as historical point-in-time investability.
"""
from .universe_v429 import UNIVERSE as CORE100
ADDITIONS={
 "SEMICONDUCTOR":"ADI NXPI WOLF AMBA NVTS POWI DIOD AOSL SYNA SLAB".split(),
 "SOFTWARE":"CRM INTU WDAY ADSK ANSS CDNS SNPS PTC MANH PAYC".split(),
 "SECURITY_NETWORK":"AKAM CHKP GEN CYBR TENB RPD VRNS FROG ESTC CFLT".split(),
 "INDUSTRIAL_POWER":"HUBB EME FIX TT CARR JCI IR ROK PH DOV".split(),
 "DATA_INFRA":"IBM PSTG SMCI CLS JBL SANM FLEX INFN COMM".split(),
}
NEW50=[t for group in ADDITIONS.values() for t in group]
UNIVERSE=CORE100+NEW50
assert len(CORE100)==100 and len(NEW50)==50 and len(UNIVERSE)==150 and len(set(UNIVERSE))==150,(len(UNIVERSE),len(set(UNIVERSE)))
