"""PRISM v4.13 80-stock technology research universe.
Research universe is retrospective; not point-in-time historical membership.
"""
from .universe_v45 import UNIVERSE as CORE50
ADDITIONS={
"SEMICONDUCTOR":"ON MCHP MPWR SWKS QRVO LSCC ACLS FORM AEHR CAMT".split(),
"AI_INFRASTRUCTURE":"TER NTAP WDC STX GLW AAOI FN KEYS".split(),
"SOFTWARE_CYBER":"MDB OKTA HUBS TEAM DOCU TWLO PATH".split(),
"POWER_AUTOMATION":"GEV PWR NVT MOD ITRI".split(),
}
NEW30=[x for g in ADDITIONS.values() for x in g]
UNIVERSE=CORE50+NEW30
assert len(CORE50)==50 and len(NEW30)==30 and len(UNIVERSE)==80 and len(set(UNIVERSE))==80
