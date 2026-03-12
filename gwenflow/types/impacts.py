from pydantic import BaseModel


class BaseImpact(BaseModel):
    type: str
    name: str
    value: str
    unit: str


class Energy(BaseImpact):
    type: str = "energy"
    name: str = "Energy"
    unit: str = "kWh"


class GWP(BaseImpact):
    type: str = "GWP"
    name: str = "Global Warming Potential"
    unit: str = "kgCO2eq"


class ADPe(BaseImpact):
    type: str = "ADPe"
    name: str = "Abiotic Depletion Potential (elements)"
    unit: str = "kgSbeq"


class PE(BaseImpact):
    type: str = "PE"
    name: str = "Primary Energy"
    unit: str = "MJ"


class WCF(BaseImpact):
    type: str = "WCF"
    name: str = "Water Consumption Footprint"
    unit: str = "L"


class Usage(BaseModel):
    energy: Energy
    gwp: GWP
    adpe: ADPe
    pe: PE
    wcf: WCF


class Embodied(BaseModel):
    gwp: GWP
    adpe: ADPe
    pe: PE


class Impacts(BaseModel):
    energy: Energy
    gwp: GWP
    adpe: ADPe
    pe: PE
    wcf: WCF
    usage: Usage
    embodied: Embodied
