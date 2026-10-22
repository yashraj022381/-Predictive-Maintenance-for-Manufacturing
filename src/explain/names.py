"""Turn cryptic feature names into plain English."""

SENSOR_NAMES = {
    "s2": "LPC outlet temperature",
    "s3": "HPC outlet temperature",
    "s4": "LPT outlet temperature",
    "s6": "Bypass-duct pressure",
    "s7": "HPC outlet pressure",
    "s8": "Fan speed",
    "s9": "Core speed",
    "s11": "HPC outlet static pressure",
    "s12": "Fuel flow ratio",
    "s13": "Corrected fan speed",
    "s14": "Corrected core speed",
    "s15": "Bypass ratio",
    "s17": "Bleed enthalpy",
    "s20": "HPT coolant bleed",
    "s21": "LPT coolant bleed",
}


def pretty(feature: str) -> str:
    base, _, kind = feature.partition("_")
    name = SENSOR_NAMES.get(base, base)
    if kind.startswith("mean"):
        return f"{name} (recent average)"
    if kind.startswith("std"):
        return f"{name} (recent jumpiness)"
    if kind.startswith("diff"):
        return f"{name} (recent change)"
    return f"{name} (current reading)"
