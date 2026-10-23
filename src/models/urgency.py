
"""Urgency rules (policies POL-01 and POL-03) as plain code. No heavy imports."""

LEVELS = {1: "WATCH", 2: "PLAN", 3: "URGENT", 4: "STOP"}
WARMUP_CYCLES = 10   # rolling features need history; ignore anomaly flags before this


def decide_urgency(risk: float, alarm: bool, rul: float, n_drift: int):
    reasons = []
    if rul < 10:
        level = 4
        reasons.append(f"estimated cycles left {rul:.0f} is under 10 [POL-01]")
    elif risk > 0.95 and n_drift >= 3:
        level = 4
        reasons.append(f"risk {risk:.1%} is above 95% with {n_drift} sensors worsening [POL-01]")
    elif rul <= 30:
        level = 3
        reasons.append(f"estimated cycles left {rul:.0f} is between 10 and 30 [POL-01]")
    elif alarm or rul <= 60:
        level = 2
        reasons.append("risk at/above the alarm line or estimated cycles left between 30 and 60 [POL-01]")
    else:
        level = 1
        reasons.append("risk below alarm line and more than 60 cycles left [POL-01]")

    if n_drift >= 3 and 2 <= level < 4:        # POL-03: raise one level
        level += 1
        reasons.append(f"{n_drift} sensors drifting together, urgency raised one level [POL-03]")
    return level, reasons
