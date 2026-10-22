# Fan and Bypass Manual (fictional training document)

## FAN-01: Bypass ratio increasing
Signature: bypass ratio (s15) creeping above its healthy typical level.
Likely causes: core components losing efficiency so the airflow balance shifts, or eroded fan blades.
Actions: inspect fan blades for erosion or damage; check bypass-duct seals; if compressor or turbine sensors also drift, follow HPC-01 and TUR-01.
Escalate if: bypass ratio keeps rising together with core temperatures.

## FAN-02: Fan and core speed deviation
Signature: fan speed (s8), core speed (s9), corrected fan speed (s13) or corrected core speed (s14) drifting away from typical for the same load.
Likely causes: bearing wear, imbalance, control sensor drift.
Actions: check vibration and bearing condition; calibrate speed sensors; inspect shaft couplings.
Escalate if: speed deviation grows or vibration rises.

## FAN-03: Bypass-duct pressure abnormal
Signature: bypass-duct pressure (s6) away from typical.
Likely causes: duct leakage, damaged acoustic liner, loose panels.
Actions: inspect duct panels and seals for gaps; pressure-test the duct.
Escalate if: pressure keeps changing across consecutive checks.