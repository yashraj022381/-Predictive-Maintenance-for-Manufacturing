# Compressor Section Maintenance Manual (fictional training document)

## HPC-01: HPC outlet temperature rising
Signature: HPC outlet temperature (s3) sits clearly above its healthy typical level and keeps climbing, often together with HPC outlet static pressure (s11) rising.
Likely causes: fouled or eroded compressor blades, worn blade-tip clearances, leaking seals. All of these reduce compressor efficiency, so the air leaves hotter.
Actions: borescope-inspect HPC blades and tip clearances; check the date of the last compressor wash and wash if overdue; compare the efficiency trend over the last 50 cycles.
Escalate if: the rise continues across consecutive checks or fuel flow ratio (s12) starts dropping.

## HPC-02: HPC pressure and fuel flow shifts
Signature: HPC outlet static pressure (s11) drifting up while HPC outlet pressure (s7) or fuel flow ratio (s12) drifts down.
Likely causes: internal air leakage, stator vane wear, compressor stall margin loss.
Actions: leak-check the compressor casing and seals; inspect variable stator vanes for wear and correct angle; review fuel flow trend.
Escalate if: pressure and fuel flow keep diverging over the next 20 cycles.

## LPC-01: LPC outlet temperature rising
Signature: LPC outlet temperature (s2) slowly higher than its healthy typical level.
Likely causes: early-stage fouling of low-pressure compressor blades, leaking bleed valve.
Actions: inspect LPC blades for deposits; test the bleed valve for leakage; clean if fouled.
Escalate if: HPC outlet temperature (s3) is also rising; see HPC-01.