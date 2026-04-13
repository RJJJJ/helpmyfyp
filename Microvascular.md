# Mathematical Justification and Clinical Interpretation of the Microvascular Health Radar

## 1. Overview and Clinical Purpose
The Microvascular Health Profile (Radar Chart) visually deconstructs the unified Microvascular Health Index (MHI) into its constituent clinical vectors. While the MHI provides a single continuous score from 0 to 100, the radar chart projects this score across five distinct dimensions. 

This multi-axis visualization allows clinicians to instantly identify the specific pathological drivers behind a low MHI—whether the primary issue is capillary loss, structural deformity, hemodynamic instability, or tissue edema. The chart plots the **Patient Profile** against a **Healthy Baseline**, which represents an optimal physiological state derived from normative data.

---

## 2. Derivation of the Clinical Axes (Subscores)
To plot variables of different scales on a unified radar chart, all subscores are normalized to a standard **0 to 100 scale**, where $100$ represents optimal health and $0$ represents the maximum clinically defined penalty for that specific vector.

### A. Density Adequacy
This axis reflects the capillary network's physical density. It is based on the EULAR consensus, which defines severe avascularity at $\le$ 3 loops/mm. The axis normalizes the computed density penalty ($P_{density}$) against its maximum possible deduction (40 points).
$$Density\_Adequacy = 100 - \left( \frac{P_{density}}{40} \right) \times 100$$

### B. Structural Integrity
This axis measures the absence of severe structural pathologies defined by Cutolo's "Active" criteria, specifically giant capillaries and microhemorrhages. It normalizes the active pathology penalty against its maximum clinical weight ($W_{active} = 35$).
$$Structural\_Integrity = 100 - \left( \frac{Ratio_{active} \times 35}{35} \right) \times 100$$
*(Note: $Ratio_{active} = Ratio_{abnormal} + Ratio_{hemo}$)*

### C. Flow Stability
This axis evaluates the hemodynamics of the capillary bed. It penalizes for sludged blood flow or aggregation, which is an indicator of ischemia. It is normalized against the flow penalty weight ($W_{flow} = 20$).
$$Flow\_Stability = 100 - \left( \frac{Ratio_{aggregation} \times 20}{20} \right) \times 100$$

### D. Edema/Clarity
This axis indicates the presence of pericapillary edema (Blur). While it carries a lower clinical risk weight ($W_{edema} = 10$) compared to active structural damage, it is a persistent indicator of inflammation or leakage.
$$Edema\_Clarity = 100 - \left( \frac{Ratio_{blur} \times 10}{10} \right) \times 100$$

### E. Overall MHI
The composite Microvascular Health Index, aggregating all density and morphological penalties into the final clinical score.
$$Overall\_MHI = 100 - P_{density} - P_{morphology}$$

---

## 3. Worked Example: Mapping a Case to the Radar
Tracing the mathematics of the sample case:
* **FOV**: 3.0 mm
* **Total Capillaries**: 15 loops
* **Morphology**: 11 Normal (73.3%), 4 Blur (26.7%)

**Step 1: Calculate $P_{density}$ and Density Adequacy**
* $Density = 15 / 3.0 = 5.0$ loops/mm.
* $P_{density} = 20$.
* $$Density\_Adequacy = 100 - \left( \frac{20}{40} \right) \times 100 = 50.0$$

**Step 2: Calculate Morphology Ratios and Subscores**
* There are 0 Abnormal, 0 Hemo, and 0 Aggregation capillaries.
* $Structural\_Integrity = 100.0$
* $Flow\_Stability = 100.0$
* $Ratio_{blur} = 4 / 15 = 0.2667$.
* $$Edema\_Clarity = 100 - \left( \frac{0.2667 \times 10}{10} \right) \times 100 = 73.3$$

**Step 3: Plotting the Final Profile**
The final array plotted on the radar chart for this patient is:
* **Density Adequacy**: 50.0
* **Structural Integrity**: 100.0
* **Flow Stability**: 100.0
* **Edema/Clarity**: 73.3
* **Overall MHI**: 77.33

**Clinical Interpretation:** When plotted, the resulting polygon will perfectly match the outer bounds of the healthy baseline on the *Structural Integrity* and *Flow Stability* axes. However, it will deeply retract toward the center on the *Density Adequacy* and *Edema/Clarity* axes. This visually communicates to the physician that the patient's primary clinical issues are capillary drop-out and fluid leakage, rather than severe structural deformities.