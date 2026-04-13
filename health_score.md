# Mathematical Justification of the Microvascular Health Index (MHI)

## 1. Overview and Core Formula
The Microvascular Health Index (MHI) is a quantitative scoring system designed to evaluate nailfold capillaroscopy results on a continuous scale from **0 (Severe Microangiopathy)** to **100 (Optimal Health)**.

Instead of relying on arbitrary "black-box" AI scores, the MHI is strictly derived from internationally recognized rheumatology guidelines. The core equation isolates two primary clinical vectors—capillary density and morphological deformity:

$$MHI = 100 - P_{density} - P_{morphology}$$

**Parameters:**
* **$100$**: The baseline score representing a perfect, healthy capillary bed.
* **$P_{density}$**: The penalty deducted for microvascular loss (avascularity).
* **$P_{morphology}$**: The penalty deducted for the presence of pathological vessel structures.

---

## 2. The Density Penalty ($P_{density}$)

### Clinical Basis
The system adopts the consensus criteria established by the **European League Against Rheumatism (EULAR)** Study Group on Microcirculation. 
* **Healthy State**: A "non-scleroderma pattern" is defined by a capillary density of $\ge$ 7 loops/mm.
* **Severe State**: Severe avascularity is clinically defined when density drops to $\le$ 3 loops/mm.

### Mathematical Implementation
The system calculates linear density based on the Field of View (FOV):
$$Density = \frac{Total\_Capillaries}{FOV}$$

The penalty function is a piecewise linear interpolation bounded by the EULAR thresholds, with a maximum penalty ($Max\_P_{density}$) of 40 points:
* **Optimal ($Density \ge 7$):** $P_{density} = 0$
* **Severe ($Density \le 3$):** $P_{density} = 40$
* **Borderline ($3 < Density < 7$):**
    $$P_{density} = 40 \times \left( \frac{7 - Density}{7 - 3} \right)$$

---

## 3. The Morphology Penalty ($P_{morphology}$)

### Clinical Basis
Based on **Cutolo’s Classification**, different structural abnormalities carry different clinical risks. For instance, giant capillaries and microhemorrhages are hallmarks of an "Active" systemic sclerosis pattern.

### Mathematical Implementation
The algorithm calculates the ratio of each pathological class relative to the total number of detected vessels:
$$Ratio_{class} = \frac{Count_{class}}{Total\_Capillaries}$$

Each pathology is assigned a distinct clinical weight ($W$) based on its severity:
* **$W_{active} = 35$**: Highest risk. Applies to **Abnormal** (Giant/Tortuous/Bushy) and **Hemo** (Microhemorrhages).
* **$W_{flow} = 20$**: Moderate risk. Applies to **Aggregation** (sludged blood flow indicating ischemia).
* **$W_{edema} = 10$**: Lower risk. Applies to **Blur** (pericapillary edema).

The total morphology penalty is the sum of these weighted ratios:
$$P_{morphology} = (W_{active} \times (Ratio_{abnormal} + Ratio_{hemo})) + (W_{flow} \times Ratio_{aggregation}) + (W_{edema} \times Ratio_{blur})$$

---

## 4. Worked Example: Deconstructing a Score of 77.33
Tracing the exact mathematics of a sample case:

**Input Data:**
* **FOV**: 3.0 mm
* **Total Capillaries**: 15 loops
* **Morphology**: 11 Normal (73.3%), 4 Blur (26.7%)

**Step 1: Calculate Density Penalty**
* $Density = 15 / 3.0 = 5.0$ loops/mm.
* $$P_{density} = 40 \times \left( \frac{7 - 5.0}{4} \right) = 20$$.

**Step 2: Calculate Morphology Penalty**
* $Ratio_{blur} = 4 / 15 = 0.2667$.
* $W_{edema} = 10$.
* $P_{morphology} = 10 \times 0.2667 = 2.67$.

**Step 3: Final MHI Calculation**
$$MHI = 100 - 20 - 2.67 = 77.33$$.

---

## 5. Conclusion
The resulting MHI is a direct, fully explainable mathematical translation of **EULAR** and **Cutolo** clinical guidelines. By mapping multi-class morphological outputs into a continuous scale, the system provides a scientifically rigorous method for clinical evaluation.