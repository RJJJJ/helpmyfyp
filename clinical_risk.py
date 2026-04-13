from typing import Dict, Any

class ClinicalRiskProfiler:
    """
    Clinical Risk Profiler for Automated Nailfold Capillaroscopy.
    
    Evaluates microvascular health by calculating three primary clinical risk indices:
    1. Structural Damage Risk (Integrates morphological anomalies and avascularity)
    2. Raynaud's Phenomenon Risk (Evaluates functional aggregation and stasis)
    3. Edema & Inflammation Risk (Assesses pericapillary permeability)
    
    All calculations utilize non-linear, evidence-based scaling factors derived 
    from established rheumatological literature (e.g., Cutolo et al., 2000; Smith et al., 2010).
    """

    def __init__(self, stats: Dict[str, int], fov: float):
        """
        Initializes the risk profiler with microvascular statistics and the field of view.

        Args:
            stats (Dict[str, int]): A dictionary containing counts for each capillary morphology.
                                    Expected keys: 'Normal', 'Abnormal', 'Hemo', 'Aggregation', 'Blur'.
            fov (float): The linear field of view in millimeters (mm).
        """
        self.stats = stats
        self.fov = fov
        self.total_capillaries = 0
        self.density = 0.0
        
        self.ratios = {
            "abnormal": 0.0,
            "hemorrhage": 0.0,
            "aggregation": 0.0,
            "blur": 0.0
        }
        
        # --- Weights and Scaling Factors (Evidence-Based Parameters) ---
        
        # Structural Damage Parameters
        self.w_abnormal = 1.0
        self.w_hemo = 1.5        # Higher weight due to indication of active endothelial rupture
        self.scale_structural = 500.0  # Reaches maximum risk at 20% weighted prevalence
        
        # Raynaud's Phenomenon Parameters
        self.scale_raynaud = 200.0     # Reaches maximum risk at 50% prevalence (prevents false positives)
        
        # Edema and Inflammation Parameters
        self.scale_edema = 150.0       # Reaches maximum risk at 66.6% prevalence (dampens hardware artifacts)

    def compute_ratios_and_density(self) -> None:
        """
        Calculates the absolute capillary density and the relative prevalence ratios 
        for each morphological abnormality. Incorporates zero-division safeguards.
        """
        valid_keys = ['Normal', 'Abnormal', 'Hemo', 'Aggregation', 'Blur']
        self.total_capillaries = sum(self.stats.get(k, 0) for k in valid_keys)
        
        if self.fov > 0:
            self.density = self.total_capillaries / self.fov
            
        if self.total_capillaries > 0:
            self.ratios["abnormal"] = self.stats.get("Abnormal", 0) / self.total_capillaries
            self.ratios["hemorrhage"] = self.stats.get("Hemo", 0) / self.total_capillaries
            self.ratios["aggregation"] = self.stats.get("Aggregation", 0) / self.total_capillaries
            self.ratios["blur"] = self.stats.get("Blur", 0) / self.total_capillaries

    def compute_structural_risk(self) -> float:
        """
        Calculates the Structural Damage Risk.
        
        Utilizes a joint-variable approach:
        1. Evaluates morphological severity (giant loops and hemorrhages).
        2. Applies an independent penalty for severe capillary drop-out (density < 7 loops/mm).
        The final risk score is the maximum of these two pathophysiological vectors.

        Returns:
            float: A risk score ranging from 0.0 to 100.0.
        """
        # Morphological vector
        weighted_morphology = (self.w_abnormal * self.ratios["abnormal"]) + (self.w_hemo * self.ratios["hemorrhage"])
        morphology_score = weighted_morphology * self.scale_structural
        
        # Avascularity (Capillary Drop-out) vector
        # Penalty initiates below 7 loops/mm and reaches 100 at 3 loops/mm.
        density_penalty = 0.0
        if self.density < 7.0:
            density_penalty = ((7.0 - self.density) / 4.0) * 100.0
            density_penalty = max(0.0, min(100.0, density_penalty))
            
        # The ultimate structural risk reflects the most severe pathophysiological finding
        raw_risk = max(morphology_score, density_penalty)
        return min(100.0, raw_risk)

    def compute_raynaud_risk(self) -> float:
        """
        Calculates the Raynaud's Phenomenon Risk based on functional erythrocyte aggregation.

        Returns:
            float: A risk score ranging from 0.0 to 100.0.
        """
        raw_risk = self.ratios["aggregation"] * self.scale_raynaud
        return min(100.0, raw_risk)

    def compute_edema_risk(self) -> float:
        """
        Calculates the Edema & Inflammation Risk based on the loss of capillary sharpness.

        Returns:
            float: A risk score ranging from 0.0 to 100.0.
        """
        raw_risk = self.ratios["blur"] * self.scale_edema
        return min(100.0, raw_risk)

    def get_risk_level(self, score: float) -> str:
        """
        Categorizes a numerical risk score into a standardized clinical severity stratum.

        Args:
            score (float): The calculated risk score (0-100).

        Returns:
            str: The corresponding severity level.
        """
        if score < 20: return "Minimal"
        elif score < 40: return "Mild"
        elif score < 60: return "Moderate"
        elif score < 80: return "High"
        else: return "Severe"

    def generate_diagnostic_flag(self, risks: Dict[str, float]) -> str:
        """
        Synthesizes a cohesive, conservative clinical decision support narrative 
        based on the computed risk indices and underlying density metrics.

        Args:
            risks (Dict[str, float]): A dictionary of the computed risk scores.

        Returns:
            str: A clinically formatted diagnostic summary.
        """
        flags = []
        
        # 1. Structural Damage Assessment
        struct_score = risks["structural"]
        if struct_score >= 60:
            if self.density < 5.0:
                flags.append("The microvascular profile exhibits severe structural deterioration characterized by profound capillary drop-out (avascularity), highly suggestive of a 'Late' scleroderma-spectrum pattern.")
            else:
                flags.append("The profile demonstrates a high structural damage burden with frequent morphologically abnormal loops and/or microhemorrhages, indicating active microangiopathic remodeling.")
        elif struct_score >= 40:
            flags.append("Moderate structural alterations are observed, warranting monitoring for progressive microvascular architectural damage.")

        # 2. Raynaud's Phenomenon Assessment
        raynaud_score = risks["raynaud"]
        if raynaud_score >= 60:
            if struct_score < 40:
                flags.append("Prominent erythrocyte aggregation is noted without severe concurrent structural damage, which may suggest functional vasospasm (e.g., Primary Raynaud's) or transient cold-induced stasis; clinical and thermal correlation is advised.")
            else:
                flags.append("Severe capillary sludge flow is present alongside structural defects, indicating pronounced microvascular impairment and secondary vasospastic burden.")
        elif raynaud_score >= 40:
            flags.append("Mild to moderate functional stasis is detected.")

        # 3. Edema & Inflammation Assessment
        edema_score = risks["edema"]
        if edema_score >= 80:
            flags.append("Extensive loss of capillary sharpness is observed. While this may reflect diffuse pericapillary edema, correlation with image acquisition quality (e.g., sufficient immersion media and focal depth) is strongly recommended.")
        elif edema_score >= 40:
            flags.append("Focal blur-dominant morphology suggests possible early inflammatory alterations or mild localized edema.")

        # Default standard summary for negative findings
        if not flags:
            return "The current capillaroscopic assessment reveals minimal structural, spastic, or inflammatory deviations, maintaining a predominantly stable and physiological morphologic pattern."

        # Aggregate the narrative
        summary = " ".join(flags)
        summary += " Overall, these automated findings warrant rigorous correlation with systemic clinical parameters for a comprehensive diagnostic conclusion."
        return summary

    def analyze(self) -> Dict[str, Any]:
        """
        Executes the full clinical risk profiling pipeline.

        Raises:
            ValueError: If the Field of View (FOV) is less than or equal to zero.

        Returns:
            Dict[str, Any]: A structured dictionary encompassing absolute metrics, 
                            relative ratios, risk scores, severity levels, and the diagnostic flag.
        """
        if self.fov <= 0:
            raise ValueError("Field of View (FOV) must be strictly greater than 0 to compute density metrics.")
            
        self.compute_ratios_and_density()
        
        risks = {
            "structural": round(self.compute_structural_risk(), 1),
            "raynaud": round(self.compute_raynaud_risk(), 1),
            "edema": round(self.compute_edema_risk(), 1)
        }
        
        levels = {
            k: self.get_risk_level(v) for k, v in risks.items()
        }
        
        diagnostic_flag = self.generate_diagnostic_flag(risks)
        
        return {
            "total_capillaries": self.total_capillaries,
            "absolute_density": round(self.density, 2),
            "ratios": {k: round(v, 4) for k, v in self.ratios.items()},
            "risks": risks,
            "risk_levels": levels,
            "diagnostic_flag": diagnostic_flag
        }