import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

class ClinicalHealthAnalyzer:
    """
    Analyzes microvascular health based on semantic segmentation statistics.
    Computes the Microvascular Health Index (MHI) incorporating evidence-based 
    non-linear threshold scaling for structural, functional, and inflammatory parameters.
    """
    
    # Evidence-based scaling multipliers to map physiological thresholds to maximum penalties
    SCALE_ACTIVE: float = 5.0   # 20% prevalence reaches maximum severity
    SCALE_FLOW: float = 2.0     # 50% prevalence reaches maximum severity
    SCALE_EDEMA: float = 1.5    # 66.6% prevalence reaches maximum severity
    
    WEIGHT_HEMO: float = 1.5    # Multiplier for acute endothelial rupture
    WEIGHT_ABNORMAL: float = 1.0

    def __init__(self, 
                 w_density: float = 40.0,
                 w_active: float = 35.0, 
                 w_flow: float = 15.0, 
                 w_edema: float = 10.0):
        """
        Initializes the analyzer with clinical penalty limits. Total weights must equal 100.
        
        Args:
            w_density: Maximum score deducted for severe avascularity (density <= 3 loops/mm).
            w_active: Maximum score deducted for active structural lesions.
            w_flow: Maximum score deducted for flow stability issues (aggregation).
            w_edema: Maximum score deducted for tissue edema indicators (blur).
        """
        self.w_density = w_density
        self.w_active = w_active
        self.w_flow = w_flow
        self.w_edema = w_edema
        
        # Validates that the maximum possible penalties form a perfect 100-point scale
        if not math.isclose(w_density + w_active + w_flow + w_edema, 100.0):
            raise ValueError("The sum of all penalty weights must strictly equal 100.0")
            
        self.standard_keys = ["Normal", "Abnormal", "Hemo", "Aggregation", "Blur"]

    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Restricts a proportional value strictly within the defined boundaries."""
        return max(min_val, min(max_val, value))

    def normalize_stats(self, raw_stats: Dict[str, int]) -> Dict[str, int]:
        """Normalizes the input dictionary to ensure required keys and standardizes nomenclature."""
        normalized = {key: 0 for key in self.standard_keys}
        for k, v in raw_stats.items():
            mapped_key = "Hemo" if k.lower() in ["hemo", "hemorrhage"] else k
            if mapped_key in normalized:
                normalized[mapped_key] += v
        return normalized

    def compute_density(self, total_capillaries: int, fov: float) -> float:
        """Calculates absolute linear capillary density (loops/mm)."""
        if fov <= 0.0:
            raise ValueError("Field of View (FOV) must be strictly greater than 0.0 mm.")
        return total_capillaries / fov

    def compute_density_penalty(self, density: float) -> float:
        """
        Calculates penalty based on established density norms.
        >= 7.0 loops/mm: 0 penalty.
        <= 3.0 loops/mm: Maximum penalty.
        """
        if density >= 7.0:
            return 0.0
        if density <= 3.0:
            return self.w_density
            
        ratio = (7.0 - density) / 4.0
        return self.w_density * self._clamp(ratio)

    def compute_morphology_penalty(self, stats: Dict[str, int], total_capillaries: int) -> Tuple[float, Dict[str, float]]:
        """
        Calculates threshold-calibrated morphological penalties.
        Returns the total morphological deduction and the individual components.
        """
        if total_capillaries == 0:
            # Complete avascularity defaults to maximum structural failure
            components = {
                "p_active": self.w_active,
                "p_flow": self.w_flow,
                "p_edema": self.w_edema
            }
            return (self.w_active + self.w_flow + self.w_edema), components

        r_abnormal = stats.get("Abnormal", 0) / total_capillaries
        r_hemo = stats.get("Hemo", 0) / total_capillaries
        r_agg = stats.get("Aggregation", 0) / total_capillaries
        r_blur = stats.get("Blur", 0) / total_capillaries

        # Apply pathophysiological weighting and prevalence threshold scaling
        weighted_active_ratio = (self.WEIGHT_ABNORMAL * r_abnormal) + (self.WEIGHT_HEMO * r_hemo)
        
        p_active = self._clamp(weighted_active_ratio * self.SCALE_ACTIVE) * self.w_active
        p_flow = self._clamp(r_agg * self.SCALE_FLOW) * self.w_flow
        p_edema = self._clamp(r_blur * self.SCALE_EDEMA) * self.w_edema

        total_p_morphology = p_active + p_flow + p_edema
        
        components = {
            "p_active": p_active,
            "p_flow": p_flow,
            "p_edema": p_edema
        }
        
        return total_p_morphology, components

    def compute_subscores(self, p_density: float, components: Dict[str, float], mhi: float) -> Dict[str, float]:
        """
        Generates clinical subscores mapped to a 0-100 optimal scale for radar visualization.
        Higher scores indicate healthier physiological status.
        """
        density_adequacy = 100.0 - (p_density / self.w_density) * 100.0 if self.w_density > 0 else 100.0
        structural_integrity = 100.0 - (components.get("p_active", self.w_active) / self.w_active) * 100.0 if self.w_active > 0 else 100.0
        flow_stability = 100.0 - (components.get("p_flow", self.w_flow) / self.w_flow) * 100.0 if self.w_flow > 0 else 100.0
        edema_clarity = 100.0 - (components.get("p_edema", self.w_edema) / self.w_edema) * 100.0 if self.w_edema > 0 else 100.0
        
        return {
            "Density Adequacy": round(max(0.0, min(100.0, density_adequacy)), 1),
            "Structural Integrity": round(max(0.0, min(100.0, structural_integrity)), 1),
            "Flow Stability": round(max(0.0, min(100.0, flow_stability)), 1),
            "Edema/Clarity": round(max(0.0, min(100.0, edema_clarity)), 1),
            "Overall MHI": round(mhi, 1)
        }

    def compute_percentile_rank(self, score: float, reference_scores: Optional[List[float]] = None) -> float:
        """Calculates the empirical or approximated percentile rank of the MHI."""
        if reference_scores and len(reference_scores) > 0:
            sorted_refs = sorted(reference_scores)
            below_count = sum(1 for s in sorted_refs if s < score)
            percentile = (below_count / len(sorted_refs)) * 100.0
        else:
            mu, sigma = 75.0, 10.0
            z_score = (score - mu) / sigma
            percentile = norm.cdf(z_score) * 100.0
            
        return max(1.0, min(99.0, round(percentile, 1)))

    def plot_radar(self, subscores: Dict[str, float], baseline_profile: Optional[Dict[str, float]] = None) -> Tuple[Any, Any]:
        """Generates a polar radar chart comparing patient subscores against a physiological baseline."""
        if not baseline_profile:
            baseline_profile = {
                "Density Adequacy": 90.0,
                "Structural Integrity": 92.0,
                "Flow Stability": 85.0,
                "Edema/Clarity": 88.0,
                "Overall MHI": 89.0
            }

        categories = ["Density Adequacy", "Structural Integrity", "Flow Stability", "Edema/Clarity", "Overall MHI"]
        N = len(categories)

        patient_values = [subscores.get(cat, 0.0) for cat in categories]
        patient_values += patient_values[:1]
        
        baseline_values = [baseline_profile.get(cat, 0.0) for cat in categories]
        baseline_values += baseline_values[:1]

        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4, 3.5), subplot_kw=dict(polar=True), dpi=100)

        plt.xticks(angles[:-1], categories, color='grey', size=8, fontweight='bold')
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
        plt.ylim(0, 100)

        ax.plot(angles, baseline_values, linewidth=1, linestyle='dashed', color='teal', label='Healthy Baseline')
        ax.fill(angles, baseline_values, 'teal', alpha=0.08)

        ax.plot(angles, patient_values, linewidth=1.5, linestyle='solid', color='crimson', label='Patient Profile')
        ax.fill(angles, patient_values, 'crimson', alpha=0.2)

        plt.title('Microvascular Health Profile', size=11, color='black', y=1.1, pad=10)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=8, frameon=False)
        plt.tight_layout(pad=0.5)

        return fig, ax

    def analyze(self, raw_stats: Dict[str, int], fov: float, reference_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """Executes the quantitative clinical health analysis pipeline."""
        stats = self.normalize_stats(raw_stats)
        total_cap = sum(stats.values())
        
        density = self.compute_density(total_cap, fov) if total_cap > 0 else 0.0
        
        p_density = self.compute_density_penalty(density)
        p_morphology, components = self.compute_morphology_penalty(stats, total_cap)
        
        mhi = 100.0 - p_density - p_morphology
        mhi_clamped = max(0.0, min(100.0, round(mhi, 2)))
        
        pr = self.compute_percentile_rank(mhi_clamped, reference_scores)
        subscores = self.compute_subscores(p_density, components, mhi_clamped)
        
        return {
            "normalized_stats": stats,
            "total_capillaries": total_cap,
            "absolute_density": round(density, 2),
            "penalties": {
                "density": round(p_density, 2),
                "morphology_total": round(p_morphology, 2),
                "active_lesions": round(components["p_active"], 2),
                "flow_stasis": round(components["p_flow"], 2),
                "edema": round(components["p_edema"], 2)
            },
            "mhi_score": mhi_clamped,
            "percentile_rank": pr,
            "subscores": subscores
        }


if __name__ == "__main__":
    sample_stats = {
        "Normal": 20,
        "Abnormal": 2,
        "Hemorrhage": 1,
        "Aggregation": 3,
        "Blur": 2
    }
    sample_fov = 3.0
    
    analyzer = ClinicalHealthAnalyzer()
    result = analyzer.analyze(sample_stats, sample_fov)
    
    import pprint
    pprint.pprint(result, sort_dicts=False)