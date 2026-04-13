import os
import cv2
import json
import tempfile
import torch
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Import custom modules
from inference import simplified_post_processing
from clinical_health_analyzer import ClinicalHealthAnalyzer

# ================= Configuration =================
INPUT_DIR = "./data/population_images"    # Directory containing source images
OUTPUT_DB_FILE = "reference_scores.json"  # Output database file
ASSUMED_FOV = 3.0                         # Assumed standard Field of View (mm)

# Model Paths (Ensure consistency with inference.py)
MODEL_FOLDER = Path("./models/nnunet_anfc")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_database():
    """
    Processes a batch of images to calculate health scores and 
    generates a reference database for population distribution analysis.
    """
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Directory not found: {INPUT_DIR}. Please create the folder and add images.")
        return

    # 1. Initialize nnU-Net Predictor
    print("🚀 Initializing nnU-Net model...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, 
        use_gaussian=True, 
        use_mirroring=True,
        perform_everything_on_device=torch.cuda.is_available(),
        device=DEVICE, 
        verbose=False, 
        verbose_preprocessing=False, 
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_FOLDER), 
        use_folds=(0,), 
        checkpoint_name="checkpoint_best.pth"
    )

    # 2. Initialize Clinical Analyzer
    analyzer = ClinicalHealthAnalyzer()
    
    # 3. Prepare file list
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("⚠️ No valid images found in the directory.")
        return
        
    print(f"📂 Found {len(image_files)} images. Starting batch inference...")
    
    # Use tempfile for high-speed batch processing via nnU-Net
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = Path(tmpdir) / "input"
        tmp_out = Path(tmpdir) / "output"
        tmp_in.mkdir()
        tmp_out.mkdir()

        # Copy and rename images to match nnU-Net format (e.g., case_001_0000.png)
        file_mapping = {}
        for idx, filename in enumerate(image_files):
            case_id = f"case_{idx:03d}"
            target_path = tmp_in / f"{case_id}_0000.png"
            
            # Read and save as PNG to ensure format consistency
            img = cv2.imread(os.path.join(INPUT_DIR, filename))
            if img is not None:
                cv2.imwrite(str(target_path), img)
                file_mapping[case_id] = filename

        # Execute Batch Inference
        print("🧠 Running model inference (this may take a few minutes)...")
        predictor.predict_from_files(
            str(tmp_in), 
            str(tmp_out), 
            save_probabilities=False, 
            overwrite=True,
            num_processes_preprocessing=2, 
            num_processes_segmentation_export=2
        )

        # 4. Parse results and calculate scores
        print("📊 Calculating Micro-Health Index (MHI) scores...")
        reference_scores = []
        
        for case_id, original_name in file_mapping.items():
            pred_path = tmp_out / f"{case_id}.png"
            if not pred_path.exists():
                continue
                
            # Load predicted mask and perform post-processing
            raw_mask = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
            stats, _ = simplified_post_processing(raw_mask, min_area=100)
            
            # Calculate absolute score (MHI) via analyzer
            analysis_result = analyzer.analyze(raw_stats=stats, fov=ASSUMED_FOV)
            mhi_score = analysis_result["mhi_score"]
            
            print(f" - {original_name}: MHI = {mhi_score:.2f}")
            reference_scores.append(mhi_score)

    # 5. Save to JSON
    with open(OUTPUT_DB_FILE, 'w') as f:
        json.dump(reference_scores, f)
        
    print(f"\n✅ Database created successfully!")
    print(f"Collected {len(reference_scores)} scores, saved to: {OUTPUT_DB_FILE}")

if __name__ == "__main__":
    build_database()