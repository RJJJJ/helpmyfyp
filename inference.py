import os
from pathlib import Path
import torch
import cv2
import numpy as np
import tempfile
import streamlit as st
import multiprocessing
from skimage.measure import label
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ================= CONFIGURATION & ENVIRONMENT =================
BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FOLDER = BASE_DIR / "models" / "nnunet_anfc"
USE_FOLDS = (0,)
CHECKPOINT_NAME = "checkpoint_best.pth"

NNUNET_RAW = BASE_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED = BASE_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = BASE_DIR / "models"

NNUNET_RAW.mkdir(parents=True, exist_ok=True)
NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)

os.environ["nnUNet_raw"] = str(NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
os.environ["nnUNet_results"] = str(NNUNET_RESULTS)


# ================= IMPORT 2=================
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


_predictor = None

# ✨ Point the path to your newly copied nnU-Net model folder
MODEL_FOLDER = Path("./models/nnunet_anfc")
USE_FOLDS = (0,)
CHECKPOINT_NAME = "checkpoint_best.pth"


# Class Definitions (must match settings from COCO conversion)
CLASSES = {
    0: "Background",
    1: "Abnormal",    # Note: verify ID mapping matches labels_dict from conversion
    2: "Aggregation",
    3: "Blur",
    4: "Hemo",
    5: "Normal"
}


# Visualization Colors
COLOR_MAP = {
    0: (0, 0, 0),      
    1: (255, 0, 255),   # Magenta (Abnormal)
    2: (255, 0, 0),     # Red (Aggregation)
    3: (255, 255, 0),   # Yellow (Blur)
    4: (0, 255, 255),   # Cyan (Hemo)
    5: (0, 255, 0)      # Green (Normal)
}


@st.cache_resource(show_spinner=False)
def get_predictor():
    """
    環境感知的 Singleton 模型載入器 (Environment-Aware Loader)。
    自動偵測硬體環境，為 GPU 啟用極速模式，為 CPU 啟用優雅降級模式。
    """
    print("🚀 [MLOps] Initializing nnU-Net Predictor...")
    
    if not MODEL_FOLDER.exists():
        raise FileNotFoundError(f"❌ 找不到 nnU-Net 模型資料夾: {MODEL_FOLDER}")

    # 1. 動態設備偵測 (Dynamic Device Routing)
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    
    if not has_cuda:
        # 2. CPU 效能優化 (Thread Throttling)
        # 避免 PyTorch 佔用所有核心導致 Streamlit 伺服器卡死
        optimal_threads = min(4, multiprocessing.cpu_count())
        torch.set_num_threads(optimal_threads)
        print(f"⚠️ [MLOps] 偵測到 CPU 環境。已將 PyTorch 執行緒限制為 {optimal_threads} 以優化效能。")
    else:
        print("⚡ [MLOps] 偵測到 GPU (CUDA) 環境。即將啟用極速推論模式。")

    try:
        # 3. 初始化 Predictor，動態配置 perform_everything_on_device
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            # 關鍵修改：只有在 CUDA 環境下才允許 device-level preprocessing
            perform_everything_on_device=has_cuda, 
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )

        predictor.initialize_from_trained_model_folder(
            str(MODEL_FOLDER),
            use_folds=USE_FOLDS,
            checkpoint_name=CHECKPOINT_NAME,
        )

        # === 加入這兩行 ===
        # 強制關閉多進程匯出與預處理，避免在 CPU 上發生 Thread 互相阻塞
        predictor.allowed_num_processes = 1 
        predictor.num_processes_preprocessing = 1
        
        # 4. 全環境相容的模型預熱 (Universal Warm-up)
        print(f"🔥 [MLOps] Warming up {device.type.upper()} Context...")
        
        # 注意：這裡的 device 參數會自動將張量放在 CPU 或 GPU 上
        dummy_input = torch.randn(1, 3, 640, 640, device=device, dtype=torch.float32)
        with torch.no_grad():
            _ = predictor.network(dummy_input)
            
        # 只有在 CUDA 環境下才需要同步等待
        if has_cuda:
            torch.cuda.synchronize() 
            
        print(f"✅ [MLOps] Predictor 已準備就緒 (運行於 {device.type.upper()})。")
        return predictor
        
    except Exception as e:
        raise RuntimeError(f"❌ Predictor 初始化失敗: {str(e)}")

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    [CRITICAL FIX] Force image resizing to resolve coordinate offset issues.
    """
    dim = None
    (h, w) = image.shape[:2]


    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))


    return cv2.resize(image, dim, interpolation=inter)

def get_magic_wand_region(full_image_rgb, x, y, tolerance=30):
    """
    [ALGORITHM OPTIMIZATION] Add area anomaly detection to prevent oversized annotations.
    """
    h, w, _ = full_image_rgb.shape
    img_area = h * w
   
    # 1. Image Enhancement
    lab = cv2.cvtColor(full_image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a, b))
    enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


    # 2. Auto-snap to darkest point
    search_r = 8
    x1, y1 = max(0, x - search_r), max(0, y - search_r)
    x2, y2 = min(w, x + search_r), min(h, y + search_r)
    local_roi = enhanced_l[y1:y2, x1:x2]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(local_roi)
    seed_point = (x1 + min_loc[0], y1 + min_loc[1])


    # 3. Execute Flood Fill
    mask = np.zeros((h+2, w+2), np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(enhanced_rgb, mask, seed_point, (255, 0, 0),
                  (tolerance, tolerance, tolerance),
                  (tolerance, tolerance, tolerance), flags)
   
    binary_mask = mask[1:-1, 1:-1]
   
    # 4. Morphological Cleaning
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
   
    # 5. Convert to contours and perform area plausibility check
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    target_cnt = None
    max_area = 0
    for cnt in contours:
        dist = cv2.pointPolygonTest(cnt, seed_point, True)
        if dist > -5:
            area = cv2.contourArea(cnt)
            # --- CRITICAL CHANGE: If a single contour exceeds 5% of full image area, consider invalid ---
            if area > (img_area * 0.05):
                continue
            if area > max_area:
                max_area = area
                target_cnt = cnt
               
    # 6. If no suitable contour is found or area is too small/large, return None to trigger Alert
    if target_cnt is None or max_area < 50:
        return None
       
    return target_cnt


def draw_filled_highlight(img, contour, color=(0, 255, 255), alpha=0.4):
    """
    [VISUAL OPTIMIZATION] Draw translucent fill effect (Cyan Glow).
    """
    overlay = img.copy()
    cv2.drawContours(overlay, [contour], -1, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.drawContours(img, [contour], -1, (255, 255, 255), 1)

def process_image(image_file):


    # 1. 讀取影像至記憶體
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
   
    # 2. 獲取已載入的模型
    predictor = get_predictor()
    if predictor is None:
        raise ValueError("❌ 模型尚未初始化。")

    # 3. 準備基礎張量格式 (C, H, W)
    input_array = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)

    # 🚀【防呆核心：動態維度適配】
    num_spatial_dims = len(predictor.plans_manager.transpose_forward)

    if num_spatial_dims == 3:
        if input_array.ndim == 3:
            input_array = np.expand_dims(input_array, axis=1)
        dummy_properties = {'spacing': [999.0, 1.0, 1.0]} 
    else:
        if input_array.ndim == 4:
            input_array = np.squeeze(input_array, axis=0) 
        dummy_properties = {'spacing': [1.0, 1.0]} 

    print(f"⚡ [MLOps] In-Memory 推論開始... 陣列已動態修正為: {input_array.shape}")
    
    # 4. 執行全記憶體推論
    with torch.no_grad():
        # 💡 【核心修復點】：解決 unpack error
        # 直接用一個變數接住回傳結果，然後透過型別判斷動態提取，
        # 這樣無論 nnU-Net 回傳的是 Tuple 還是單一 Numpy Array 都不會崩潰。
        prediction_result = predictor.predict_single_npy_array(
            input_image=input_array, 
            image_properties=dummy_properties, 
            segmentation_previous_stage=None, 
            output_file_truncated=None, 
            save_or_return_probabilities=False
        )
    
    # 動態提取預測的 Mask
    if isinstance(prediction_result, tuple):
        pred_mask = prediction_result[0]
    else:
        pred_mask = prediction_result
    
    # 5. 後處理：將輸出的 Mask 壓平回乾淨的 2D 陣列 (H, W)
    pred_mask = np.squeeze(pred_mask).astype(np.uint8)
    print("✅ [MLOps] In-Memory 推論完成！")

    # 接續呼叫原有的後處理邏輯
    stats, cleaned_mask = simplified_post_processing(
        pred_mask, 
        min_area=st.session_state.get('current_min_area', 100)
    )
    result_img = draw_result_on_image(image_rgb, cleaned_mask)
   
    return image_rgb, result_img, stats, pred_mask


def simplified_post_processing(raw_mask, min_area=100):
    """
    nnU-Net predictions are usually high quality; filter tiny noise and count objects.
    [Modified] Supports dynamic min_area parameter.
    """
    final_stats = {}
    cleaned_mask = np.zeros_like(raw_mask)
   
    for cls_id, cls_name in CLASSES.items():
        if cls_id == 0: continue
       
        binary_mask = (raw_mask == cls_id).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
       
        count = 0
        for i in range(1, num_labels):
            # [Modified] Use passed min_area variable instead of hardcoded constant
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = cls_id
                count += 1
               
        if count > 0:
            final_stats[cls_name] = count
           
    return final_stats, cleaned_mask


def draw_result_on_image(original_image, mask):
    """
    Image overlay logic (maintains elegant glowing borders and labeling effects)
    """
    color_mask = np.zeros_like(original_image)
    for cls_id, color in COLOR_MAP.items():
        if cls_id == 0: continue
        color_mask[mask == cls_id] = color
       
    mask_bool = (mask > 0)
    final_img = original_image.copy()
    final_img[mask_bool] = cv2.addWeighted(original_image, 0.6, color_mask, 0.5, 0)[mask_bool]


    for cls_id, class_name in CLASSES.items():
        if cls_id == 0: continue
        binary_mask = np.uint8(mask == cls_id) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        count = 0
        for cnt in contours:
            # Only label sufficiently large regions
            if cv2.contourArea(cnt) < 50: continue
           
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                label_text = f"{class_name[0]}{count+1}"
                cv2.putText(final_img, label_text, (cX-10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                cv2.putText(final_img, label_text, (cX-10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                count += 1
               
    return final_img


def recalculate_overlay(image_rgb, raw_mask, min_area):
    """
    Receives original image and unfiltered raw_mask; recalculates statistics and visual overlay based on new min_area.
    """
    new_stats, new_cleaned_mask = simplified_post_processing(raw_mask, min_area)
    new_overlay_img = draw_result_on_image(image_rgb, new_cleaned_mask)
    
    return new_stats, new_cleaned_mask, new_overlay_img