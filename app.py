import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import cv2
import inference
import report_generator
import json
import matplotlib.pyplot as plt
from datetime import date
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from clinical_health_analyzer import ClinicalHealthAnalyzer
from clinical_risk import ClinicalRiskProfiler
from dotenv import load_dotenv
from datetime import date, datetime  # 修改這行：加入 datetime
import io                            # 新增這行：處理二進位圖片流


# 載入 .env 檔案
load_dotenv()           
api_key = os.getenv("API_KEY")


try:
    from google import genai
    USE_NEW_GENAI = True
except ImportError:
    import google.generativeai as genai
    USE_NEW_GENAI = False

# ================= 1. PAGE CONFIG & STYLING =================
st.set_page_config(layout="wide", page_title="Microcirculation Research Platform")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --color-primary: #173B45;
        --color-secondary: #5F7C82;
        --color-bg: #F6F4EF;
        --color-surface: #FCFBF8;
        --color-viewer-dark: #20282D;
        --color-accent: #C58B47;
        --color-success: #6D8A63;
        --color-warning: #C58B47;
        --color-danger: #A45A52;
        --color-info: #7A8E98;
        --color-border: #d9d7d2;
        --radius-md: 12px;
        --space-md: 16px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--color-primary);
    }

    .stApp {
        background: var(--color-bg);
    }

    .layer-wrap {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        margin-bottom: 14px;
    }

    .global-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
    }

    .page-title { font-size: 1.9rem; font-weight: 700; line-height: 1.2; letter-spacing: 0.01em; }
    .section-title { font-size: 1.25rem; font-weight: 600; color: var(--color-primary); margin-bottom: 8px; }
    .panel-title { font-size: 1rem; font-weight: 600; color: var(--color-primary); margin-bottom: 10px; }
    .card-header { font-size: 1rem; font-weight: 600; color: var(--color-primary); margin-bottom: 10px; }
    .body-text { font-size: 0.96rem; font-weight: 400; color: var(--color-secondary); }
    .caption-text { font-size: 0.84rem; color: var(--color-secondary); }
    .micro-label { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--color-info); font-weight: 600; }

    .status-badge {
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(23, 59, 69, 0.25);
        color: var(--color-primary);
        background: rgba(95, 124, 130, 0.10);
        white-space: nowrap;
    }

    .workflow-strip {
        display: grid;
        grid-template-columns: repeat(6, minmax(110px, 1fr));
        gap: 10px;
    }

    .workflow-stage {
        border: 1px solid var(--color-border);
        border-radius: 10px;
        background: #f8f6f1;
        padding: 8px 10px;
        min-height: 64px;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .workflow-stage:hover { background: #f2eee7; }
    .workflow-stage.current { border-color: var(--color-accent); background: #f5ece0; }
    .workflow-stage.done { border-color: var(--color-success); background: #eef3ec; }
    .workflow-stage.ready { border-color: #4b6f57; background: #e6efe8; box-shadow: 0 0 0 1px rgba(109,138,99,0.2) inset; }
    .workflow-stage.disabled { opacity: 0.72; }
    .workflow-stage-icon { font-size: 0.85rem; margin-bottom: 2px; }
    .study-card {
        border: 1px solid var(--color-border);
        border-radius: 10px;
        background: #fbfaf7;
        padding: 10px;
        min-height: 320px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .study-meta-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        margin-top: 8px;
    }
    .study-summary-card {
        border: 1px solid var(--color-border);
        border-radius: 10px;
        background: #fbfaf7;
        padding: 12px;
        min-height: 280px;
    }
    .badge-muted {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 0.74rem;
        border: 1px solid #d8d5cf;
        color: #4d6268;
        background: #f4f2ee;
        margin-right: 6px;
    }

    .card-panel {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: var(--space-md);
    }

    .viewer-panel {
        background: var(--color-viewer-dark);
        border-radius: var(--radius-md);
        padding: 12px;
        border: 1px solid #2f3b42;
    }
    .viewer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #d9e2e7;
        margin-bottom: 8px;
    }
    .viewer-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e6eef2;
    }
    .workstation-strip {
        background: #263238;
        border: 1px solid #324147;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
    }
    .clinical-card {
        background: #fbfaf7;
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .risk-card {
        background: #f8f7f4;
        border: 1px solid #d8d5cf;
        border-radius: 10px;
        padding: 12px;
        min-height: 170px;
    }
    .insight-card {
        border: 1px solid #d8d5cf;
        border-radius: 10px;
        background: #fcfbf8;
        padding: 12px;
        min-height: 145px;
    }
    .insight-chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .insight-positive { background: #eaf3e7; color: #45653c; border: 1px solid #c8dbc2; }
    .insight-neutral { background: #eef2f4; color: #4d6770; border: 1px solid #d0dade; }
    .insight-concern { background: #f8ecea; color: #8a4d47; border: 1px solid #e7cbc6; }
    .risk-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        background: #ece9e2;
        border: 1px solid #d5d0c8;
        color: #4b5f65;
    }
    .audit-row {
        border: 1px solid #ddd9d2;
        border-radius: 8px;
        background: #fcfbf9;
        padding: 8px 10px;
        margin-bottom: 8px;
    }
    .mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }

    [data-testid="stMetricLabel"] { font-size: 0.84rem !important; font-weight: 600 !important; color: var(--color-secondary) !important; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; color: var(--color-primary) !important; }

    div[data-testid="stDialog"] div[role="dialog"] {
        width: 480px;
        border-radius: 12px;
        border: 1px solid var(--color-border);
    }

    .final-output-zone {
        background: linear-gradient(180deg, #f8f6f1 0%, #f3f0e8 100%);
        border: 1px solid #d6d2c9;
        border-radius: 14px;
        padding: 16px;
        margin-top: 16px;
        margin-bottom: 16px;
    }
    .finalization-card {
        background: #fcfbf8;
        border: 1px solid #d9d7d2;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .report-header-grid {
        display: grid;
        grid-template-columns: 2fr 1.2fr 1.2fr;
        gap: 10px;
        margin: 10px 0 12px 0;
    }
    .report-header-cell {
        border: 1px solid #ddd8ce;
        border-radius: 10px;
        background: #fbfaf7;
        padding: 10px;
    }
    .report-status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        border: 1px solid #9fb09a;
        color: #2e4a35;
        background: #e8f2e6;
    }
    .report-narrative-shell {
        border: 1px solid #d8d4ca;
        border-radius: 12px;
        background: #fffefb;
        padding: 18px;
    }
    .report-narrative-body {
        max-width: 860px;
        margin: 0 auto;
        line-height: 1.75;
        font-size: 0.98rem;
        color: #243b44;
        white-space: pre-wrap;
    }
    .hero-result-card {
        background: linear-gradient(180deg, #fffdf8 0%, #f6f0e3 100%);
        border: 1px solid #d8cfbd;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 12px;
    }
    .hero-score {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.1;
        color: #10353f;
        margin: 6px 0 8px 0;
    }
    .hero-percentile {
        font-size: 1.25rem;
        font-weight: 600;
        color: #2a4f59;
        margin-top: 4px;
    }
    .hero-summary {
        font-size: 0.98rem;
        color: #3f5960;
        line-height: 1.6;
        margin-top: 6px;
    }
    .grade-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid #d4d0c8;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }
    .grade-excellent { background: #e8f2e6; border-color: #a9c5a2; color: #2e4a35; }
    .grade-good { background: #edf3f6; border-color: #b9cad3; color: #294752; }
    .grade-fair { background: #f8f0e4; border-color: #d9be97; color: #6d4d2b; }
    .grade-attention { background: #f7e8e6; border-color: #d8ada7; color: #7a3f38; }
    .result-label-chip {
        display: inline-block;
        margin-top: 10px;
        padding: 5px 10px;
        border-radius: 999px;
        border: 1px solid #d3d0c8;
        background: #f4f2ed;
        color: #3a4f56;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .user-soft-card {
        background: #fdfcf9;
        border: 1px solid #ddd8ce;
        border-radius: 12px;
        padding: 14px;
    }
    .user-secondary {
        background: #faf8f3;
        border: 1px solid #e3dfd6;
        border-radius: 10px;
        padding: 12px;
    }
    .research-emphasis {
        border-left: 3px solid #c58b47;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ================= CONSTANTS =================
UI_COLOR_MAP = {
    "Normal": (0, 255, 0),       # Green
    "Abnormal": (255, 0, 255),   # Purple/Magenta
    "Hemorrhage": (0, 255, 255), # Cyan/Yellow (BGR vs RGB depending on cv2 format, usually RGB in st)
    "Aggregation": (255, 0, 0),  # Red
    "Blur": (255, 255, 0)        # Yellow
}

# ================= STATE MANAGEMENT =================
if 'user_data' not in st.session_state: st.session_state.user_data = None
if 'run_analysis' not in st.session_state: st.session_state.run_analysis = False
if 'inference_done' not in st.session_state: st.session_state.inference_done = False
if 'manual_regions' not in st.session_state: st.session_state.manual_regions = []
if 'last_clicked_coords' not in st.session_state: st.session_state.last_clicked_coords = None
# --- 新增：Gallery 專用的狀態管理 ---
if 'is_previewing' not in st.session_state: st.session_state.is_previewing = False
if 'confirmed_file' not in st.session_state: st.session_state.confirmed_file = False
if 'selected_local_file' not in st.session_state: st.session_state.selected_local_file = None
if 'manual_uploaded_file' not in st.session_state: st.session_state.manual_uploaded_file = None
if 'upload_source' not in st.session_state: st.session_state.upload_source = None
if 'open_patient_dialog' not in st.session_state: st.session_state.open_patient_dialog = False
if 'analysis_requested' not in st.session_state: st.session_state.analysis_requested = False
# ================= MLOPS: PRE-LOADING & WARM-UP =================
# 在任何 UI 元件渲染之前，強制執行一次模型加載與預熱。
# 由於使用了 @st.cache_resource，這段程式碼只會在伺服器重啟後的第一次載入時耗時。
with st.spinner("Initializing analysis pipeline..."):
    _ = inference.get_predictor()
    
# ================= HELPER FUNCTIONS =================
@st.cache_data
def load_reference_database():
    try:
        with open("reference_scores.json", "r") as f:
            scores = json.load(f)
            return scores
    except FileNotFoundError:
        return None 
def get_default_baseline_profile():
    """Returns a fixed healthy baseline for radar chart comparison."""
    return {
        "Density Adequacy": 90.0,
        "Structural Integrity": 92.0,
        "Flow Stability": 85.0,
        "Edema/Clarity": 88.0,
        "Overall MHI": 89.0
    }

def compute_baseline_profile_from_database(db_scores):
    """
    Computes dynamic baseline from reference database.
    Note: Currently reference_scores.json only stores overall MHI float values.
    Once the DB is updated to store full subscore dicts, implement the average logic here.
    For now, it securely falls back to the default profile.
    """

    # if db_scores and isinstance(db_scores, list) and isinstance(db_scores[0], dict):
    #     return calculate_mean_profile_from_dicts(db_scores)
    
    return get_default_baseline_profile()

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
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

def display_reference_table():
    with st.expander("📖 Reference Guide: Terminology Definitions", expanded=False):
        st.markdown("""
        | **Term** | **Icon** | **Description** | **Clinical Relevance** |
        | :--- | :---: | :--- | :--- |
        | **Normal** | 🟢 | U-shaped, hair-pin like loops. | Healthy microvascular status. |
        | **Abnormal** | 🟣 | **Giant (>50µm)**, twisted loops. | Specific for Systemic Sclerosis (SSc). |
        | **Hemorrhage** | 🔵 | Extravasation of blood. | Active microvascular damage. |
        | **Aggregation** | 🔴 | Sludge flow, congested loops. | Related to Raynaud's Phenomenon. |
        | **Blur** | 🟡 | Indistinct borders. | Can imply tissue edema. |
        """)

def plot_capillary_distribution(stats):
    """Generates independent horizontal progress bars for each capillary type."""
    categories = ['Normal', 'Blur', 'Abnormal', 'Hemo', 'Aggregation']
    # Calculate total strictly from morphology categories (ignores 'Physician_Added')
    total = sum(stats.get(cat, 0) for cat in categories)
    total = total if total > 0 else 1 
    
    colors = ['#00FF00', '#FFFF00', '#800080', '#00FFFF', '#FF0000']
    
    data = []
    for cat in categories:
        val = stats.get(cat, 0)
        pct = (val / total) * 100
        data.append({'Type': cat, 'Percentage': pct, 'Max': 100, 'Label': f"{pct:.1f}%"})
        
    df = pd.DataFrame(data)
    
    bg = alt.Chart(df).mark_bar(color='#f0f0f0', height=20).encode(
        x=alt.X('Max:Q', scale=alt.Scale(domain=[0, 100]), 
                axis=alt.Axis(values=[0, 100], title=None, grid=False, domain=False, ticks=False)),
        y=alt.Y('Type:N', sort=categories, title=None, axis=alt.Axis(grid=False, tickBand='extent'))
    )
    
    fg = alt.Chart(df).mark_bar(height=20).encode(
        x=alt.X('Percentage:Q'),
        y=alt.Y('Type:N', sort=categories),
        color=alt.Color('Type:N', scale=alt.Scale(domain=categories, range=colors), legend=None),
        tooltip=['Type', alt.Tooltip('Percentage:Q', format='.1f')]
    )
    
    text = alt.Chart(df).mark_text(align='left', baseline='middle', dx=5, fontWeight='bold').encode(
        x=alt.X('Percentage:Q'),
        y=alt.Y('Type:N', sort=categories),
        text=alt.Text('Label:N')
    )
    
    chart = (bg + fg + text).properties(
        height=250, title="Capillary Composition Breakdown"
    ).configure_view(strokeWidth=0)
    
    return chart

def plot_health_score_bar(score):
    """Generates a horizontal progress bar for the Overall Health Score."""
    df = pd.DataFrame({'Score': [score], 'Bar': ['Health Score'], 'Max': [100], 'Label': f"{score}"})
    
    bg = alt.Chart(df).mark_bar(color='#f0f0f0', height=30).encode(
        x=alt.X('Max:Q', scale=alt.Scale(domain=[0, 100]), 
                axis=alt.Axis(values=[0, 100], title=None, grid=False, domain=False, ticks=False)),
        y=alt.Y('Bar:N', title=None, axis=alt.Axis(labels=False, ticks=False, grid=False))
    )
    
    fg = alt.Chart(df).mark_bar(color='#0e2a47', height=30).encode(
        x='Score:Q',
        y=alt.Y('Bar:N')
    )
    
    text = alt.Chart(df).mark_text(align='left', baseline='middle', dx=5, fontSize=14, fontWeight='bold').encode(
        x='Score:Q',
        y='Bar:N',
        text=alt.Text('Label:N')
    )
    
    chart = (bg + fg + text).properties(
        height=80, title=f"Overall Health Score"
    ).configure_view(strokeWidth=0)
    
    return chart

def plot_gaussian_comparison(density_val):
    mu, sigma = 9, 2
    x = np.linspace(2, 16, 200)
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (x - mu))**2)
    source = pd.DataFrame({'Density': x, 'Probability': y})
    base = alt.Chart(source).mark_area(opacity=0.3, color='#4c78a8').encode(
        x=alt.X('Density', title='Capillary Density (loops/mm)'), y=alt.Y('Probability', axis=None)
    )
    rule = alt.Chart(pd.DataFrame({'x': [density_val]})).mark_rule(color='red', size=4).encode(x='x')
    return (base + rule).properties(title="Comparison vs. Healthy Norm (Etehad Tavakol et al., 2015)", height=250)

def get_workflow_stage_index():
    """Returns current workflow index (1-6) for UI strip rendering only."""
    if not st.session_state.get('confirmed_file', False):
        if st.session_state.get('is_previewing', False):
            return 2
        return 1
    if st.session_state.get('confirmed_file', False) and not st.session_state.get('analysis_requested', False):
        return 2
    if st.session_state.get('analysis_requested', False) and (
        st.session_state.get('open_patient_dialog', False) or not st.session_state.get('user_data')
    ):
        return 3
    if st.session_state.get('analysis_requested', False) and st.session_state.get('user_data') and not st.session_state.get('inference_done', False):
        return 4
    if st.session_state.get('inference_done', False) and not st.session_state.get('report_generated', False):
        return 6
    if st.session_state.get('report_generated', False):
        return 6
    return 2

def render_global_header():
    st.markdown(
        """
        <div class="layer-wrap">
            <div class="global-header">
                <div>
                    <div class="micro-label">Microcirculation Platform</div>
                    <div class="page-title">Microcirculation Research Platform</div>
                    <div class="body-text">Quantitative Nailfold Capillaroscopy Analysis</div>
                </div>
                <div class="status-badge">Research Prototype</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_workflow_strip():
    stages = [
        "Select Image",
        "Confirm Input",
        "Subject Metadata",
        "Run Analysis",
        "Review & Validate",
        "Generate Report",
    ]
    current = get_workflow_stage_index()
    icons = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥"}
    cards = []
    report_ready = st.session_state.get('report_generated', False)
    for i, name in enumerate(stages, start=1):
        if i == 6 and report_ready:
            status_class = "ready"
            stage_icon = "✅"
        elif i < current:
            status_class = "done"
            stage_icon = "✅"
        elif i == current:
            status_class = "current"
            stage_icon = icons[i]
        else:
            status_class = "disabled"
            stage_icon = icons[i]
        cards.append(
            f"""
            <div class="workflow-stage {status_class}">
                <div class="workflow-stage-icon">{stage_icon}</div>
                <div class="micro-label">Stage {i}</div>
                <div class="caption-text"><b>{name}</b></div>
            </div>
            """
        )
    st.markdown(
        f"""
        <div class="layer-wrap">
            <div class="section-title">Workflow Strip</div>
            <div class="workflow-strip">
                {''.join(cards)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def collect_device_studies(img_dir="machine/Guests-Image/Guest"):
    studies = []
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                full_path = os.path.join(img_dir, f)
                studies.append({
                    "path": full_path,
                    "filename": os.path.basename(full_path),
                    "study_id": os.path.splitext(os.path.basename(full_path))[0],
                    "capture_time": datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M"),
                    "source": "Device",
                    "status": "Available"
                })
    studies.sort(key=lambda x: os.path.getmtime(x["path"]), reverse=True)
    return studies

def render_study_browser():
    studies = collect_device_studies()
    if st.session_state.get("manual_uploaded_file") is not None:
        manual_name = st.session_state.manual_uploaded_file.name
        studies.insert(0, {
            "path": None,
            "filename": manual_name,
            "study_id": f"upload-{os.path.splitext(manual_name)[0]}",
            "capture_time": "Session Upload",
            "source": "Uploaded",
            "status": "Ready"
        })

    top_l, top_m, top_r, top_f = st.columns([3, 1.3, 1, 2])
    with top_l:
        st.markdown("### Study Browser")
        st.caption("Structured imaging studies for selection and verification")
    with top_m:
        st.metric("Total Studies", len(studies))
    with top_r:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    with top_f:
        st.selectbox("Filter (placeholder)", ["Recent", "Device", "Uploaded"], index=0)

    manual_upload = st.file_uploader("Add Study (Manual Upload)", type=["jpg", "png", "jpeg", "bmp"])
    if manual_upload is not None:
        st.session_state.manual_uploaded_file = manual_upload
        st.session_state.selected_local_file = None
        st.session_state.upload_source = "manual"
        st.session_state.is_previewing = True
        st.rerun()

    if not studies:
        st.info("No studies found. Use manual upload to add a study.")
        return

    grid_cols = 3
    for i in range(0, len(studies), grid_cols):
        cols = st.columns(grid_cols)
        for j, col in enumerate(cols):
            if i + j >= len(studies):
                continue
            study = studies[i + j]
            with col:
                st.markdown('<div class="study-card">', unsafe_allow_html=True)
                if study["source"] == "Device" and study["path"]:
                    st.image(Image.open(study["path"]), use_container_width=True)
                else:
                    st.image(st.session_state.manual_uploaded_file, use_container_width=True)

                st.markdown(f"**{study['study_id']}**")
                st.markdown(f"<span class='badge-muted'>{study['source']}</span><span class='badge-muted'>{study['status']}</span>", unsafe_allow_html=True)
                st.caption(f"Capture: {study['capture_time']}")
                open_key = f"open_study_{i+j}"
                if st.button("Open Study", key=open_key, use_container_width=True):
                    if study["source"] == "Device":
                        st.session_state.selected_local_file = study["path"]
                        st.session_state.upload_source = "device"
                    else:
                        st.session_state.selected_local_file = None
                        st.session_state.upload_source = "manual"
                    st.session_state.is_previewing = True
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

def render_confirm_input():
    selected_source = st.session_state.get("upload_source")
    selected_name = "N/A"
    capture_time = "N/A"
    preview_obj = None
    status = "Ready for confirmation"

    if selected_source == "device" and st.session_state.get("selected_local_file"):
        fpath = st.session_state.selected_local_file
        selected_name = os.path.basename(fpath)
        preview_obj = Image.open(fpath)
        capture_time = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M")
    elif selected_source == "manual" and st.session_state.get("manual_uploaded_file"):
        up = st.session_state.manual_uploaded_file
        selected_name = up.name
        preview_obj = up
        capture_time = "Session Upload"

    st.markdown("### Confirm Input")
    left, right = st.columns([3, 2], gap="large")
    with left:
        if preview_obj is not None:
            st.image(preview_obj, caption=f"Preview: {selected_name}", use_container_width=True)
        else:
            st.warning("No study selected.")
    with right:
        st.markdown('<div class="study-summary-card">', unsafe_allow_html=True)
        st.markdown("#### Study Summary")
        st.markdown(f"**Study/File ID**: `{selected_name}`")
        st.markdown(f"**Source**: {selected_source.capitalize() if selected_source else 'N/A'}")
        st.markdown(f"**Capture Time**: {capture_time}")
        st.markdown(f"**Status**: {status}")
        st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.5, 0.2, 2])
    with c1:
        if st.button("Back to Gallery", use_container_width=True):
            st.session_state.is_previewing = False
            st.session_state.selected_local_file = None
            if st.session_state.get("upload_source") == "manual":
                st.session_state.manual_uploaded_file = None
            st.rerun()
    with c3:
        if st.button("Confirm Input", type="primary", use_container_width=True):
            st.session_state.is_previewing = False
            st.session_state.confirmed_file = True
            st.rerun()

def get_selected_study_context(uploaded_file=None):
    source = st.session_state.get('upload_source', 'N/A')
    study_id = getattr(uploaded_file, "name", "N/A") if uploaded_file is not None else "N/A"
    image_id = "N/A"
    capture_date = date.today().strftime("%Y-%m-%d")
    if source == "device" and st.session_state.get("selected_local_file"):
        fpath = st.session_state.selected_local_file
        study_id = os.path.splitext(os.path.basename(fpath))[0]
        image_id = os.path.basename(fpath)
        capture_date = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d")
    elif source == "manual" and st.session_state.get("manual_uploaded_file"):
        fname = st.session_state.manual_uploaded_file.name
        study_id = os.path.splitext(fname)[0]
        image_id = fname
    return {
        "source": source.capitalize() if source else "N/A",
        "study_id": study_id,
        "image_id": image_id,
        "date": capture_date
    }

def render_subject_context_card(uploaded_file=None, show_edit=False):
    user = st.session_state.get("user_data") or {}
    ctx = get_selected_study_context(uploaded_file)
    st.markdown("#### Subject Context")
    st.markdown('<div class="study-summary-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Subject ID**  \n`{user.get('name', 'Not set')}`")
    c2.markdown(f"**Age / Gender**  \n{user.get('age', 'Not set')} / {user.get('gender', 'Not set')}")
    c3.markdown(f"**FOV**  \n{user.get('fov', 'Not set')} mm")
    c4, c5, c6 = st.columns(3)
    c4.markdown(f"**Date**  \n{ctx['date']}")
    c5.markdown(f"**Source**  \n{ctx['source']}")
    c6.markdown(f"**Study/Image ID**  \n`{ctx['study_id']} / {ctx['image_id']}`")
    if show_edit:
        if st.button("Edit Metadata", key=f"edit_metadata_{ctx['image_id']}", use_container_width=False):
            st.session_state.open_patient_dialog = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_stage4_readiness():
    st.markdown("### Run Analysis")
    st.info("Analysis readiness confirmed. Pipeline is prepared for execution.")
    with st.container(border=True):
        st.markdown("**Pipeline Steps**")
        st.markdown("- Model Warm-Up")
        st.markdown("- Image Inference")
        st.markdown("- Segmentation Overlay")
        st.markdown("- Morphology Statistics")
    if st.button("Run Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
        st.rerun()

def get_risk_interpretation(risk_name, score):
    if score >= 70:
        level_text = "High"
    elif score >= 40:
        level_text = "Moderate"
    else:
        level_text = "Low"
    return f"{risk_name} risk is {level_text.lower()}, suggesting closer longitudinal follow-up."

def get_health_band(score):
    score = float(score)
    if score >= 85:
        return {"label": "Excellent", "badge_class": "grade-excellent"}
    if score >= 70:
        return {"label": "Good", "badge_class": "grade-good"}
    if score >= 55:
        return {"label": "Fair", "badge_class": "grade-fair"}
    return {"label": "Needs Attention", "badge_class": "grade-attention"}

def get_percentile_text(percentile):
    pct = int(round(percentile))
    return f"Higher than {pct}% of comparable users", pct

def get_personal_result_label(score, risk_profile, live_stats):
    band = get_health_band(score)["label"]
    levels = risk_profile.get("risk_levels", {})
    high_risk_count = sum(1 for key in ["structural", "raynaud", "edema"] if levels.get(key) == "High")
    abnormal_like = int(live_stats.get("Abnormal", 0)) + int(live_stats.get("Hemo", 0)) + int(live_stats.get("Aggregation", 0))
    normal = int(live_stats.get("Normal", 0))

    if band == "Excellent" and high_risk_count == 0 and normal >= abnormal_like:
        return "Stable Flow Profile"
    if band in ["Excellent", "Good"] and high_risk_count <= 1:
        return "Balanced Microcirculation Profile"
    if high_risk_count >= 2:
        return "Follow-Up Recommended Profile"
    return "Mild Structural Variation Profile"

def get_personal_result_statement(score, risk_profile, live_stats):
    levels = risk_profile.get("risk_levels", {})
    structural_level = levels.get("structural", "Moderate").lower()
    edema_level = levels.get("edema", "Moderate").lower()
    band = get_health_band(score)["label"]

    if band in ["Excellent", "Good"] and structural_level in ["low", "moderate"]:
        return "Your current microcirculation pattern appears generally stable, with mild structural variation."
    if edema_level == "high":
        return "Your profile suggests a stable baseline with signs of edema-related variation that may benefit from follow-up."
    if levels.get("raynaud", "").lower() == "high":
        return "Your profile shows meaningful flow variability, and trend monitoring is recommended."
    return "Your profile shows mixed microcirculatory features, and periodic follow-up can help track changes over time."

def get_score_drivers(health_score, final_density, live_stats, risk_profile):
    abnormal_like = int(live_stats.get("Abnormal", 0)) + int(live_stats.get("Hemo", 0)) + int(live_stats.get("Aggregation", 0))
    normal = int(live_stats.get("Normal", 0))
    levels = risk_profile.get("risk_levels", {})
    structural_level = levels.get("structural", "Moderate").lower()
    edema_level = levels.get("edema", "Moderate").lower()

    density_direction = "positive" if final_density >= 7 else "concern" if final_density < 5 else "neutral"
    density_line = (
        "Your density profile supported a stronger overall score."
        if density_direction == "positive"
        else "Your density level was acceptable but not a major score driver."
        if density_direction == "neutral"
        else "Lower density had a noticeable downward effect on your score."
    )
    density_support = f"Measured density: {final_density:.1f}/mm, interpreted against the healthy reference range."

    structure_direction = "positive" if normal >= abnormal_like and structural_level == "low" else "concern" if structural_level == "high" else "neutral"
    structure_line = (
        "Your morphology profile remained mostly regular, which supported score stability."
        if structure_direction == "positive"
        else "Mild structural irregularities had the biggest downward effect."
        if structure_direction == "concern"
        else "Structural variation was present, with a moderate influence on the score."
    )
    structure_support = f"Normal loops: {normal}, irregular-pattern loops: {abnormal_like}."

    edema_direction = "positive" if edema_level == "low" else "concern" if edema_level == "high" else "neutral"
    edema_line = (
        "Edema-related indicators remained relatively controlled."
        if edema_direction == "positive"
        else "Edema-related variation was present and influenced the score trend."
        if edema_direction == "concern"
        else "Edema-related signal remained stable with mild variability."
    )
    edema_support = f"Edema / inflammation risk level: {levels.get('edema', 'Moderate')}."

    return [
        {"title": "Density impact", "direction": density_direction, "line": density_line, "support": density_support},
        {"title": "Structural impact", "direction": structure_direction, "line": structure_line, "support": structure_support},
        {"title": "Edema signal impact", "direction": edema_direction, "line": edema_line, "support": edema_support},
    ]

def get_risk_plain_language(key, level):
    level = (level or "Moderate").lower()
    mapping = {
        "structural": {
            "low": "Structural pattern: within expected range.",
            "moderate": "Structural pattern: mild concern.",
            "high": "Structural pattern: notable concern and should be reviewed.",
        },
        "raynaud": {
            "low": "Flow stability: within expected range.",
            "moderate": "Flow stability: some variability detected.",
            "high": "Flow stability: marked variability that may need follow-up.",
        },
        "edema": {
            "low": "Inflammatory signal: low.",
            "moderate": "Inflammatory signal: mild elevation.",
            "high": "Inflammatory signal: elevated and worth closer tracking.",
        },
    }
    return mapping.get(key, {}).get(level, "Pattern suggests moderate variation.")

def generate_user_summary(score, risk_profile, live_stats):
    band = get_health_band(score)["label"]
    levels = risk_profile.get("risk_levels", {})
    structural_level = levels.get("structural", "Moderate").lower()
    raynaud_level = levels.get("raynaud", "Moderate").lower()
    edema_level = levels.get("edema", "Moderate").lower()
    statement = get_personal_result_statement(score, risk_profile, live_stats)

    if band in ["Excellent", "Good"] and all(level != "high" for level in [structural_level, raynaud_level, edema_level]):
        lead = "Your current result suggests generally stable microcirculation."
    elif any(level == "high" for level in [structural_level, raynaud_level, edema_level]):
        lead = "Your current result indicates mixed stability with specific features that may reflect ongoing stress."
    else:
        lead = "Your current result appears moderately stable, with targeted areas to monitor."

    return (
        f"{lead} {statement} This analysis is intended for screening and research interpretation, "
        "and should be considered together with clinical context rather than treated as a standalone diagnosis."
    )

def generate_next_steps(risk_profile, final_density):
    levels = risk_profile.get("risk_levels", {})
    immediate_steps = ["Recheck one scan under consistent lighting, magnification, and finger temperature to improve comparability."]
    long_term_steps = ["Track trends across repeated scans rather than relying on a single reading."]

    if levels.get("structural", "").lower() in ["moderate", "high"]:
        immediate_steps.append("Flag structural irregular regions for focused clinician review on the next visit.")
        long_term_steps.append("Compare morphology composition over time to confirm whether irregular patterns persist or normalize.")
    if levels.get("raynaud", "").lower() in ["moderate", "high"]:
        immediate_steps.append("Document symptoms and exposure triggers (cold/stress) near the scan time to contextualize flow variability.")
        long_term_steps.append("Schedule follow-up captures under similar conditions to assess flow stability trends.")
    if levels.get("edema", "").lower() in ["moderate", "high"]:
        immediate_steps.append("Review edema-related regions with a clinician if swelling/inflammatory signs continue.")
        long_term_steps.append("Monitor edema index direction over serial scans to identify sustained increases.")
    if final_density < 5:
        immediate_steps.append("Repeat imaging with careful focus and contact pressure control to verify low-density signal.")

    return {"immediate": immediate_steps[:2], "long_term": long_term_steps[:2]}

def render_hero_result(health_score, percentile, risk_profile, live_stats):
    band = get_health_band(health_score)
    percentile_line, _ = get_percentile_text(percentile)
    profile_label = get_personal_result_label(health_score, risk_profile, live_stats)
    statement = get_personal_result_statement(health_score, risk_profile, live_stats)
    st.markdown(
        f"""
        <div class="hero-result-card">
            <div class="micro-label">Personal Result Overview</div>
            <div class="section-title">Your Health Score</div>
            <div class="hero-score">
                {int(round(health_score))} / 100
                <span class="grade-badge {band['badge_class']}">{band['label']}</span>
            </div>
            <div class="hero-percentile">{percentile_line}</div>
            <div class="caption-text">Compared with users in the available reference dataset.</div>
            <div class="result-label-chip">{profile_label}</div>
            <div class="hero-summary">{statement}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_percentile_card(percentile):
    percentile_line, pct = get_percentile_text(percentile)
    top_pct = max(0, 100 - pct)
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown("#### Relative Position")
    left, right = st.columns([2.3, 1])
    with left:
        st.markdown(f"### {percentile_line}")
        st.caption("Compared with users in the available reference dataset")
    with right:
        st.metric("Reference Rank", f"Top {top_pct}%")
    st.progress(min(max(pct, 0), 100))
    st.markdown('</div>', unsafe_allow_html=True)

def render_research_detail_sections(live_stats, health_score, risk_profile, auto_count, manual_count, final_density):
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown("#### Quantitative Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Health Score", f"{health_score}")
    m2.metric("Live Density", f"{final_density:.1f}/mm")
    m3.metric("Auto Count", f"{auto_count}")
    m4.metric("Physician Added", f"+{manual_count}")
    st.altair_chart(plot_health_score_bar(health_score), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown("#### Clinical Risk Indices")
    risk_defs = [
        ("Structural Damage", "structural"),
        ("Raynaud's Risk", "raynaud"),
        ("Edema / Inflammation", "edema"),
    ]
    rcols = st.columns(3)
    for col, (title, key) in zip(rcols, risk_defs):
        score = int(risk_profile['risks'][key])
        level = risk_profile['risk_levels'][key]
        with col:
            st.markdown('<div class="risk-card">', unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.markdown(f"<span class='risk-badge'>{level}</span>", unsafe_allow_html=True)
            st.markdown(f"### {score}/100")
            st.caption(get_risk_plain_language(key, level))
            st.caption(get_risk_interpretation(title, score))
            st.progress(min(max(score, 0), 100))
            st.markdown('</div>', unsafe_allow_html=True)
    st.info(f"Diagnostic Flag: {risk_profile['diagnostic_flag']}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown("#### Morphology Composition")
    st.altair_chart(plot_capillary_distribution(live_stats), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown("#### Validation Log")
    if not st.session_state.manual_regions:
        st.write("No manual validation records.")
    else:
        for i, r in enumerate(st.session_state.manual_regions):
            left, right = st.columns([6, 1])
            with left:
                st.markdown(
                    f"""
                    <div class="audit-row">
                        <span class="risk-badge">{r['type']}</span>
                        <span class="mono">({r['x']}, {r['y']})</span><br/>
                        <span class="caption-text">Source: manual / physician · Entry #{i+1:03d}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with right:
                if st.button("Delete", key=f"del_audit_{i}", use_container_width=True):
                    st.session_state.manual_regions.pop(i)
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_user_view(user, live_stats, health_score, percentile, risk_profile, auto_count, manual_count, final_density):
    st.markdown("### Personal Health Analysis")
    st.markdown('<div class="clinical-card user-soft-card">', unsafe_allow_html=True)
    st.markdown("#### Personal Result Label")
    st.markdown(f"**{get_personal_result_label(health_score, risk_profile, live_stats)}**")
    st.caption("A concise summary of your current microcirculation pattern in this scan.")
    st.markdown('</div>', unsafe_allow_html=True)

    render_hero_result(health_score, percentile, risk_profile, live_stats)
    render_percentile_card(percentile)

    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown("#### What influenced your score")
    drivers = get_score_drivers(health_score, final_density, live_stats, risk_profile)
    dcols = st.columns(3)
    for col, driver in zip(dcols, drivers):
        chip_label = {"positive": "Positive influence", "neutral": "Neutral influence", "concern": "Concern influence"}[driver["direction"]]
        chip_class = {"positive": "insight-positive", "neutral": "insight-neutral", "concern": "insight-concern"}[driver["direction"]]
        with col:
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown(f"<span class='insight-chip {chip_class}'>{chip_label}</span>", unsafe_allow_html=True)
            st.markdown(f"**{driver['title']}**")
            st.write(driver["line"])
            st.caption(driver["support"])
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown("#### What this means")
    band = get_health_band(health_score)["label"]
    summary = generate_user_summary(health_score, risk_profile, live_stats)
    st.markdown(f"**Current status:** {band}")
    st.write(summary)
    st.caption("Screening/research interpretation only; combine with clinical context.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown("#### Recommended next steps")
    next_steps = generate_next_steps(risk_profile, final_density)
    st.markdown("**Immediate next step**")
    for idx, step in enumerate(next_steps["immediate"], start=1):
        st.markdown(f"{idx}. {step}")
    st.markdown("**Longer-term tracking suggestion**")
    for idx, step in enumerate(next_steps["long_term"], start=1):
        st.markdown(f"{idx}. {step}")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Research details (secondary)", expanded=False):
        st.markdown('<div class="user-secondary">', unsafe_allow_html=True)
        render_research_detail_sections(live_stats, health_score, risk_profile, auto_count, manual_count, final_density)
        st.markdown('</div>', unsafe_allow_html=True)

def render_research_view(user, live_stats, health_score, percentile, risk_profile, auto_count, manual_count, final_density):
    ctx = get_selected_study_context()
    st.markdown("### Research View")
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown("#### Subject / Study Summary")
    render_subject_context_card(show_edit=True)
    sum_c1, sum_c2, sum_c3 = st.columns(3)
    sum_c1.markdown(f"**Analysis Status**  \n`Completed`")
    sum_c2.markdown(f"**Validated Count**  \n`{manual_count}`")
    sum_c3.markdown(f"**Review Timestamp**  \n`{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}`")
    st.caption(f"Study `{ctx['study_id']}` — analytical benchmark aligned to the reference cohort.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="clinical-card research-emphasis">', unsafe_allow_html=True)
    st.markdown("#### Personal result snapshot")
    render_hero_result(health_score, percentile, risk_profile, live_stats)
    st.markdown('</div>', unsafe_allow_html=True)
    render_research_detail_sections(live_stats, health_score, risk_profile, auto_count, manual_count, final_density)

def render_viewer_workspace():
    st.markdown(
        """
        <div class="viewer-panel">
            <div class="viewer-header">
                <div class="viewer-title">Imaging Workstation Viewer</div>
                <span class="risk-badge">Review Mode</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tabs = st.tabs(["Raw", "Overlay", "Validated"])
    with tabs[0]:
        st.image(st.session_state.processed_original, caption="Raw Input", use_container_width=True)

    with tabs[1]:
        st.image(st.session_state.base_overlay, caption="AI Overlay", use_container_width=True)

    with tabs[2]:
        display_img = st.session_state.base_overlay.copy()
        label_map = {"Normal": "N", "Abnormal": "A", "Hemorrhage": "H", "Aggregation": "Ag", "Blur": "B"}

        for region in st.session_state.manual_regions:
            region_color = UI_COLOR_MAP.get(region['type'], (255, 255, 255))
            if 'contour' in region:
                inference.draw_filled_highlight(display_img, region['contour'], color=region_color, alpha=0.5)
            else:
                cv2.circle(display_img, (region['x'], region['y']), 15, region_color, 2)

            label_short = label_map.get(region['type'], region['type'][0])
            cv2.putText(display_img, label_short, (region['x']-9, region['y']-19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
            cv2.putText(display_img, label_short, (region['x']-10, region['y']-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        pil_image = Image.fromarray(display_img)
        value = streamlit_image_coordinates(pil_image, key="click_interaction", use_column_width=True)

        if value is not None:
            x, y = value['x'], value['y']
            last_pt = st.session_state.last_clicked_coords
            is_new_click = True
            if last_pt and abs(last_pt[0] - x) < 10 and abs(last_pt[1] - y) < 10:
                is_new_click = False

            if is_new_click:
                st.session_state.last_clicked_coords = (x, y)
                interaction_mode = st.session_state.get('interaction_mode_state', "🪄 Add (Magic Wand)")

                if "Add" in interaction_mode:
                    annotate_popup(x, y)
                elif "Delete" in interaction_mode:
                    mask = st.session_state.cleaned_mask
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
                        class_name = inference.CLASSES.get(mask[y, x])
                        h, w = mask.shape
                        ff_mask1, ff_mask2 = np.zeros((h+2, w+2), np.uint8), np.zeros((h+2, w+2), np.uint8)
                        cv2.floodFill(mask, ff_mask1, (x, y), 0, flags=4)
                        cv2.floodFill(st.session_state.raw_mask, ff_mask2, (x, y), 0, flags=4)

                        if class_name in st.session_state.stats and st.session_state.stats[class_name] > 0:
                            st.session_state.stats[class_name] -= 1

                        st.session_state.base_overlay = inference.draw_result_on_image(st.session_state.processed_original, mask)
                        st.session_state.cleaned_mask = mask
                        st.rerun()
                    else:
                        st.toast("⚠️ No AI prediction found at this location.", icon="⚠️")

    st.markdown('<div class="workstation-strip">', unsafe_allow_html=True)
    st.markdown('<div class="micro-label">Workstation Control Strip</div>', unsafe_allow_html=True)
    st.radio("Interaction Tool", ["🪄 Add (Magic Wand)", "🗑️ Delete (Remove Prediction)"], horizontal=True, key="interaction_mode_state", label_visibility="collapsed")
    min_area = st.slider("Minimum Area Threshold (px)", 0, 1000, st.session_state.get('current_min_area', 100), 10)
    if min_area != st.session_state.get('current_min_area', 100):
        st.session_state.current_min_area = min_area
        new_stats, new_cleaned_mask, new_overlay = inference.recalculate_overlay(st.session_state.processed_original, st.session_state.raw_mask, min_area)
        st.session_state.stats, st.session_state.base_overlay, st.session_state.cleaned_mask = new_stats.copy(), new_overlay, new_cleaned_mask
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ================= 2. DATA INPUT DIALOG =================
@st.dialog("🧪 Subject & Clinical Metadata")
def get_patient_info():
    st.write("Complete subject metadata to establish standardized research context.")
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Subject ID", value="Subj-001")
            age = st.number_input("Age", 18, 100, 30)
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            fov = st.number_input("Field of View Width (mm)", 1.0, 5.0, 3.0)
           
        st.markdown("---")
        if st.form_submit_button("✅ Initialize Research Pipeline"):
            st.session_state.user_data = {"name": name, "age": age, "gender": gender, "fov": fov}
            st.session_state.run_analysis = False
            st.session_state.inference_done = False
            st.session_state.manual_regions = []
            st.session_state.open_patient_dialog = False
            st.session_state.analysis_requested = True
            st.rerun()

@st.dialog("Define Morphology")
def annotate_popup(x, y):
    current_img = st.session_state.processed_original
   
    if 'temp_contour' not in st.session_state:
        with st.spinner("Extracting Contour..."):
            res = inference.get_magic_wand_region(current_img, x, y)
            if res is None:
                st.error("⚠️ Detection failed: Please try clicking on the center of the blood vessel.")
                if st.button("Close"):
                    st.rerun()
                return 
            st.session_state.temp_contour = res
   
    st.markdown(f"**Target Location:** ({x}, {y})")
    st.info("✨ Magic Wand Extraction: **Active**")
   
    new_type = st.radio(
        "Select Pathology:",
        ["Normal", "Abnormal", "Hemorrhage", "Aggregation", "Blur"],
        index=0,
        horizontal=False
    )
   
    col_submit, col_cancel = st.columns([1, 1])
    with col_submit:
        if st.button("Confirm & Add", type="primary", width="stretch"):
            st.session_state.manual_regions.append({
                "x": x, "y": y,
                "type": new_type,
                "contour": st.session_state.temp_contour
            })
            del st.session_state.temp_contour
            st.rerun()
           
    with col_cancel:
        if st.button("Cancel", width="stretch"):
            if 'temp_contour' in st.session_state: del st.session_state.temp_contour
            st.rerun()

# ================= MAIN UI =================
#with st.sidebar:
#    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083206.png", width=60)
#    st.title("Research Config")
#    api_key = st.text_input("Gemini API Key", type="password")
#    
#    st.markdown("### 📚 Core References")
#    st.info("""
#    **1. Density Norms:**
#    *Etehad Tavakol et al. (2015)*
#    BioMed Research International
#    
#    **2. Pattern Classification:**
#    *Smith et al. (2023)* / *Cutolo*
#    EULAR Study Group
#    """)

render_global_header()
render_workflow_strip()
# ================= MLOPS: PRE-LOADING & WARM-UP =================
# 使用 session_state 作為絕對鎖，防止 Streamlit 瘋狂重複執行
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    with st.spinner("Initializing analysis pipeline..."):
        # 這裡會觸發 inference.py 裡面的 get_predictor
        _ = inference.get_predictor()
        st.session_state.model_loaded = True
        st.success("Analysis ready.")

# ================= PACS-LIKE DUAL-TRACK IMAGE LOADING =================
st.markdown("""
<style>
    .med-card-container { transition: background 0.2s ease, border-color 0.2s ease; border-radius: 8px; }
    .med-card-container:hover { background: #f7f3eb; }
    .med-card-id { color: #173B45; font-weight: 700; font-size: 1em; margin-bottom: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .med-card-date { color: #5F7C82; font-size: 0.82em; font-weight: 400; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="layer-wrap"><div class="section-title">Main Workspace</div>', unsafe_allow_html=True)

uploaded_file = None
show_gallery = not st.session_state.confirmed_file

if show_gallery:
    if st.session_state.is_previewing:
        render_confirm_input()
    else:
        render_study_browser()

# === File Consolidation & Auto-Trigger Dialog ===
if st.session_state.confirmed_file:
    if st.session_state.get('upload_source') == "device" and st.session_state.get('selected_local_file'):
        fpath = st.session_state.selected_local_file
        if os.path.exists(fpath):
            with open(fpath, "rb") as f: file_bytes = f.read()
            uploaded_file = io.BytesIO(file_bytes)
            uploaded_file.name = os.path.basename(fpath)
            uploaded_file.size = len(file_bytes)
            
    elif st.session_state.get('upload_source') == "manual" and st.session_state.get('manual_uploaded_file'):
        uploaded_file = st.session_state.manual_uploaded_file

# ================= SINGLE DIALOG RENDER POINT =================
if st.session_state.get('open_patient_dialog', False):
    get_patient_info()
# ====================================================================
# 佈局與分析核心邏輯 (嚴謹狀態機隔離)
# ====================================================================
if uploaded_file is not None and st.session_state.get('confirmed_file', False):
    
    # ---------------- 狀態一：尚未完成分析 (置中顯示確認介面) ----------------
    if not st.session_state.get('inference_done', False):
        st.markdown("### Confirm Input")
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.image(uploaded_file, caption="Selected Input", use_container_width=True)
        with right:
            source = st.session_state.get('upload_source', 'N/A')
            capture_time = "Session Upload"
            study_name = getattr(uploaded_file, "name", "N/A")
            if source == "device" and st.session_state.get("selected_local_file"):
                fpath = st.session_state.selected_local_file
                study_name = os.path.basename(fpath)
                capture_time = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M")
            st.markdown('<div class="study-summary-card">', unsafe_allow_html=True)
            st.markdown("#### Study Summary")
            st.markdown(f"**Study/File ID**: `{study_name}`")
            st.markdown(f"**Source**: {source.capitalize()}")
            st.markdown(f"**Capture Time**: {capture_time}")
            st.markdown("**Status**: Ready")
            st.markdown('</div>', unsafe_allow_html=True)

        col_btn_back, _, col_btn_exec = st.columns([1.5, 0.2, 2])
        with col_btn_back:
            if st.button("Back to Gallery", use_container_width=True):
                st.session_state.confirmed_file = False
                st.session_state.selected_local_file = None
                st.session_state.manual_uploaded_file = None
                st.session_state.open_patient_dialog = False
                st.session_state.analysis_requested = False
                st.rerun()
        with col_btn_exec:
            if st.button("Continue to Metadata", type="primary", use_container_width=True):
                if not api_key: 
                    st.error("Error: API Key Missing.")
                else:
                    st.session_state.analysis_requested = True
                    if not st.session_state.get('user_data'):
                        st.session_state.open_patient_dialog = True
                    st.session_state.run_analysis = False
                    st.session_state.inference_done = False
                    st.rerun()

        if st.session_state.get('analysis_requested', False):
            render_subject_context_card(uploaded_file=uploaded_file, show_edit=True)
            if not st.session_state.get('user_data'):
                st.warning("Metadata review required before analysis can proceed.")
                if st.button("Enter Metadata", type="primary", use_container_width=True):
                    st.session_state.open_patient_dialog = True
                    st.rerun()
            else:
                render_stage4_readiness()

        # 背景推論邏輯 (由 Stage 4 Run Analysis 觸發)
        if st.session_state.run_analysis and st.session_state.user_data and not st.session_state.inference_done:
            with st.spinner("Running segmentation model..."):
                uploaded_file.seek(0)
                original, overlay, stats, raw_mask = inference.process_image(uploaded_file)
                with st.spinner("Computing morphology metrics..."):
                    pass
                
                FIXED_WIDTH = 800
                st.session_state.processed_original = inference.resize_with_aspect_ratio(original, width=FIXED_WIDTH)
                st.session_state.base_overlay = inference.resize_with_aspect_ratio(overlay, width=FIXED_WIDTH)
                st.session_state.raw_mask = inference.resize_with_aspect_ratio(raw_mask, width=FIXED_WIDTH, inter=cv2.INTER_NEAREST)
                st.session_state.stats = stats.copy()
                st.session_state.current_min_area = 100 
                _, st.session_state.cleaned_mask = inference.simplified_post_processing(st.session_state.raw_mask, min_area=100)
                st.session_state.inference_done = True
                st.success("Analysis ready.")
                st.rerun()

    # ---------------- 狀態二：推論完成 (顯示 4:6 控制台與數據面版) ----------------
    else:
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = "User View"

        st.radio(
            "Display Mode",
            ["User View", "Research View"],
            key="view_mode",
            horizontal=True,
            help="User View emphasizes personal insights. Research View exposes full technical depth."
        )

        col_left, col_right = st.columns([4, 6], gap="large")

        # 🔴 左半部：影像工作站
        with col_left:
            render_viewer_workspace()

        # 🟢 右半部：Clinical dashboard
        with col_right:
            user = st.session_state.user_data
            
            live_stats = st.session_state.stats.copy()
            auto_count = sum(live_stats.values())
            manual_count = len(st.session_state.manual_regions)
            live_stats['Physician_Added'] = manual_count
            
            key_map = {"Normal": "Normal", "Abnormal": "Abnormal", "Hemorrhage": "Hemo", "Aggregation": "Aggregation", "Blur": "Blur"}
            for region in st.session_state.manual_regions:
                mapped_key = key_map.get(region['type'], "Normal")
                live_stats[mapped_key] = live_stats.get(mapped_key, 0) + 1

            final_total = auto_count + manual_count
            final_density = final_total / user['fov'] if user['fov'] > 0 else 0
            
            db_scores = load_reference_database()
            analyzer = ClinicalHealthAnalyzer()
            analysis_result = analyzer.analyze(raw_stats=live_stats, fov=user['fov'], reference_scores=db_scores)
            health_score, percentile, subscores = analysis_result["mhi_score"], analysis_result["percentile_rank"], analysis_result["subscores"]

            profiler = ClinicalRiskProfiler(stats=live_stats, fov=user['fov'])
            risk_profile = profiler.analyze()
            if st.session_state.view_mode == "User View":
                render_user_view(
                    user=user,
                    live_stats=live_stats,
                    health_score=health_score,
                    percentile=percentile,
                    risk_profile=risk_profile,
                    auto_count=auto_count,
                    manual_count=manual_count,
                    final_density=final_density
                )
            else:
                render_research_view(
                    user=user,
                    live_stats=live_stats,
                    health_score=health_score,
                    percentile=percentile,
                    risk_profile=risk_profile,
                    auto_count=auto_count,
                    manual_count=manual_count,
                    final_density=final_density
                )

    # ---------------- 狀態三：生成專業報告 ----------------
    if st.session_state.get('inference_done', False):
        st.markdown('<div class="final-output-zone"><div class="section-title">Final Clinical Report Zone</div>', unsafe_allow_html=True)
        if 'report_generated' not in st.session_state:
            st.session_state.report_generated = False
        if 'report_content' not in st.session_state:
            st.session_state.report_content = None
        if 'pdf_bytes' not in st.session_state:
            st.session_state.pdf_bytes = None
        if 'report_generated_at' not in st.session_state:
            st.session_state.report_generated_at = None

        st.markdown(
            f"""
            <div class="finalization-card">
                <div class="micro-label">Stage 6 · Finalization</div>
                <div class="panel-title">Prepare validated report for formal delivery</div>
                <div class="body-text">Complete finalization and generate the final clinical narrative package.</div>
                <div style="margin-top:10px;">
                    <span class="badge-muted">Subject ID: {user['name']}</span>
                    <span class="badge-muted">Age/Gender: {user['age']} / {user['gender']}</span>
                    <span class="badge-muted">FOV: {user['fov']} mm</span>
                    <span class="badge-muted">Analysis Date: {date.today().strftime("%Y-%m-%d")}</span>
                </div>
                <div style="margin-top:8px;">
                    <span class="badge-muted">Manual Validation Count: {manual_count}</span>
                    <span class="badge-muted">Health Score: {health_score}</span>
                    <span class="badge-muted">Density: {final_density:.2f} loops/mm</span>
                    <span class="badge-muted">Diagnostic Flag: {risk_profile['diagnostic_flag']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Generate Final Clinical Report", type="primary", use_container_width=True):
            st.session_state.report_generated = True
            st.session_state.report_content = None
            st.session_state.pdf_bytes = None
            st.session_state.report_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if st.session_state.report_generated:
            if st.session_state.report_content is None:
                final_overlay_for_report = st.session_state.base_overlay.copy()
                for region in st.session_state.manual_regions:
                    rc = UI_COLOR_MAP.get(region['type'], (255, 255, 255))
                    if 'contour' in region: 
                        inference.draw_filled_highlight(final_overlay_for_report, region['contour'], color=rc, alpha=0.5)
                    else:
                        cv2.circle(final_overlay_for_report, (region['x'], region['y']), 15, rc, 2)
               
                raw_pil = Image.fromarray(st.session_state.processed_original)
                overlay_pil = Image.fromarray(final_overlay_for_report)
                today_str = date.today().strftime("%Y-%m-%d")

                prompt = f"""
                Act as an expert Clinical Pathologist. Write a quantitative, professional medical analysis report based on the provided data.
                
                CRITICAL INSTRUCTIONS: 
                - DO NOT use any terminology related to AI. Completely filter out phrases like "AI Generating", "As an AI", "AI analysis", or "Generated by AI". 
                - The tone must be strictly clinical, objective, and indistinguishable from a standard human-written hospital laboratory report.
               
                **Validation Methodology:**
                - Protocol: **Human-in-the-Loop (HITL)**.
                - Visual Key: **Physician-validated regions are highlighted in contour color coding.**
                - Interventions: {manual_count} additional capillaries validated by physician.
               
                **Subject Data:**
                - ID: {user['name']} ({user['age']}y, {user['gender']})
                - Date: {today_str}
                - Field of View: {user['fov']} mm
               
                **Computed Diagnostics (Refined Post-Validation):**
                - Overall Health Score: {health_score}/100
                - Linear Density: {final_density:.2f} loops/mm
                - Absolute Morphological Counts: {live_stats}
                
                **Clinical Risk Profile & Diagnostic Flag:**
                - Structural Damage Risk: {risk_profile['risks']['structural']}/100 ({risk_profile['risk_levels']['structural']})
                - Raynaud's Risk: {risk_profile['risks']['raynaud']}/100 ({risk_profile['risk_levels']['raynaud']})
                - Edema & Inflammation Risk: {risk_profile['risks']['edema']}/100 ({risk_profile['risk_levels']['edema']})
                - System Generated Flag: "{risk_profile['diagnostic_flag']}"
               
                **Instructions:**
                1. **Executive Summary**: State the patient's Health Score prominently.
                2. **Quantitative Breakdown**: Compare the density ({final_density:.2f}) with the healthy norm (9±2 loops/mm) (Cite: Etehad Tavakol et al., 2015). Emphasize the HITL protocol.
                3. **Visual & Pathological Assessment**: Describe the structural features observed. Integrally discuss the three risk indices (Structural, Raynaud's, Edema) and what they imply pathophysiologically.
                4. **Conclusion**: Provide a standard medical summary. Incorporate the "System Generated Flag" narrative logically into your closing clinical advice.
                """
               
                try:
                    with st.spinner("Synthesizing Validated Clinical Report..."):
                        if USE_NEW_GENAI:
                            client = genai.Client(api_key=api_key)
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[prompt, raw_pil, overlay_pil]
                            )
                        else:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel('gemini-2.5-flash-lite')
                            response = model.generate_content([prompt, raw_pil, overlay_pil])

                        st.session_state.report_content = response.text
                        user['date'] = today_str
                        st.session_state.pdf_bytes = report_generator.create_pdf(
                            user_data=user, stats=live_stats, health_score=health_score,  
                            density=final_density, ai_text=response.text, overlay_image_rgb=final_overlay_for_report
                        )
                        st.success("Report successfully generated.")
                except Exception as e:
                    st.error(f"Generation Error: {e}")
                    import traceback
                    st.text(traceback.format_exc())

            if st.session_state.report_content:
                generated_at = st.session_state.get('report_generated_at', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                st.markdown(
                    f"""
                    <div class="finalization-card">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;">
                            <div>
                                <div class="micro-label">Final Deliverable</div>
                                <div class="section-title" style="margin-bottom:4px;">Final Clinical Report</div>
                                <div class="caption-text">Validated clinical narrative and PDF package for formal handoff.</div>
                            </div>
                            <div class="report-status-badge">Validated</div>
                        </div>
                        <div class="report-header-grid">
                            <div class="report-header-cell">
                                <div class="micro-label">Subject Summary</div>
                                <div class="caption-text"><b>ID:</b> {user['name']}</div>
                                <div class="caption-text"><b>Age/Gender:</b> {user['age']} / {user['gender']}</div>
                                <div class="caption-text"><b>FOV:</b> {user['fov']} mm</div>
                            </div>
                            <div class="report-header-cell">
                                <div class="micro-label">Study Summary</div>
                                <div class="caption-text"><b>Analysis Date:</b> {date.today().strftime("%Y-%m-%d")}</div>
                                <div class="caption-text"><b>Manual Validation:</b> {manual_count}</div>
                                <div class="caption-text"><b>Diagnostic Flag:</b> {risk_profile['diagnostic_flag']}</div>
                            </div>
                            <div class="report-header-cell">
                                <div class="micro-label">Report Status</div>
                                <div class="caption-text"><b>Status:</b> Generated / Ready for Download</div>
                                <div class="caption-text"><b>Timestamp:</b> {generated_at}</div>
                                <div class="caption-text"><b>Health Score:</b> {health_score} · <b>Density:</b> {final_density:.2f}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div class="report-narrative-shell">
                        <div class="panel-title">Clinical Narrative</div>
                        <div class="report-narrative-body">{st.session_state.report_content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)
                display_reference_table()

                st.markdown("---")
                col_dl, col_reset = st.columns([1.6, 1])
                with col_dl:
                    st.download_button(
                        label="Download Validated PDF Report", 
                        data=st.session_state.pdf_bytes,
                        file_name=f"Report_{user['name']}_Validated.pdf", 
                        mime="application/pdf", 
                        type="primary",
                        use_container_width=True
                    )
                with col_reset:
                    if st.button("Start New Analysis", use_container_width=True):
                        # 徹底重置狀態
                        st.session_state.run_analysis = False
                        st.session_state.inference_done = False
                        st.session_state.manual_regions = []
                        st.session_state.user_data = None
                        st.session_state.report_generated = False
                        st.session_state.report_content = None
                        st.session_state.pdf_bytes = None
                        st.session_state.report_generated_at = None
                        st.session_state.confirmed_file = False
                        st.session_state.selected_local_file = None
                        st.session_state.manual_uploaded_file = None
                        st.session_state.open_patient_dialog = False
                        st.session_state.analysis_requested = False
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
