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
        padding: 8px;
    }

    [data-testid="stMetricLabel"] { font-size: 0.84rem !important; font-weight: 600 !important; color: var(--color-secondary) !important; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; color: var(--color-primary) !important; }

    div[data-testid="stDialog"] div[role="dialog"] {
        width: 480px;
        border-radius: 12px;
        border: 1px solid var(--color-border);
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
        return 5
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
    for i, name in enumerate(stages, start=1):
        if i < current:
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
        render_subject_context_card(uploaded_file=uploaded_file, show_edit=True)
        col_left, col_right = st.columns([4, 6], gap="large")

        # 🔴 左半部：視覺與互動控制
        with col_left:
            st.markdown('<div class="card-header">📷 Microscopic View</div>', unsafe_allow_html=True)
            with st.container(border=True):
                show_raw = st.toggle("👁️ Show Raw Input", value=False)
                if show_raw:
                    st.image(st.session_state.processed_original, caption="Raw Input", use_container_width=True)
                else:
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

            st.markdown('<div class="card-header">🛠️ Interactive Tools</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.radio("🔧 Interaction Mode:", ["🪄 Add (Magic Wand)", "🗑️ Delete (Remove Prediction)"], horizontal=True, key="interaction_mode_state")
                min_area = st.slider("Minimum Area Threshold (px)", 0, 1000, st.session_state.get('current_min_area', 100), 10)
                
                if min_area != st.session_state.get('current_min_area', 100):
                    st.session_state.current_min_area = min_area
                    new_stats, new_cleaned_mask, new_overlay = inference.recalculate_overlay(st.session_state.processed_original, st.session_state.raw_mask, min_area)
                    st.session_state.stats, st.session_state.base_overlay, st.session_state.cleaned_mask = new_stats.copy(), new_overlay, new_cleaned_mask
                    st.rerun()

        # 🟢 右半部：數據與風險分析
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

            st.markdown('<div class="card-header">📊 Live Clinical Metrics</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"<span style='color: #666;'>Demographic Standings: Your score beats **{percentile}%** of the regional demographic.</span>", unsafe_allow_html=True)
                st.altair_chart(plot_health_score_bar(health_score), use_container_width=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Live Density", f"{final_density:.1f}/mm")
                m2.metric("Auto Count", f"{auto_count}")
                m3.metric("Physician Added", f"+{manual_count}")

            st.write("") 
            st.markdown('<div class="card-header">⚠️ Clinical Risk Indices</div>', unsafe_allow_html=True)
            with st.container(border=True):
                r1, r2, r3 = st.columns(3)
                r1.metric("Structural Damage", f"{risk_profile['risks']['structural']}", risk_profile['risk_levels']['structural'], delta_color="inverse")
                r2.metric("Raynaud's Risk", f"{risk_profile['risks']['raynaud']}", risk_profile['risk_levels']['raynaud'], delta_color="inverse")
                r3.metric("Edema / Inflam.", f"{risk_profile['risks']['edema']}", risk_profile['risk_levels']['edema'], delta_color="inverse")
                st.info(f"**Diagnostic Flag:** {risk_profile['diagnostic_flag']}")

            st.write("") 
            c_chart, c_valid = st.columns([1.5, 1], gap="medium")
            
            with c_chart:
                st.markdown('<div class="card-header">🔬 Composition</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.altair_chart(plot_capillary_distribution(live_stats), use_container_width=True)
            
            with c_valid:
                st.markdown('<div class="card-header">✅ Validated Regions</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    if not st.session_state.manual_regions:
                        st.write("No manual validations.")
                    else:
                        for i, r in enumerate(st.session_state.manual_regions):
                            col_t, col_d = st.columns([4, 1])
                            col_t.markdown(f"**{r['type']}** `({r['x']}, {r['y']})`")
                            if col_d.button("🗑️", key=f"del_list_{i}", use_container_width=True):
                                st.session_state.manual_regions.pop(i)
                                st.rerun()

    # ---------------- 狀態三：生成專業報告 ----------------
    if st.session_state.get('inference_done', False):
        st.markdown('<div class="layer-wrap"><div class="section-title">Final Output Zone</div>', unsafe_allow_html=True)
        if 'report_generated' not in st.session_state:
            st.session_state.report_generated = False
        if 'report_content' not in st.session_state:
            st.session_state.report_content = None
        if 'pdf_bytes' not in st.session_state:
            st.session_state.pdf_bytes = None

        if st.button("✅ Generate Validated Report", type="primary", use_container_width=True):
            st.session_state.report_generated = True
            st.session_state.report_content = None
            st.session_state.pdf_bytes = None

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
               
                st.markdown("---")
                st.markdown('<div class="section-title">📑 Clinical Pathology Report</div>', unsafe_allow_html=True)
               
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
                except Exception as e:
                    st.error(f"Generation Error: {e}")
                    import traceback
                    st.text(traceback.format_exc())

            if st.session_state.report_content:
                st.markdown(f'<div class="paper-container">{st.session_state.report_content}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                display_reference_table()

                st.markdown("---")
                col_dl, col_reset = st.columns([1, 1])
                with col_dl:
                    st.download_button(
                        label="📄 Download Validated Professional Report (PDF)", 
                        data=st.session_state.pdf_bytes,
                        file_name=f"Report_{user['name']}_Validated.pdf", 
                        mime="application/pdf", 
                        use_container_width=True
                    )
                with col_reset:
                    if st.button("🔄 Analyze Next Subject", use_container_width=True):
                        # 徹底重置狀態
                        st.session_state.run_analysis = False
                        st.session_state.inference_done = False
                        st.session_state.manual_regions = []
                        st.session_state.user_data = None
                        st.session_state.report_generated = False
                        st.session_state.report_content = None
                        st.session_state.pdf_bytes = None
                        st.session_state.confirmed_file = False
                        st.session_state.selected_local_file = None
                        st.session_state.manual_uploaded_file = None
                        st.session_state.open_patient_dialog = False
                        st.session_state.analysis_requested = False
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
