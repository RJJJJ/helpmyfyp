# HelpMyFYP

A quantitative **Nailfold Capillaroscopy** research platform that combines deep learning segmentation, clinically aligned scoring, human-in-the-loop validation, and automated report generation for microvascular assessment.

Repository: https://github.com/RJJJJ/helpmyfyp

---

## 1. Project Overview

**HelpMyFYP** is a Final Year Project focused on building an end-to-end platform for analyzing **nailfold capillaroscopy images**.

The system is designed to support a workflow that goes beyond simple image classification. It aims to:

- load nailfold microscopy images from a connected image source or manual upload
- run deep learning-based segmentation on capillary structures
- classify important pathological morphology
- calculate a clinically interpretable quantitative health score
- estimate several clinical risk indices
- support physician-style manual correction and validation
- generate a professional PDF report

This project is therefore not only an image-analysis demo, but a prototype of a **clinically interpretable microcirculation analysis system**.

---

## 2. Research Motivation

Traditional capillaroscopy interpretation depends heavily on visual expertise and manual review. That creates several practical limitations:

- evaluation can be time-consuming
- quantitative comparison is not always standardized
- morphological findings may be difficult to summarize consistently
- research workflows often lack an integrated digital pipeline

This project attempts to solve that by combining:

1. **deep learning segmentation** for image understanding
2. **clinically aligned rules** for interpretable scoring
3. **human-in-the-loop correction** for practical validation
4. **automated reporting** for downstream clinical and research use

---

## 3. Core Objective

The main objective of this project is to create a platform that can transform raw nailfold microscopy images into a structured clinical-style analysis workflow.

More specifically, the system aims to:

- detect and segment capillary morphology from microscopy images
- map segmented findings to clinically meaningful categories
- quantify microvascular condition using a mathematically explainable score
- support expert correction instead of relying on a pure black-box pipeline
- provide a reproducible and extensible foundation for future medical AI research

---

## 4. What the System Does

Based on the current implementation, the system supports the following pipeline:

### Image Input
- Load images from the local acquisition folder under `machine/`
- Upload images manually through the interface

### Deep Learning Inference
- Preload an nnU-Net predictor for segmentation
- Run inference fully in memory
- Produce a class mask and visual overlay

### Morphology Analysis
The system currently works with five morphology-related output classes:

- **Normal**
- **Abnormal**
- **Hemorrhage / Hemo**
- **Aggregation**
- **Blur**

### Human-in-the-Loop Validation
- Add physician-validated regions manually
- Remove incorrect predicted regions
- Use contour-based region extraction for annotation assistance

### Quantitative Assessment
- compute capillary density
- compute overall health score
- compute percentile standing
- compute risk indices for structural damage, Raynaud's phenomenon, and edema/inflammation

### Report Generation
- generate a professional analysis narrative
- export a PDF report for the current subject

---

## 5. Clinical and Research Positioning

This project is important because it does **not** treat the model as an isolated AI detector.

Instead, it tries to connect image segmentation outputs with established clinical concepts in microcirculation analysis. The repository already includes supporting documentation such as:

- `Clinical_Alignment_Proof.md`
- `health_score.md`
- `Microvascular.md`

These documents indicate that the project is structured around:

- capillary density criteria
- morphology-based disease relevance
- explainable health scoring
- evidence-aligned quantitative interpretation

This makes the project stronger as a Final Year Project because it combines:

- software implementation
- medical image processing
- explainable scoring logic
- research justification
- practical reporting workflow

---

## 6. System Architecture

At a high level, the project can be understood as the following pipeline:

```text
Microscopy Image
   ↓
Input Loader (device folder or manual upload)
   ↓
Deep Learning Segmentation (nnU-Net)
   ↓
Post-processing and Morphology Counting
   ↓
Clinical Scoring and Risk Analysis
   ↓
Human Validation / Correction
   ↓
Professional Report Generation (PDF)
```

---

## 7. Main Application

The main entry point is:

- `app.py`

This Streamlit application provides:

- image gallery / upload workflow
- subject metadata form
- inference trigger
- visual inspection panel
- manual validation tools
- real-time metrics dashboard
- report generation and download

---

## 8. Important Files

### Application Layer
- `app.py` — main Streamlit application and interface logic
- `report_generator.py` — PDF report generation

### Inference Layer
- `inference.py` — model loading, nnU-Net inference, post-processing, overlay generation
- `capillary_analysis.py` — analysis-related logic
- `clinical_health_analyzer.py` — health score computation and interpretation
- `clinical_risk.py` — clinical risk profiling

### Research / Documentation Layer
- `Clinical_Alignment_Proof.md` — explanation of clinical alignment and scoring logic
- `health_score.md` — mathematical justification of the Microvascular Health Index
- `Microvascular.md` — related documentation

### Data / Utility Layer
- `reference_scores.json` — reference score database
- `build_reference_db.py` — reference database preparation
- `paper_extractor.py` — literature / extraction-related script

### Device / Acquisition Layer
- `machine/` — device-related resources used by the project

---

## 9. About the `machine/` Folder

The `machine/` folder is part of the project’s acquisition-side workflow.

From the current application logic, the system expects local device images to be available through a folder path under `machine/Guests-Image/Guest`.

That means the `machine/` directory is not just an extra asset folder. It is conceptually tied to the real image-source side of the project and is relevant to the overall research pipeline.

In other words, this repository contains both:

- the **analysis platform**
- and the **image input / acquisition-side integration path**

This is useful for demonstrating that the FYP is not only a static software prototype, but a more complete workflow-oriented system.

---

## 10. Quantitative Scoring

A major contribution of this project is the use of an explainable quantitative score rather than an opaque model output.

The repository documents a **Microvascular Health Index (MHI)** that maps capillary density and pathological morphology into a continuous health score.

This is significant because it gives the project a stronger research foundation:

- the system is not only detecting structures
- it is converting findings into a clinically interpretable metric
- the score can be explained mathematically
- the logic can be audited and improved in future work

---

## 11. Human-in-the-Loop Design

One of the strongest aspects of this project is the inclusion of human validation.

Instead of assuming that automatic predictions are always correct, the system explicitly supports:

- manual addition of validated capillaries
- manual removal of incorrect detections
- interactive correction during the visual review stage

This makes the system much more realistic as a clinical research prototype.

---

## 12. Technical Stack

Based on the current repository, the project uses a Python-based stack centered around medical image analysis and interactive deployment.

### Core Frameworks and Libraries
- **Python**
- **Streamlit** for the web interface
- **PyTorch** for deep learning execution
- **nnU-Net v2** for medical image segmentation
- **OpenCV** and **scikit-image** for image processing
- **NumPy / Pandas** for data handling
- **Altair / Matplotlib** for visualization
- **ReportLab / FPDF** for PDF export
- **Google Generative AI SDK** for professional report text generation

---

## 13. Current Features

The repository currently demonstrates the following implemented or partially implemented capabilities:

- image gallery loading from device-side folder
- manual image upload
- model preloading and warm-up
- semantic segmentation inference
- class-based morphology statistics
- adjustable post-processing thresholding
- interactive region annotation and removal
- live quantitative metric display
- health score computation
- clinical risk profiling
- generated narrative report
- PDF export

---

## 14. Planned / Natural Next Steps

This repository is already functionally interesting, but there are several natural directions for improvement.

### UI / UX
- cleaner layout hierarchy
- more consistent styling
- better spacing and panel organization
- stronger visual emphasis on workflow stages
- improved report presentation

### Engineering
- clearer module boundaries
- stronger configuration management
- environment setup documentation
- model path validation and deployment notes
- better error handling and logging

### Research / Productization
- clearer evaluation metrics
- sample data and demo workflow documentation
- comparison with manual-only review
- richer database / subject history support
- more structured clinician feedback workflow

---

## 15. Installation

Clone the repository:

```bash
git clone https://github.com/RJJJJ/helpmyfyp.git
cd helpmyfyp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 16. Environment Setup

The application currently expects an API key for report generation.

Create a `.env` file in the project root and add:

```env
API_KEY=your_api_key_here
```

This key is used by the report generation workflow in `app.py`.

---

## 17. Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 18. Model Requirements

The inference pipeline expects a trained nnU-Net model folder under:

```text
models/nnunet_anfc
```

Important notes:

- the required checkpoint is expected to be `checkpoint_best.pth`
- the system automatically initializes nnU-Net-related environment folders if needed
- inference can run on GPU or CPU, with device-aware loading logic implemented in `inference.py`

---

## 19. Suggested Repository Improvements for Codex

This repository is a good candidate for AI-assisted refinement.

Codex or other code agents should focus on:

1. auditing the current structure and removing duplication
2. improving UI hierarchy and responsiveness
3. tightening module naming and consistency
4. documenting the `machine/` workflow more clearly
5. adding setup instructions for models and sample images
6. improving README diagrams and screenshots
7. separating research documentation from runtime code more cleanly

---

## 20. Why This README Matters

This README is intentionally detailed because the project is more than a small script.

It combines:

- research context
- model inference
- clinical scoring
- validation workflow
- reporting output

A concise README would under-explain the value of the repository. A detailed README helps:

- teachers and evaluators understand the FYP scope
- collaborators understand the system structure
- future AI coding agents improve the project more effectively

---

## 21. Limitations

At the current stage, several areas may still require refinement:

- setup complexity for models and environment
- incomplete deployment instructions
- lack of public sample data
- UI still prototype-oriented
- clinical validation still requires further evidence and formal evaluation

These are normal limitations for an academic prototype and also define the next stage of improvement.

---

## 22. Author / Project Status

This repository is part of a Final Year Project and is currently under active development and refinement.

Author: **RJ**  
University of Macau  
GitHub: https://github.com/RJJJJ/helpmyfyp

---

## 23. License

No license has been specified yet.

If you intend this project to be reusable by others, consider adding a license such as MIT.
