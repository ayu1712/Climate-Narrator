# ClimateNarrator

**Expert-Authored Storm Narratives Improve Assessment of Extreme Weather Damage**

ClimateNarrator is a dual-stream multimodal model that fuses structured meteorological
features from the NOAA Storm Events Database with expert-authored National Weather
Service (NWS) event narratives to predict economic damage from extreme weather events.
It outperforms all structured-only baselines with a statistically significant
improvement of 46.5% in R² over XGBoost (p < 0.001).

> Associated paper: *Expert-Authored Storm Narratives Improve Assessment of Extreme
> Weather Damage* — submitted to Communications Earth & Environment.

---

## Table of Contents

- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Running the Notebook](#running-the-notebook)
- [Pre-trained Model](#pre-trained-model)
- [Data](#data)
- [Citation](#citation)

---

## Results

Performance on the temporally held-out test set (2022–2024, n = 41,027 events):

| Model | R² | MAE (log) | MAPE (%) | Spearman ρ |
|---|---|---|---|---|
| Random Forest | 0.264 | 1.478 | 18.94 | 0.474 |
| XGBoost | 0.317 | 1.418 | 18.25 | 0.518 |
| FT-Transformer | 0.206 | 1.477 | 17.71 | 0.330 |
| TabNet | 0.298 | 1.420 | 17.88 | 0.473 |
| Text-only (TF-IDF + Ridge) | 0.394 | 1.268 | 15.80 | 0.601 |
| Concat (TF-IDF + Ridge) | 0.405 | 1.259 | 15.74 | 0.607 |
| **ClimateNarrator (Fusion)** | **0.465** | **1.120** | **13.60** | **0.676** |

All improvements over XGBoost are statistically significant
(paired bootstrap, p < 0.001, n = 1,000 resamples;
95% CI for ΔR²: [0.453, 0.476]).

---

## Repository Structure

```
climate-narrator/
│
├── climate-narrator.ipynb       # Main notebook — full pipeline end to end
│
├── models/
│   └── README.md                # Pre-trained model download link (Google Drive)
│
├── figures/                     # Generated figures (created at runtime)
├── results/                     # Generated CSVs — metrics, predictions, SHAP
│
└── README.md                    # This file
```

The notebook is self-contained. Running it top to bottom will:
1. Download NOAA Storm Events data (2000–2024) directly from NOAA's servers
2. Engineer structured features and TF-IDF narrative representations
3. Train ClimateNarrator and all baselines
4. Run evaluation, uncertainty quantification, and all analyses
5. Save all figures and result CSVs

---

## Setup

### Requirements

- Python 3.9+
- CUDA-capable GPU recommended (the notebook was developed on a Kaggle P100, 16 GB VRAM)
- CPU-only execution is possible but slow for the fusion model training

### Option A — Google Colab or Kaggle (recommended)

Upload `climate-narrator.ipynb` directly to
[Google Colab](https://colab.research.google.com) or
[Kaggle Notebooks](https://www.kaggle.com/code).
Set the runtime to GPU. The first cell installs all dependencies automatically.

### Option B — Local environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/climate-narrator.git
cd climate-narrator

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers scikit-learn pandas numpy matplotlib seaborn
pip install shap xgboost tqdm accelerate
```

> **Note on PyTorch version:** The notebook was developed with PyTorch 2.0 /
> CUDA 11.8. If you have a different CUDA version, replace `cu118` with your
> version (e.g. `cu121` for CUDA 12.1). For CPU-only: omit the
> `--index-url` flag entirely.

---

## Running the Notebook

Open `climate-narrator.ipynb` in Jupyter or Colab and run all cells in order.

**Cell guide:**

| Cell | Description |
|---|---|
| Cell 1 | Install dependencies |
| Cell 2 | Imports, global config, reproducibility seed |
| Cell 3 | NOAA data download and parsing (2000–2024) |
| Cell 4 | Feature engineering — damage parsing, event taxonomy, TF-IDF |
| Cell 5 | Dataset construction — train/val/test temporal split |
| Cell 6 | Model architecture — dual-stream fusion with cross-modal attention |
| Cell 7 | Training utilities — loss functions, metrics |
| Cell 8 | Training loop |
| Cell 9 | Structured-only baselines (XGBoost, Random Forest) |
| Cell 10–16 | Evaluation, uncertainty quantification, SHAP, token attribution |
| Cell NEW-F | New scientific analyses — mechanism taxonomy, regional vulnerability, conformal prediction |

**Key config options** (Cell 2, `CFG` dictionary):

```python
CFG = {
    "years":           list(range(2005, 2024)),  # years to download
    "test_year_start": 2022,                     # temporal holdout start
    "val_year_start":  2019,                     # validation start
    "batch_size":      256,
    "epochs":          20,
    "lr_struct":       3e-4,
}
```

To run a quick smoke test, set `"years": list(range(2020, 2024))` and
`"epochs": 2`.

---

## Pre-trained Model

The best trained ClimateNarrator checkpoint is available under models directory.

---

## Data

NOAA Storm Events data is downloaded automatically by the notebook directly from
NOAA's public servers — no manual download is required. The full dataset
(2000–2024, ~380,000 filtered events) requires approximately 2 GB of disk space
and 20–30 minutes to download on a standard connection.

**Data source:**
NOAA National Centers for Environmental Information.
Storm Events Database.
https://www.ncdc.noaa.gov/stormevents/

---

<!-- ## Citation

If you use this code or the pre-trained model in your work, please cite:

```bibtex
@article{goyal2025climatenarrrator,
  title   = {Expert-Authored Storm Narratives Improve Assessment of Extreme Weather Damage},
  author  = {Goyal, Aayush and Khan, Talha Ali},
  journal = {Communications Earth \& Environment},
  year    = {2025},
  note    = {Under review}
}
``` -->

---

## License

This repository is released under the MIT License.
NOAA Storm Events data is public domain.
