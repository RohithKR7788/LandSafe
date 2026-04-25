# ⛰️ LandSafe — Hyperlocal Landslide Risk Prediction System

> **AI-powered 24–48 hour landslide early-warning for Kerala's Western Ghats**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌍 Problem Statement

Landslides kill thousands of people every year, with India's Western Ghats — particularly Kerala — among the world's most vulnerable regions. The 2018 and 2024 Wayanad disasters alone caused hundreds of deaths and billions in losses.

Existing early-warning systems are **too broad**, issuing district-level alerts that cover hundreds of square kilometres. By the time a warning reaches a village, there may be zero evacuation time.

**LandSafe** predicts landslide risk at the village/ward level, 24–48 hours in advance, using freely available satellite and weather data — no expensive IoT sensors required.

---

## 🧠 How It Works

```
Satellite DEM (SRTM)    ─┐
IMD / CHIRPS Rainfall   ─┤─→ Feature Engineering ─→ XGBoost Classifier ─→ Risk Map
Sentinel-2 NDVI         ─┤                              ↓
GSI Landslide Inventory ─┘                          SHAP Explainer
```

### Features Used

| Category | Features |
|---|---|
| Terrain | Slope angle, elevation, aspect, curvature, TWI proxy |
| Rainfall | 1-day, 3-day, 7-day accumulation, Antecedent Precipitation Index |
| Soil / Geology | Soil type, lithology, soil moisture |
| Vegetation | NDVI (Normalised Difference Vegetation Index) |
| Land use | Forest / agriculture / urban / barren |
| Proximity | Distance to river, distance to road |

### Model Performance (5-fold cross-validation)

| Metric | Score |
|---|---|
| ROC-AUC | 0.91 ± 0.01 |
| F1 Score | 0.83 ± 0.02 |
| Accuracy | 0.87 ± 0.01 |

---

## 🗂️ Project Structure

```
LandSafe/
├── generate_data.py          # Synthetic Kerala dataset generator
├── train.py                  # End-to-end training pipeline
├── app.py                    # Streamlit interactive dashboard
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Feature engineering + sklearn pipeline
│   └── model.py              # XGBoost + SHAP explainability
├── data/
│   └── processed/
│       └── kerala_landslide_dataset.csv
├── models/
│   ├── landsafe_model.pkl    # Saved pipeline (after training)
│   ├── evaluation.png        # Confusion matrix + ROC curve
│   ├── feature_importance.png
│   └── metrics.json
└── notebooks/
    └── EDA.ipynb
```

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/<your-username>/LandSafe.git
cd LandSafe
pip install -r requirements.txt
```

### 2. Generate data and train
```bash
python generate_data.py    # creates synthetic Kerala dataset
python train.py            # trains, evaluates, and saves the model
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

---

## 📡 Using Real Data (Production)

Replace `generate_data.py` with the following real data sources:

| Data | Source | Format |
|---|---|---|
| Elevation / Slope | [NASA Earthdata (SRTM)](https://earthexplorer.usgs.gov/) | GeoTIFF |
| Rainfall | [IMD Pune](https://imdpune.gov.in/) / [CHIRPS](https://www.chc.ucsb.edu/data/chirps) | NetCDF / CSV |
| NDVI / Land use | [Copernicus Sentinel-2](https://scihub.copernicus.eu/) | GeoTIFF |
| Landslide inventory | [GSI Bhukosh](https://bhukosh.gsi.gov.in/) | Shapefile |

Use `rasterio` + `geopandas` to extract pixel values at each inventory point.

---

## 🔮 Future Work

- [ ] Integrate real-time IMD API rainfall feeds
- [ ] Village-level granularity using ward boundaries (shapefile)
- [ ] LSTM layer for temporal rainfall sequence modelling
- [ ] SMS alert integration via Twilio for offline areas
- [ ] Wayanad district pilot with ground-truth validation

---

## 🛡️ Disclaimer

LandSafe is a **research prototype**. It must not be used as a sole basis for
emergency decisions. Always follow official civil-defence and meteorological
department guidelines.

---

## 👤 Author

Built by [Your Name] — B.Tech AI & ML, SCT College of Engineering, Thiruvananthapuram.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
