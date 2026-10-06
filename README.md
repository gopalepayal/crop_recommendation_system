# Crop Recommendation System

**M.Sc. Computer Science — Semester III Research Project**

A Django-based web application that recommends the most suitable crop to cultivate based on soil and environmental parameters, using a trained machine learning model.

---

## Project Overview

This system analyses 7 soil and environmental parameters — **N, P, K, Temperature, Humidity, pH, and Rainfall** — to predict the best crop from 22 possible crop classes. Five ML classifiers are trained and compared; the best-performing model is saved and served through an interactive research dashboard.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Django 4+ |
| ML | scikit-learn, XGBoost, pandas, numpy |
| Serialisation | Joblib |
| Frontend | HTML5, CSS3, Bootstrap 5, JavaScript |
| Visualisation | Plotly |
| Database | SQLite (Django default) |

---

## Dataset

- **Source:** Publicly available Crop Recommendation Dataset
- **File:** `data/Crop_recommendation.csv`
- **Records:** 2,200 (100 per crop class)
- **Features (7):** N, P, K, temperature, humidity, ph, rainfall
- **Target:** label (22 crop classes)
- **Missing values:** 0 | **Duplicates:** 0

---

## ML Models Trained

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. XGBoost

Metrics computed: Accuracy, Precision, Recall, F1-Score (weighted average).
Best model selected dynamically from actual results.

---

## Installation

```bat
:: Create virtual environment
python -m venv venv

:: Activate virtual environment
venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt
```

---

## ML Training

Run once before starting the server:

```bat
python ml\train_model.py
```

This trains all 5 models, saves `best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, and `model_results.csv` to `ml/saved_models/`.

---

## Django Setup

```bat
python manage.py migrate
```

---

## Run

```bat
python manage.py runserver
```

Open your browser at: **http://127.0.0.1:8000/**

---

## Dashboard Pages

| Page | URL |
|---|---|
| Overview | `/` |
| Crop Analysis | `/crop-analysis/` |
| Environmental Analysis | `/environmental-analysis/` |
| Model Performance | `/model-performance/` |
| Crop Recommendation | `/recommendation/` |

---

## Project Structure

```
crop_recommendation_system/
├── data/                   # Original dataset (unchanged)
│   └── Crop_recommendation.csv
├── ml/
│   ├── train_model.py      # Training pipeline
│   ├── predict.py          # Prediction test script
│   └── saved_models/       # Saved pkl files + model_results.csv
├── dashboard/
│   ├── views.py            # Django views
│   ├── utils.py            # Data loading and chart helpers
│   └── urls.py             # App URL routes
├── templates/              # HTML templates
├── static/css/             # Custom stylesheet
├── manage.py
└── requirements.txt
```
