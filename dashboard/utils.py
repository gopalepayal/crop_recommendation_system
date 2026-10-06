import os
import json
import pandas as pd
import numpy as np
import joblib
from django.conf import settings
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

BASE_DIR = settings.BASE_DIR

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# ─────────────────────────────────────────────
#  Path helpers
# ─────────────────────────────────────────────
def get_data_path():
    return os.path.join(BASE_DIR, 'data', 'Crop_recommendation.csv')

def get_results_path():
    return os.path.join(BASE_DIR, 'ml', 'model_results.csv')

def get_model_dir():
    return os.path.join(BASE_DIR, 'ml', 'saved_models')

# ─────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────
def load_data():
    return pd.read_csv(get_data_path())

def load_results():
    df = pd.read_csv(get_results_path())
    df = df.rename(columns={'F1-score': 'F1_score'})
    return df

def _load_model_objects():
    d = get_model_dir()
    model  = joblib.load(os.path.join(d, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(d, 'scaler.pkl'))
    le     = joblib.load(os.path.join(d, 'label_encoder.pkl'))
    return model, scaler, le

# ─────────────────────────────────────────────
#  Overview / KPI
# ─────────────────────────────────────────────
def get_best_model_info():
    results = load_results()
    best_row = results.sort_values(
        by=['Accuracy', 'F1_score'], ascending=[False, False]
    ).iloc[0]
    return {
        'name':     best_row['Model'],
        'accuracy': best_row['Accuracy'],
        'f1':       best_row['F1_score'],
    }

def get_overview_stats():
    df = load_data()
    best_info = get_best_model_info()
    return {
        'total_records': len(df),
        'total_classes': df['label'].nunique(),
        'num_features':  len(FEATURE_COLS),
        'best_model':    best_info['name'],
        'best_accuracy': round(best_info['accuracy'] * 100, 2),
    }

# ─────────────────────────────────────────────
#  Overview / Crop distribution chart
# ─────────────────────────────────────────────
def generate_crop_distribution_chart():
    df = load_data()
    counts = df['label'].value_counts().reset_index()
    counts.columns = ['Crop', 'Count']
    fig = px.bar(counts, x='Crop', y='Count',
                 title='Crop Distribution', color='Crop',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(showlegend=False, xaxis_tickangle=-35)
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# ─────────────────────────────────────────────
#  Crop Analysis — nutrient + environment charts
# ─────────────────────────────────────────────
def generate_nutrient_charts():
    df = load_data()
    avg_df = df.groupby('label')[['N', 'P', 'K']].mean().reset_index()
    fig_n  = px.bar(avg_df, x='label', y='N',  title='Average Nitrogen by Crop',    color='label')
    fig_p  = px.bar(avg_df, x='label', y='P',  title='Average Phosphorus by Crop',  color='label')
    fig_k  = px.bar(avg_df, x='label', y='K',  title='Average Potassium by Crop',   color='label')
    for f in (fig_n, fig_p, fig_k):
        f.update_layout(showlegend=False, xaxis_tickangle=-35)
    return {
        'n': fig_n.to_html(full_html=False, include_plotlyjs='cdn'),
        'p': fig_p.to_html(full_html=False, include_plotlyjs='cdn'),
        'k': fig_k.to_html(full_html=False, include_plotlyjs='cdn'),
    }

def generate_environmental_charts_crop():
    df = load_data()
    avg_df = df.groupby('label')[['temperature', 'humidity', 'ph', 'rainfall']].mean().reset_index()
    fig_t  = px.bar(avg_df, x='label', y='temperature', title='Average Temperature by Crop', color='label')
    fig_h  = px.bar(avg_df, x='label', y='humidity',    title='Average Humidity by Crop',    color='label')
    fig_ph = px.bar(avg_df, x='label', y='ph',          title='Average Soil pH by Crop',     color='label')
    fig_r  = px.bar(avg_df, x='label', y='rainfall',    title='Average Rainfall by Crop',    color='label')
    for f in (fig_t, fig_h, fig_ph, fig_r):
        f.update_layout(showlegend=False, xaxis_tickangle=-35)
    return {
        'temp':     fig_t.to_html(full_html=False, include_plotlyjs='cdn'),
        'humidity': fig_h.to_html(full_html=False, include_plotlyjs='cdn'),
        'ph':       fig_ph.to_html(full_html=False, include_plotlyjs='cdn'),
        'rainfall': fig_r.to_html(full_html=False, include_plotlyjs='cdn'),
    }

# ─────────────────────────────────────────────
#  Crop Analysis — Crop Profile
# ─────────────────────────────────────────────
def get_crop_profile_data():
    """
    Returns a dict keyed by crop label, each containing per-feature averages
    and overall dataset averages, as well as the list of all crop labels.
    """
    df = load_data()
    overall = df[FEATURE_COLS].mean().round(3).to_dict()
    crop_list = sorted(df['label'].unique().tolist())

    profiles = {}
    for crop in crop_list:
        sub = df[df['label'] == crop]
        avgs = sub[FEATURE_COLS].mean().round(3).to_dict()
        profiles[crop] = {
            'count': int(len(sub)),
            'averages': avgs,
        }

    return {
        'crop_list':    crop_list,
        'profiles':     profiles,
        'overall_avgs': overall,
    }

def generate_crop_profile_comparison_charts(crop, df=None):
    """
    Returns two Plotly bar comparison charts for selected crop vs dataset overall:
    1. Soil Nutrients (N, P, K)
    2. Environmental parameters (Temp, Humidity, pH, Rainfall)
    """
    if df is None:
        df = load_data()
    overall = df[FEATURE_COLS].mean()
    crop_sub = df[df['label'] == crop]
    if crop_sub.empty:
        return None
    crop_avg = crop_sub[FEATURE_COLS].mean()

    # Nutrients (N, P, K)
    fig_nut = go.Figure()
    fig_nut.add_trace(go.Bar(
        name=crop.capitalize(),
        x=['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'],
        y=[round(crop_avg['N'], 2), round(crop_avg['P'], 2), round(crop_avg['K'], 2)],
        marker_color='#1a6b3c',
        text=[round(crop_avg['N'], 2), round(crop_avg['P'], 2), round(crop_avg['K'], 2)],
        textposition='auto',
    ))
    fig_nut.add_trace(go.Bar(
        name='Dataset Overall Average',
        x=['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'],
        y=[round(overall['N'], 2), round(overall['P'], 2), round(overall['K'], 2)],
        marker_color='#aacbb5',
        text=[round(overall['N'], 2), round(overall['P'], 2), round(overall['K'], 2)],
        textposition='auto',
    ))
    fig_nut.update_layout(
        title=f'{crop.capitalize()} vs Dataset Avg — Soil Nutrients (kg/ha)',
        barmode='group',
        xaxis_title='Nutrient',
        yaxis_title='Mean Value',
        legend=dict(orientation='h', y=-0.25),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Environmental (Temp, Humidity, pH, Rainfall)
    fig_env = go.Figure()
    env_cols = ['temperature', 'humidity', 'ph', 'rainfall']
    env_labels = ['Temperature (°C)', 'Humidity (%)', 'Soil pH', 'Rainfall (mm)']
    fig_env.add_trace(go.Bar(
        name=crop.capitalize(),
        x=env_labels,
        y=[round(crop_avg[f], 2) for f in env_cols],
        marker_color='#2d9e6b',
        text=[round(crop_avg[f], 2) for f in env_cols],
        textposition='auto',
    ))
    fig_env.add_trace(go.Bar(
        name='Dataset Overall Average',
        x=env_labels,
        y=[round(overall[f], 2) for f in env_cols],
        marker_color='#c2e5d0',
        text=[round(overall[f], 2) for f in env_cols],
        textposition='auto',
    ))
    fig_env.update_layout(
        title=f'{crop.capitalize()} vs Dataset Avg — Environmental Requirements',
        barmode='group',
        xaxis_title='Environmental Metric',
        yaxis_title='Mean Value',
        legend=dict(orientation='h', y=-0.25),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return {
        'nutrients': fig_nut.to_html(full_html=False, include_plotlyjs='cdn'),
        'env':       fig_env.to_html(full_html=False, include_plotlyjs='cdn'),
    }

def generate_crop_profile_comparison_chart(crop, df=None):
    """Backwards-compatible single comparison chart."""
    charts = generate_crop_profile_comparison_charts(crop, df)
    return charts['nutrients'] if charts else None

# ─────────────────────────────────────────────
#  Environmental Analysis charts
# ─────────────────────────────────────────────
def generate_environmental_analysis_charts():
    df = load_data()
    fig_temp_dist = px.histogram(df, x='temperature', title='Temperature Distribution', marginal='box')
    fig_hum_dist  = px.histogram(df, x='humidity',    title='Humidity Distribution',    marginal='box')
    fig_rain_dist = px.histogram(df, x='rainfall',    title='Rainfall Distribution',    marginal='box')
    fig_ph_dist   = px.histogram(df, x='ph',          title='Soil pH Distribution',     marginal='box')
    fig_npk = px.box(
        df.melt(id_vars=['label'], value_vars=['N', 'P', 'K']),
        x='variable', y='value', color='variable',
        title='N/P/K Distribution Comparison',
    )
    corr = df[FEATURE_COLS].corr()
    fig_corr = px.imshow(
        corr, text_auto=True, aspect='auto',
        title='Feature Correlation Heatmap',
        color_continuous_scale='RdBu_r',
    )
    fig_tvh = px.scatter(df, x='temperature', y='humidity', color='label',
                         title='Temperature vs Humidity')
    return {
        'temp_dist': fig_temp_dist.to_html(full_html=False, include_plotlyjs='cdn'),
        'hum_dist':  fig_hum_dist.to_html(full_html=False, include_plotlyjs='cdn'),
        'rain_dist': fig_rain_dist.to_html(full_html=False, include_plotlyjs='cdn'),
        'ph_dist':   fig_ph_dist.to_html(full_html=False, include_plotlyjs='cdn'),
        'npk':       fig_npk.to_html(full_html=False, include_plotlyjs='cdn'),
        'corr':      fig_corr.to_html(full_html=False, include_plotlyjs='cdn'),
        'tvh':       fig_tvh.to_html(full_html=False, include_plotlyjs='cdn'),
    }

# ─────────────────────────────────────────────
#  Model Performance — existing charts
# ─────────────────────────────────────────────
def generate_model_performance_charts():
    results = load_results()
    fig_acc = px.bar(
        results, x='Model', y='Accuracy',
        title='Model Accuracy Comparison', color='Model', text_auto='.4f',
    )
    fig_acc.update_layout(yaxis_range=[0.9, 1.0], showlegend=False)
    metrics_melted = results.melt(
        id_vars=['Model'],
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1_score'],
    )
    fig_metrics = px.bar(
        metrics_melted, x='Model', y='value', color='variable',
        barmode='group', title='All Metrics Comparison',
    )
    fig_metrics.update_layout(yaxis_range=[0.9, 1.0])
    return {
        'accuracy':    fig_acc.to_html(full_html=False, include_plotlyjs='cdn'),
        'all_metrics': fig_metrics.to_html(full_html=False, include_plotlyjs='cdn'),
    }

# ─────────────────────────────────────────────
#  Model Performance — Feature Importance
# ─────────────────────────────────────────────
def get_feature_importance_table_data():
    """Returns sorted list of feature importances with percentage shares."""
    model, scaler, le = _load_model_objects()
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'feature':    FEATURE_COLS,
        'importance': importances,
    }).sort_values('importance', ascending=False)

    records = []
    for _, row in imp_df.iterrows():
        records.append({
            'feature':    row['feature'],
            'importance': round(float(row['importance']), 4),
            'percentage': round(float(row['importance']) * 100, 2),
        })
    return records

def generate_feature_importance_chart():
    model, scaler, le = _load_model_objects()
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature':    FEATURE_COLS,
        'Importance': importances,
    }).sort_values('Importance', ascending=True)  # ascending so bar chart is top-to-bottom

    fig = px.bar(
        imp_df, x='Importance', y='Feature',
        orientation='h',
        title='Feature Importance (Trained Best Model)',
        color='Importance',
        color_continuous_scale='Greens',
        text=imp_df['Importance'].round(4),
    )
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_traces(textposition='outside')
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# ─────────────────────────────────────────────
#  Model Performance — Confusion Matrix
# ─────────────────────────────────────────────
def generate_confusion_matrix_chart():
    """
    Reproduces the same train/test split (random_state=42, stratified) used
    during training, applies the saved scaler, runs the saved best model, and
    returns an interactive Plotly heatmap of the confusion matrix.
    """
    df = load_data()
    model, scaler, le = _load_model_objects()

    X = df[FEATURE_COLS]
    y_encoded = le.transform(df['label'])

    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    class_labels = le.classes_          # sorted list of crop names
    cm = confusion_matrix(y_test, y_pred)

    fig = px.imshow(
        cm,
        x=class_labels,
        y=class_labels,
        labels=dict(x='Predicted', y='Actual', color='Count'),
        title='Confusion Matrix — Best Model (Test Set)',
        color_continuous_scale='Greens',
        aspect='auto',
        text_auto=True,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title='Predicted Crop',
        yaxis_title='Actual Crop',
        height=650,
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# ─────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────
def predict_crop(n, p, k, temp, humidity, ph, rainfall):
    model, scaler, le = _load_model_objects()
    input_df = pd.DataFrame({
        'N': [n], 'P': [p], 'K': [k],
        'temperature': [temp], 'humidity': [humidity],
        'ph': [ph], 'rainfall': [rainfall],
    })
    input_scaled = scaler.transform(input_df)
    pred_encoded = model.predict(input_scaled)
    return le.inverse_transform(pred_encoded)[0]

def predict_crop_with_probabilities(n, p, k, temp, humidity, ph, rainfall):
    """
    Uses saved Random Forest model and scaler to predict crop and return
    actual prediction confidence (%) alongside top 3 ranked crop recommendations.
    """
    model, scaler, le = _load_model_objects()
    input_df = pd.DataFrame({
        'N': [n], 'P': [p], 'K': [k],
        'temperature': [temp], 'humidity': [humidity],
        'ph': [ph], 'rainfall': [rainfall],
    })
    input_scaled = scaler.transform(input_df)

    probs = model.predict_proba(input_scaled)[0]
    classes = le.classes_

    sorted_indices = np.argsort(probs)[::-1]

    top_class_idx = sorted_indices[0]
    top_crop_raw  = classes[top_class_idx]
    top_confidence = float(probs[top_class_idx])

    top3 = []
    for rank, idx in enumerate(sorted_indices[:3], start=1):
        prob_val = float(probs[idx])
        top3.append({
            'rank':        rank,
            'crop':        classes[idx].capitalize(),
            'raw_crop':    classes[idx],
            'probability': prob_val,
            'percentage':  round(prob_val * 100, 2),
        })

    return {
        'prediction':     top_crop_raw.capitalize(),
        'raw_pred':       top_crop_raw,
        'confidence':     top_confidence,
        'confidence_pct': round(top_confidence * 100, 2),
        'top3':           top3,
    }

# ─────────────────────────────────────────────
#  Recommendation — "Why this crop?" crop profile
# ─────────────────────────────────────────────
def get_crop_averages(crop_label):
    """Returns actual dataset averages for the given crop label."""
    df = load_data()
    sub = df[df['label'] == crop_label]
    if sub.empty:
        return None
    avgs = sub[FEATURE_COLS].mean().round(3).to_dict()
    avgs['count'] = len(sub)
    return avgs

def get_why_comparison_data(user_input, crop_label):
    """
    Returns structured feature-by-feature comparison between user input
    and dataset averages for the recommended crop class.
    """
    crop_avgs = get_crop_averages(crop_label)
    if not crop_avgs:
        return []

    metrics = [
        ('N',           'Nitrogen (N)',       'kg/ha'),
        ('P',           'Phosphorus (P)',     'kg/ha'),
        ('K',           'Potassium (K)',      'kg/ha'),
        ('temperature', 'Temperature',        '°C'),
        ('humidity',    'Humidity',           '%'),
        ('ph',          'Soil pH',            ''),
        ('rainfall',    'Rainfall',           'mm'),
    ]

    rows = []
    for key, name, unit in metrics:
        u_val = user_input[key]
        c_val = crop_avgs[key]
        diff  = round(u_val - c_val, 2)
        rows.append({
            'key':      key,
            'name':     name,
            'unit':     unit,
            'user_val': round(u_val, 2),
            'crop_avg': round(c_val, 2),
            'diff':     diff,
        })
    return rows

def generate_why_chart(user_input, crop_label):
    """
    Bar chart comparing user's submitted values vs the predicted crop's
    dataset averages.
    """
    crop_avgs = get_crop_averages(crop_label)
    if crop_avgs is None:
        return None

    feature_labels = {
        'N':           'N (Nitrogen)',
        'P':           'P (Phosphorus)',
        'K':           'K (Potassium)',
        'temperature': 'Temperature (°C)',
        'humidity':    'Humidity (%)',
        'ph':          'Soil pH',
        'rainfall':    'Rainfall (mm)',
    }
    features = list(feature_labels.keys())
    user_vals = [round(user_input[f], 2) for f in features]
    crop_vals = [round(crop_avgs[f], 2) for f in features]
    labels    = [feature_labels[f] for f in features]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Submitted Input', x=labels, y=user_vals, marker_color='#2d9e6b', text=user_vals, textposition='auto'))
    fig.add_trace(go.Bar(name=f'{crop_label.capitalize()} (Dataset Avg)', x=labels, y=crop_vals, marker_color='#aacbb5', text=crop_vals, textposition='auto'))
    fig.update_layout(
        barmode='group',
        title='Submitted Input vs Recommended Crop Dataset Averages',
        xaxis_title='Feature',
        yaxis_title='Value',
        legend=dict(orientation='h', y=-0.28),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
