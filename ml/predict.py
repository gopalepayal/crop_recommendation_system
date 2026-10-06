import pandas as pd
import joblib
import os

def test_prediction():
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(BASE_DIR, 'ml', 'saved_models')
    data_path = os.path.join(BASE_DIR, 'data', 'Crop_recommendation.csv')

    # Load one sample from the original dataset
    df = pd.read_csv(data_path)
    # Let's take row index 150 (should be 'maize')
    sample = df.iloc[[150]]
    actual_label = sample['label'].values[0]
    
    # Feature columns expected
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X_sample = sample[feature_cols]

    print(f"Sample Input:\n{X_sample}")
    print(f"Actual Label: {actual_label}")

    # Load models
    best_model = joblib.load(os.path.join(save_dir, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
    le = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))

    # Preprocess
    X_sample_scaled = scaler.transform(X_sample)

    # Predict
    prediction_encoded = best_model.predict(X_sample_scaled)
    prediction_label = le.inverse_transform(prediction_encoded)[0]

    print(f"Predicted Label: {prediction_label}")

    if actual_label == prediction_label:
        print("Verification: SUCCESS")
    else:
        print("Verification: PREDICTION MISMATCH (This can happen if the model made an error on this sample)")

if __name__ == "__main__":
    test_prediction()
