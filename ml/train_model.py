import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def main():
    print("Loading dataset...")
    # Get absolute path to the dataset relative to this script
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'Crop_recommendation.csv')
    df = pd.read_csv(data_path)
    
    # Feature names expected
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_cols]
    y = df['label']
    
    print("Preprocessing...")
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }
    
    results = []
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        })
        
        # Check best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_f1 = f1
            best_model = model
            best_model_name = name
        elif accuracy == best_accuracy and f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            
    # Create results dataframe
    results_df = pd.DataFrame(results)
    print("\nModel Results:")
    print(results_df)
    
    # Save outputs
    save_dir = os.path.join(BASE_DIR, 'ml', 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving best model ({best_model_name}) and preprocessing objects...")
    joblib.dump(best_model, os.path.join(save_dir, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(save_dir, 'label_encoder.pkl'))
    
    # Save model_results.csv to ml/ as requested, and optionally inside saved_models too
    results_df.to_csv(os.path.join(BASE_DIR, 'ml', 'model_results.csv'), index=False)
    results_df.to_csv(os.path.join(save_dir, 'model_results.csv'), index=False)
    
    print("ML pipeline complete.")

if __name__ == "__main__":
    main()
