# src/train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import joblib

def build_fraud_brain():
    # Resolve local path maps relative to this file's directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "creditcard.csv")
    model_output_path = os.path.join(base_dir, "data", "fraud_model.pkl")

    print("📊 Loading the Credit Card Transaction Dataset...")
    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find creditcard.csv at {data_path}.")
        return

    df = pd.read_csv(data_path)
    
    total_records = len(df)
    fraud_records = df['Class'].sum()
    normal_records = total_records - fraud_records
    print(f"ℹ️ Total Rows: {total_records} | Normal: {normal_records} | Fraud: {fraud_records} ({round((fraud_records/total_records)*100, 3)}%)")

    # Drop 'Time' as it's just an absolute sequence row, not a behavioral feature
    X = df.drop(columns=['Time', 'Class'])
    y = df['Class']

    # Stratified split to ensure train and test sets have the exact same ratio of fraud cases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("🛡️ Applying SMOTE to synthesize minority fraud points...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"✅ Rebalanced Training Set: Normal={np.sum(y_train_resampled==0)} | Synthesized Fraud={np.sum(y_train_resampled==1)}")

    print("🌲 Training the Standalone Random Forest Engine (this may take a moment)...")
    model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)

    print("\n🔍 Evaluating Model Performance on Test Data...")
    predictions = model.predict(X_test)
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, predictions))
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions, target_names=["Normal", "FRAUD"]))

    macro_f1 = f1_score(y_test, predictions, average='macro')
    print(f"🏆 System Balance Score achieved (Macro F1): {round(macro_f1 * 100, 2)}%")

    print(f"💾 Exporting the trained AI brain file to: {model_output_path}...")
    joblib.dump(model, model_output_path)
    print("🚀 Standalone model saved successfully!")

if __name__ == "__main__":
    build_fraud_brain()