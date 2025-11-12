# run_pipeline.py
# Main script to run data preprocessing → model training & evaluation (for multiple models)

import os
import sys

# --- 1️⃣ FIX PYTHON PATH ---
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# --- 2️⃣ IMPORT MODULES ---
from data_preprocessing import load_and_preprocess
from model_training import train_and_evaluate_models

# --- 3️⃣ STEP 1: Data Preprocessing ---
print("\n=== STEP 1: Data Preprocessing ===")
X_train, X_test, y_train, y_test = load_and_preprocess("data/Shipping_data.csv")

# --- 4️⃣ STEP 2: Model Training + Evaluation ---
print("\n=== STEP 2: Model Training & Evaluation ===")
results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# --- 5️⃣ FINAL SUMMARY ---
print("\n✅ Pipeline executed successfully!")
print("\n📊 Final Model Comparison Summary:\n")
print(results_df)
