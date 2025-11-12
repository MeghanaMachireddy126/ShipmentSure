# ==============================================
# 📄 data_preprocessing.py (Final Fixed)
# ==============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_and_preprocess(filepath='data/Shipping_data.csv'):
    """
    Loads the dataset, preprocesses it, and splits into train & test sets.
    """

    # --- 1️⃣ Check file path ---
    full_path = os.path.join(os.getcwd(), filepath)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ File not found at: {full_path}")

    # --- 2️⃣ Load dataset ---
    df = pd.read_csv(full_path)
    print("✅ Data loaded successfully!")
    print("📊 Shape of data:", df.shape)
    print(df.head(), "\n")

    # --- 3️⃣ Handle missing values ---
    print("🔍 Missing values:\n", df.isnull().sum())
    df.fillna(df.mode().iloc[0], inplace=True)
    print("✅ Missing values filled.\n")

    # --- 4️⃣ Encode categorical columns ---
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
            print(f"🔠 Encoded column: {col}")
    print()

    # --- 5️⃣ Pick target column ---
    # In your dataset, the target is clearly "on_time"
    target_col = 'on_time'
    if target_col not in df.columns:
        raise KeyError(f"❌ Target column '{target_col}' not found. Columns: {df.columns.tolist()}")

    print(f"🎯 Target column detected: {target_col}")

    # --- 6️⃣ Split features and target ---
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # --- 7️⃣ Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✅ Data preprocessing completed.")
    print(f"   Train shape: {X_train.shape} | Test shape: {X_test.shape}\n")

    return X_train, X_test, y_train, y_test


# --- Run directly for testing ---
if __name__ == "__main__":
    load_and_preprocess()
