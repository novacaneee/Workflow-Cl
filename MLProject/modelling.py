import os
import shutil
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from joblib import dump

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "heart_failure_preprocessing", "train_data.csv")
TEST_PATH  = os.path.join(BASE_DIR, "heart_failure_preprocessing", "test_data.csv")

def load_data():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("train_data.csv atau test_data.csv tidak ditemukan.")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df.drop("DEATH_EVENT", axis=1)
    y_train = train_df["DEATH_EVENT"]
    X_test  = test_df.drop("DEATH_EVENT", axis=1)
    y_test  = test_df["DEATH_EVENT"]

    train_clean = pd.concat([X_train, y_train], axis=1).dropna(subset=["DEATH_EVENT"])
    test_clean  = pd.concat([X_test,  y_test],  axis=1).dropna(subset=["DEATH_EVENT"])

    X_train = train_clean.drop("DEATH_EVENT", axis=1)
    y_train = train_clean["DEATH_EVENT"]
    X_test  = test_clean.drop("DEATH_EVENT", axis=1)
    y_test  = test_clean["DEATH_EVENT"]

    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    print(f"[METRIC] Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    export_dir = os.path.join(BASE_DIR, "exported_model")
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    model_path = os.path.join(export_dir, "model.joblib")
    dump(model, model_path)
    print(f"[INFO] Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
