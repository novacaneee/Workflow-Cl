import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# =======================
# KONFIGURASI PATH
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "heart_failure_preprocessing", "train_data.csv")
TEST_PATH  = os.path.join(BASE_DIR, "heart_failure__preprocessing", "test_data.csv")

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

def run_skilled_local():
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Eksperimen_Heart_Failure_IdaBagus_Skilled")

    n_estimators_list = [50, 100, 200]
    max_depth_list    = [3, 5, None]

    best_f1 = -1.0
    best_model = None

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            with mlflow.start_run():
                print(f"[SKILLED] Run n_estimators={n_estimators}, max_depth={max_depth}")

                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("model_type", "RandomForest_Skilled")

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    class_weight="balanced"
                )
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                acc  = accuracy_score(y_test, y_pred)
                f1   = f1_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec  = recall_score(y_test, y_pred)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)

                mlflow.sklearn.log_model(model, "random_forest_model")

                print(f"[SKILLED] ACC={acc:.4f}, F1={f1:.4f}, PREC={prec:.4f}, REC={rec:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model

    # Export best model untuk Docker
    if best_model is not None:
        export_dir = os.path.join(BASE_DIR, "exported_model")
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        mlflow.sklearn.save_model(best_model, export_dir)
        print(f"[INFO] Best model exported to {export_dir}")

if __name__ == "__main__":
    run_skilled_local()
