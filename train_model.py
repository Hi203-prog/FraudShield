import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "rf_fraud.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.joblib")

def train():
    print("Chargement des donnees...")
    df = pd.read_csv(DATA_PATH)

    # Features et cible
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Normalisation de Amount et Time (V1-V28 deja normalisees par PCA)
    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    print(f"Dataset charge : {len(df)} transactions ({y.sum()} fraudes)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE pour reequilibrer (dataset tres desequilibre)
    print("Reequilibrage SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"    {y_resampled.sum()} fraudes apres SMOTE")

    # Entrainement Random Forest 
    print("Entrainement Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_resampled, y_resampled)

    # Evaluation
    print("\nResultats sur le test set :")
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=["Legitime", "Fraude"]))
    print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.4f}")

    # Sauvegarde
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModele sauvegarde -> {MODEL_PATH}")
    print(f"Scaler sauvegarde -> {SCALER_PATH}")

if __name__ == "__main__":
    train()