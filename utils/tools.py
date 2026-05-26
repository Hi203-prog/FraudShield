"""
utils/tools.py -- Outils partages entre les agents CrewAI.
"""

import os
import joblib
import numpy as np
import pandas as pd
from crewai.tools import tool

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rf_fraud.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.joblib")

# Chargement unique du modele (singleton)
_rf_model = None
_scaler = None

def _load_model():
    global _rf_model, _scaler
    if _rf_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Modele introuvable. Lance d'abord : python train_model.py"
            )
        _rf_model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _rf_model, _scaler


@tool("fraud_score_calculator")
def fraud_score_calculator(transaction_json: str) -> str:
    """
    Calcule un score de risque (0.0 a 1.0) pour une transaction.
    Entree : JSON string avec les champs Time, V1..V28, Amount.
    Sortie : score de risque + top features.
    """
    import json

    try:
        tx = json.loads(transaction_json)
    except Exception:
        return "ERREUR: JSON invalide."

    rf, scaler = _load_model()

    # Construction du vecteur de features dans le bon ordre
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    row = pd.DataFrame([{col: tx.get(col, 0.0) for col in feature_cols}])

    # Normalisation Amount et Time
    row[["Amount", "Time"]] = scaler.transform(row[["Amount", "Time"]])

    score = rf.predict_proba(row)[0][1]

    # Top 5 features importantes
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    top_features = [
        f"{feature_cols[i]}={float(row.iloc[0, i]):.3f} (importance={importances[i]:.3f})"
        for i in top_idx
    ]

    return (
        f"SCORE_RISQUE: {score:.4f}\n"
        f"TOP_FEATURES: {', '.join(top_features)}\n"
        f"MONTANT: {tx.get('Amount', 0):.2f} EUR\n"
        f"TEMPS: {tx.get('Time', 0):.0f}s depuis premiere transaction"
    )


@tool("decision_engine")
def decision_engine(score_result: str) -> str:
    """
    Prend le resultat du fraud_score_calculator et decide BLOQUE / A_VERIFIER / VALIDE.
    """
    try:
        # Extraction du score - version plus robuste
        import re
        
        # Chercher SCORE_RISQUE: suivi d'un nombre
        score_match = re.search(r"SCORE_RISQUE:\s*([\d.]+)", score_result)
        if not score_match:
            # Alternative: chercher juste le nombre apres SCORE_RISQUE
            score_match = re.search(r"SCORE_RISQUE[:\s]*([\d.]+)", score_result)
            
        if not score_match:
            return f"ERREUR: Score introuvable dans l'entree. Entree recue: {score_result[:200]}"

        score = float(score_match.group(1))

        # Regles de decision
        if score > 0.7:
            decision = "BLOQUE"
            confiance = 0.95
            action = "Bloquer la carte. Contacter le client immediatement."
        elif score > 0.3:
            decision = "A_VERIFIER"
            confiance = 0.70
            action = "Mettre en attente. Appeler le client pour confirmation."
        else:
            decision = "VALIDE"
            confiance = 0.90
            action = "Autoriser la transaction."

        result = (
            f"DECISION: {decision}\n"
            f"SCORE: {score:.4f}\n"
            f"CONFIANCE: {confiance:.2f}\n"
            f"ACTION: {action}"
        )
        return result

    except Exception as e:
        return f"ERREUR dans decision_engine: {str(e)}"