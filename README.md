# FraudShield — Système Multi-Agents de Détection de Fraude

Pipeline séquentiel à 3 agents spécialisés qui analyse chaque transaction suspecte en < 2 secondes.

```
Transaction → [Agent Scoreur] → [Agent Décision] → [Agent Explication] → Rapport analyste
```

---

##  Structure du projet

```
fraudshield/
├── agents/
│   ├── scorer_agent.py      # Agent 1 : calcule le score de risque (Random Forest)
│   ├── decision_agent.py    # Agent 2 : tranche BLOQUÉ / À_VÉRIFIER / VALIDÉ
│   └── explanation_agent.py # Agent 3 : rédige le rapport pour l'analyste
├── utils/
│   └── tools.py             # Outils CrewAI partagés (fraud_score_calculator, decision_engine)
├── data/
│   └── creditcard.csv       # Dataset Kaggle (284k transactions)
├── models/                  # Créé automatiquement par train_model.py
│   ├── rf_fraud.joblib
│   └── scaler.joblib
├── train_model.py           # Entraîne le Random Forest (à faire 1 seule fois)
├── main.py                  # Point d'entrée principal
├── requirements.txt
└── .env.example
```

---

##  Installation (VS Code, pas de GPU requis)

### Étape 1 — Créer l'environnement Python

```bash
# Dans le terminal VS Code (Ctrl + `)
python -m venv venv

# Windows :
venv\Scripts\activate

# Mac / Linux :
source venv/bin/activate
```

### Étape 2 — Installer les dépendances

```bash
pip install -r requirements.txt
```


### Étape 3 — Configurer le LLM

**Option A — OpenAI (recommandée si vous avez une clé) :**

```bash
cp .env.example .env
# Éditez .env et décommentez : OPENAI_API_KEY=sk-votre-cle
```

**Option B — LLM local Bonsai 1.7B (zéro coût, CPU only) :**

1. Télécharger llama-server : https://github.com/ggerganov/llama.cpp/releases
2. Télécharger le modèle : https://huggingface.co/bartowski/Bonsai-1.7B-GGUF
3. Lancer dans un terminal séparé :
   ```bash
   ./llama-server -m bonsai-1.7b-q4_k_m.gguf --port 8080 --ctx-size 2048
   ```
4. Dans `.env` : `LOCAL_LLM_URL=http://localhost:8080`

### Étape 4 — Entraîner le modèle (1 seule fois)

```bash
python train_model.py
```

---

##  Lancer le projet

### Mode démo (3 transactions de test — recommandé pour la présentation)

```bash
python main.py
```

### Mode batch (analyser N transactions réelles du CSV)

```bash
python main.py --batch 5
```

### Mode interactif (coller un JSON de transaction)

```bash
python main.py --live
```

---

##  Métriques attendues

| Métrique | Valeur cible | Valeur typique |
|---|---|---|
| AUC-ROC | > 0.95 | ~0.97-0.99 |
| Taux de détection (recall fraude) | > 85% | ~92% |
| Temps d'analyse par transaction | < 2s (scoring) | 0.1s |
| Temps total pipeline (avec LLM) | < 30s | 15-25s |

---

##  Architecture multi-agents

### Pourquoi 3 agents et pas 1 seul LLM ?

Un LLM seul ne peut pas naturellement se contredire lui-même. Ici :

- **Agent Scoreur** → spécialiste quantitatif (données, modèle ML)
- **Agent Décision** → spécialiste règles métier (seuils, conformité)
- **Agent Explication** → spécialiste communication (rapport opérationnel)

Chaque agent a un `backstory` différent qui oriente ses réponses vers sa spécialité.

### Patterns de coordination utilisés

- **Task parallelism** : les 3 agents peuvent tourner en parallèle (ici séquentiel car chaque agent dépend du précédent)
- **Context passing** : `context=[task_score, task_decision]` passe les outputs en chaîne
- **Agent specialization** : `backstory` + `tools` différents par agent

---

##  Choix techniques

| Choix | Justification |
|---|---|
| Random Forest vs LLM pour le scoring | RF : déterministe, rapide (0.1s), pas de GPU, AUC > 0.97 |
| SMOTE | Dataset déséquilibré (0.17% de fraudes) → SMOTE rééquilibre |
| temperature=0.1 | Réponses déterministes pour les décisions critiques |
| n_estimators=100 | Bon équilibre précision/vitesse sur CPU |

---

##  FAQ / Problèmes courants

**`FileNotFoundError: Modèle introuvable`**
→ Lance d'abord `python train_model.py`

**`Connection refused` pour le LLM local**
→ Vérifie que llama-server tourne sur le port 8080

**`ModuleNotFoundError: crewai`**
→ Vérifie que ton venv est activé : `venv\Scripts\activate` (Windows)

**`MemoryError` pendant train_model.py**
→ Normal sur machines avec < 4GB RAM. Réduire `n_estimators=50` dans `train_model.py`
