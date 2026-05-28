#  FraudShield — Système Multi-Agents de Détection de Fraude Bancaire

**Auteurs :** Wissal Mahboub & Hiba Hamdouni  


Pipeline séquentiel à 3 agents spécialisés qui analyse chaque transaction suspecte et génère un rapport structuré pour l'analyste fraude.

```
Transaction JSON → [Agent Scoreur] → [Agent Décision] → [Agent Explication] → Rapport analyste
                        ↓                   ↓
               fraud_score_calculator  decision_engine
               (Random Forest .joblib) (seuils métier)
```

---

## Structure du projet

```
fraudshield/
├── agents/
│   ├── scorer_agent.py       # Agent 1 : calcule le score de risque via Random Forest
│   ├── decision_agent.py     # Agent 2 : tranche BLOQUÉ / À_VÉRIFIER / VALIDÉ
│   └── explanation_agent.py  # Agent 3 : rédige le rapport structuré pour l'analyste
├── utils/
│   └── tools.py              # Outils CrewAI : fraud_score_calculator + decision_engine
├── data/
│   └── creditcard.csv        # Dataset Kaggle Credit Card Fraud (284 807 transactions)
├── models/                   # Créé automatiquement par train_model.py
│   ├── rf_fraud.joblib       # Modèle Random Forest entraîné
│   ├── scaler.joblib         # StandardScaler pour Amount et Time
│   ├── bonsai.exe            # Exécutable llama-server
│   └── qwen3-1.7b-q4_k_m.gguf  # Modèle LLM local Bonsai 1.7B (quantisé 4 bits)
├── train_model.py            # Entraîne le Random Forest (à faire 1 seule fois)
├── main.py                   # Point d'entrée principal
├── requirements.txt
└── .env.example
```

---

## Installation (pas de GPU requis)

### Étape 1 — Créer l'environnement Python

```bash
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

**Option A — LLM local Bonsai 1.7B (utilisée dans ce projet, zéro coût, CPU only) :**

Le modèle `qwen3-1.7b-q4_k_m.gguf` et l'exécutable `bonsai.exe` (llama-server) sont
placés dans le dossier `models/`. Lancer dans un terminal séparé :

```bash
# Windows
models\bonsai.exe -m models\qwen3-1.7b-q4_k_m.gguf --port 8080 --ctx-size 2048

# Mac / Linux
./models/bonsai -m models/qwen3-1.7b-q4_k_m.gguf --port 8080 --ctx-size 2048
```

Vérifier que le serveur tourne : http://localhost:8080

Dans `.env` :
```
LOCAL_LLM_URL=http://localhost:8080
```

**Option B — OpenAI (si vous avez une clé API) :**

```bash
cp .env.example .env

```

### Étape 4 — Télécharger le dataset

Télécharger `creditcard.csv` depuis [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
et le placer dans `data/creditcard.csv`.

### Étape 5 — Entraîner le modèle (1 seule fois)

```bash
python train_model.py
```

> Durée : ~2–4 minutes (CPU uniquement). Crée `models/rf_fraud.joblib` et `models/scaler.joblib`.  
> Résultat attendu : **AUC-ROC ≈ 0.974**, recall fraude ≈ 85%

---

## Lancer le projet

### Mode démo — 3 transactions de test (recommandé pour la présentation)

```bash
python main.py
```

Analyse automatiquement 3 transactions prédéfinies : une fraude, une ambiguë, une légitime.

### Mode batch — analyser N transactions réelles du CSV

```bash
python main.py --batch 5
```

### Mode interactif — coller un JSON de transaction manuellement

```bash
python main.py --live
```

---

## Métriques obtenues

| Métrique | Valeur obtenue |
|---|---|
| AUC-ROC (test set 20%) | 0.974 |
| Précision fraude | 0.96 |
| Recall fraude | 0.85 |
| F1-score fraude | 0.90 |
| LLM utilisé | Bonsai 1.7B (qwen3-1.7b-q4_k_m.gguf) |
| Inférence | CPU uniquement, llama-server local (port 8080) |
| Temperature | 0.1 (sorties déterministes) |

---

## Architecture multi-agents

### Les 3 agents et leurs rôles

| Agent | Rôle | Outil | Backstory |
|---|---|---|---|
| **Scoreur** | Analyste de Risque Quantitatif | `fraud_score_calculator` | Expert modélisation risque, 10 ans expérience, banques marocaines |
| **Décision** | Responsable Anti-Fraude | `decision_engine` | Responsable cellule anti-fraude, règles strictes, impact client |
| **Explication** | Expert Communication Risque | Aucun (contexte seul) | Spécialiste rapports fraude, 30 secondes de lecture max |

### Pourquoi 3 agents et pas 1 seul LLM ?

Un LLM seul ne peut pas naturellement diverger de lui-même ni garantir une décision déterministe.
Ici chaque agent a une responsabilité unique :

- **Agent Scoreur** → appelle directement `rf.predict_proba()` — aucune hallucination possible sur le score
- **Agent Décision** → applique des seuils codés en dur (0.3 / 0.7) — décision reproductible
- **Agent Explication** → utilise le LLM uniquement pour reformuler en langage naturel, là où le non-déterminisme n'a pas d'impact critique

### Patterns de coordination

- **Context passing** : `context=[task_score, task_decision]` injecte les outputs en chaîne dans chaque agent
- **Agent specialization** : `backstory` + `tools` différents par agent orientent le comportement du LLM
- **Tool isolation** : les outils Python déterministes (`fraud_score_calculator`, `decision_engine`) sont séparés du LLM
- **Pipeline séquentiel** : chaque agent attend le résultat du précédent — garantit la cohérence causale

### Seuils de décision

```
Score > 0.7  →  BLOQUÉ      (confiance 0.95) → Bloquer la carte, contacter le client
Score > 0.3  →  À_VÉRIFIER  (confiance 0.70) → Mettre en attente, appeler pour confirmation
Score ≤ 0.3  →  VALIDÉ      (confiance 0.90) → Autoriser la transaction
```

---

## Choix techniques

| Choix | Justification |
|---|---|
| Random Forest pour le scoring | Déterministe, rapide, AUC > 0.97, pas de GPU, interprétable via feature importances |
| SMOTE | Dataset très déséquilibré (0.17% de fraudes) — SMOTE génère des fraudes synthétiques |
| Bonsai 1.7B local | Zéro coût, zéro clé API, fonctionne entièrement en CPU |
| temperature=0.1 | Sorties structurées et déterministes pour les décisions critiques |
| n_estimators=100, max_depth=15 | Bon équilibre précision/vitesse sur CPU |
| class_weight='balanced' | Renforce la détection des fraudes minoritaires |

---

## Exemple de rapport généré

### Transaction frauduleuse (TX-DEMO-FRAUDE-60PCT)
RESUME EXECUTIF
Décision : A_VERIFIER
Score : 61.24%
Montant : 3450.00 EUR

ANALYSE DES SIGNAUX

V14 (-5.670, imp=20%) : Indique un comportement suspect avec des transactions répétées.

V4 (2.870, imp=10%) : Montant élevé en comparaison avec les normes.

V10 (-4.120, imp=10%) : Évite la confusion avec des transactions légales.

RECOMMANDATION ACTION
Vérifiez immédiatement le montant et les transactions associés.
Confirmez la conformité avec les règles de la banque.



---

