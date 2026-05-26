import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


#  CONFIGURATION LLM


def get_llm():
    from dotenv import load_dotenv
    load_dotenv()
    from crewai import LLM

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        console.print("[green]LLM: OpenAI GPT-3.5-turbo[/green]")
        return LLM(model="gpt-3.5-turbo", api_key=openai_key, temperature=0.1)

    # Ollama (si installe localement)
    ollama_model = os.getenv("OLLAMA_MODEL", "")
    if ollama_model:
        console.print(f"[yellow]LLM: Ollama ({ollama_model})[/yellow]")
        return LLM(model=f"ollama/{ollama_model}", base_url="http://localhost:11434", temperature=0.1)

    # llama-server (Bonsai 1.7B ou autre)
    local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:8080")
    console.print(f"[yellow]LLM: llama-server local ({local_url})[/yellow]")
    return LLM(
        model="openai/local-model",
        base_url=f"{local_url}/v1",
        api_key="no-key-needed",
        temperature=0.1,
        max_tokens=800,
    )



#  MODE TOOLS-ONLY : pipeline sans LLM


def analyze_tools_only(transaction: dict) -> dict:
    """
    Pipeline direct sans LLM : Scoreur -> Decision -> Rapport genere en Python.
    Resultat identique, ~0.2s, aucun serveur LLM requis.
    """
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from utils.tools import fraud_score_calculator, decision_engine

    start = time.time()
    tx_id = transaction.get("transaction_id", "TX-???")
    tx_copy = {k: v for k, v in transaction.items() if k != "transaction_id"}

    score_out = fraud_score_calculator.run(json.dumps(tx_copy))
    decision_out = decision_engine.run(score_out)

    # Extraction des valeurs cles pour le rapport
    import re
    score_match = re.search(r"SCORE_RISQUE:\s*([\d.]+)", score_out)
    score = float(score_match.group(1)) if score_match else 0.0
    decision_match = re.search(r"DECISION:\s*(\S+)", decision_out)
    decision = decision_match.group(1) if decision_match else "INCONNU"
    montant_match = re.search(r"MONTANT:\s*([\d.]+)", score_out)
    montant = montant_match.group(1) if montant_match else "?"
    features_match = re.search(r"TOP_FEATURES:\s*(.+)", score_out)
    features = features_match.group(1) if features_match else ""

    # Rapport genere par regles (sans LLM)
    if "BLOQUE" in decision or "BLOQUE" in decision.upper():
        resume = f"Transaction BLOQUEE -- Score de fraude tres eleve ({score*100:.1f}%). Montant : {montant} EUR."
        reco = "Bloquer la carte immediatement. Contacter le client pour verification d'identite. Ouvrir un dossier fraude."
    elif "A_VERIFIER" in decision or "VERIFIER" in decision.upper():
        resume = f"Transaction A VERIFIER -- Score ambigu ({score*100:.1f}%). Montant : {montant} EUR."
        reco = "Mettre la transaction en attente (max 15 min). Appeler le client pour confirmer. Si non joignable -> bloquer."
    else:
        resume = f"Transaction VALIDEE -- Profil de risque normal ({score*100:.1f}%). Montant : {montant} EUR."
        reco = "Autoriser la transaction. Aucune action requise."

    top_f = [f.strip() for f in features.split(",")][:3]
    signaux = "\n".join(f"  - {f}" for f in top_f)

    rapport = (
        f"## RESUME EXECUTIF\n{resume}\n\n"
        f"## ANALYSE DES SIGNAUX\n"
        f"Top features qui ont influence le score :\n{signaux}\n\n"
        f"## RECOMMANDATION ACTION\n{reco}"
    )

    elapsed = time.time() - start
    return {
        "transaction_id": tx_id,
        "elapsed_seconds": round(elapsed, 3),
        "final_report": rapport,
        "raw_tasks": {"score": score_out, "decision": decision_out},
        "decision": decision,
    }



#  MODE FULL : pipeline avec LLM (CrewAI)

def analyze_transaction(transaction: dict, llm) -> dict:
    """Pipeline complet via CrewAI (Scoreur -> Decision -> Explication LLM)."""
    from crewai import Crew, Task, Process
    from agents.scorer_agent import create_scorer_agent
    from agents.decision_agent import create_decision_agent
    from agents.explanation_agent import create_explanation_agent

    tx_json = json.dumps({k: v for k, v in transaction.items() if k != "transaction_id"})
    tx_id = transaction.get("transaction_id", "TX-???")
    start_time = time.time()

    scorer = create_scorer_agent(llm)
    decision_maker = create_decision_agent(llm)
    explainer = create_explanation_agent(llm)

    task_score = Task(
        description=(
            f"Calcule le score de risque de fraude pour cette transaction :\n{tx_json}\n\n"
            "Utilise l'outil fraud_score_calculator avec ce JSON exact comme argument. "
            "Retourne le resultat complet de l'outil sans modification."
        ),
        expected_output="Texte contenant SCORE_RISQUE: X.XXXX, TOP_FEATURES, MONTANT et TEMPS.",
        agent=scorer,
    )

    task_decision = Task(
        description=(
            "Utilise le resultat de l'agent Scoreur. Appelle l'outil decision_engine "
            "en passant le texte recu comme argument score_result. "
            "Retourne le resultat complet."
        ),
        expected_output="Texte contenant DECISION: (VALIDE|A_VERIFIER|BLOQUE), SCORE, CONFIANCE, ACTION.",
        agent=decision_maker,
        context=[task_score],
    )

    task_explain = Task(
        description=(
            f"Tu es l'Agent Explication pour la transaction {tx_id}.\n"
            "En utilisant les resultats du Scoreur et de la Decision fournis en contexte, "
            "redige un rapport structure EXACTEMENT en 3 sections :\n\n"
            "## RESUME EXECUTIF\n"
            "[1-2 phrases : decision + score en % + montant]\n\n"
            "## ANALYSE DES SIGNAUX\n"
            "[3 bullets : quelles features ont declenche l'alerte et pourquoi]\n\n"
            "## RECOMMANDATION ACTION\n"
            "[Action immediate pour l'analyste]\n\n"
            "Sois concis et oriente action. Max 30 secondes de lecture."
        ),
        expected_output="Rapport structure en 3 sections : RESUME EXECUTIF, ANALYSE DES SIGNAUX, RECOMMANDATION ACTION.",
        agent=explainer,
        context=[task_score, task_decision],
    )

    crew = Crew(
        agents=[scorer, decision_maker, explainer],
        tasks=[task_score, task_decision, task_explain],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    elapsed = time.time() - start_time

    decision_text = str(task_decision.output) if task_decision.output else ""
    return {
        "transaction_id": tx_id,
        "elapsed_seconds": round(elapsed, 2),
        "final_report": str(result),
        "raw_tasks": {
            "score": str(task_score.output) if task_score.output else "",
            "decision": decision_text,
        },
        "decision": decision_text,
    }



#  AFFICHAGE


def display_result(result: dict):
    report = result["final_report"]
    tx_id = result["transaction_id"]
    elapsed = result["elapsed_seconds"]
    decision_raw = result.get("decision", "")

    if "BLOQUE" in decision_raw or "BLOQUE" in decision_raw.upper():
        color = "red"
        icon = "BLOCKED"
    elif "A_VERIFIER" in decision_raw or "VERIFIER" in decision_raw.upper():
        color = "yellow"
        icon = "WARNING"
    else:
        color = "green"
        icon = "OK"

    console.print(Panel(
        report,
        title=f"{icon} [bold {color}]FraudShield -- {tx_id}[/bold {color}]",
        subtitle=f"Analyse en {elapsed}s",
        border_style=color,
        padding=(1, 2),
    ))



#  DONNEES DE DEMONSTRATION


def get_demo_transactions():
    return [
        {
            "transaction_id": "TX-DEMO-FRAUDE",
            "Time": 406, "Amount": 2125.87,
            "V1": -3.04, "V2": 2.14, "V3": -4.89, "V4": 3.21, "V5": -2.11,
            "V6": -1.98, "V7": -4.31, "V8": 0.87, "V9": -2.54, "V10": -4.78,
            "V11": 2.01, "V12": -5.43, "V13": 0.32, "V14": -6.12, "V15": 0.11,
            "V16": -1.98, "V17": -6.43, "V18": -2.34, "V19": 0.56, "V20": 0.23,
            "V21": 0.87, "V22": -0.98, "V23": 0.12, "V24": -0.54, "V25": 0.34,
            "V26": -0.21, "V27": 0.65, "V28": 0.11,
        },
        {
            "transaction_id": "TX-DEMO-AMBIGU",
            "Time": 54832, "Amount": 347.50,
            "V1": -0.82, "V2": 0.54, "V3": -1.23, "V4": 0.87, "V5": -0.43,
            "V6": -0.56, "V7": -0.98, "V8": 0.21, "V9": -0.67, "V10": -1.12,
            "V11": 0.54, "V12": -1.43, "V13": 0.11, "V14": -1.78, "V15": 0.09,
            "V16": -0.45, "V17": -1.23, "V18": -0.67, "V19": 0.23, "V20": 0.12,
            "V21": 0.34, "V22": -0.21, "V23": 0.05, "V24": -0.15, "V25": 0.09,
            "V26": -0.07, "V27": 0.18, "V28": 0.04,
        },
        {
            "transaction_id": "TX-DEMO-LEGITIME",
            "Time": 86400, "Amount": 42.30,
            "V1": 1.23, "V2": 0.18, "V3": 0.87, "V4": 0.43, "V5": 0.12,
            "V6": 0.34, "V7": 0.21, "V8": 0.09, "V9": 0.15, "V10": 0.07,
            "V11": 0.23, "V12": 0.11, "V13": 0.06, "V14": 0.18, "V15": 0.04,
            "V16": 0.09, "V17": 0.12, "V18": 0.05, "V19": 0.08, "V20": 0.03,
            "V21": 0.07, "V22": 0.02, "V23": 0.04, "V24": 0.01, "V25": 0.03,
            "V26": 0.02, "V27": 0.01, "V28": 0.005,
        },
    ]


def get_random_transactions_from_csv(n: int) -> list:
    data_path = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")
    df = pd.read_csv(data_path)
    n_fraud = min(max(1, n // 2), len(df[df["Class"] == 1]))
    frauds = df[df["Class"] == 1].sample(n_fraud, random_state=42)
    legit = df[df["Class"] == 0].sample(n - n_fraud, random_state=42)
    sample = pd.concat([frauds, legit]).sample(frac=1, random_state=42)
    txs = []
    for i, (_, row) in enumerate(sample.iterrows()):
        tx = row.drop("Class").to_dict()
        label = "FRAUDE" if row["Class"] == 1 else "LEGIT"
        tx["transaction_id"] = f"TX-CSV-{label}-{i+1:03d}"
        txs.append(tx)
    return txs



#  ENTRY POINT


def main():
    parser = argparse.ArgumentParser(description="FraudShield -- Detection de fraude multi-agents")
    parser.add_argument("--tools-only", action="store_true",
                        help="Mode rapide sans LLM (scoring + decision Python, aucun serveur requis)")
    parser.add_argument("--live", action="store_true", help="Mode interactif (saisie JSON)")
    parser.add_argument("--batch", type=int, default=0, help="Analyser N transactions du CSV")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]FraudShield[/bold cyan] -- Systeme Multi-Agents de Detection de Fraude\n"
        "[dim]Stack: CrewAI 1.14 + Random Forest + 3 Agents specialises[/dim]",
        border_style="cyan"
    ))

    # Verification modele
    model_path = os.path.join(os.path.dirname(__file__), "models", "rf_fraud.joblib")
    if not os.path.exists(model_path):
        console.print("[red]Modele non trouve. Lance d'abord :[/red]")
        console.print("   [yellow]python train_model.py[/yellow]")
        sys.exit(1)

    # Choisir le mode d'analyse
    if args.tools_only:
        console.print("[dim]Mode : Tools-Only (sans LLM, ultra-rapide)[/dim]")
        analyze_fn = analyze_tools_only
    else:
        llm = get_llm()
        analyze_fn = lambda tx: analyze_transaction(tx, llm)

    if args.live:
        console.print("\n[cyan]Mode interactif -- Colle un JSON de transaction (Ctrl+C pour quitter)[/cyan]")
        while True:
            try:
                raw = input("\n> JSON transaction : ")
                tx = json.loads(raw)
                if "transaction_id" not in tx:
                    tx["transaction_id"] = f"TX-LIVE-{int(time.time())}"
                result = analyze_fn(tx)
                display_result(result)
            except KeyboardInterrupt:
                console.print("\n[dim]Au revoir.[/dim]")
                break
            except json.JSONDecodeError:
                console.print("[red]JSON invalide.[/red]")

    elif args.batch > 0:
        console.print(f"\n[cyan]Analyse de {args.batch} transactions depuis creditcard.csv...[/cyan]")
        transactions = get_random_transactions_from_csv(args.batch)
        results_summary = []
        for tx in transactions:
            console.print(f"\n[dim]--> Analyse {tx['transaction_id']}...[/dim]")
            result = analyze_fn(tx)
            display_result(result)
            results_summary.append(result)

        table = Table(title="Recapitulatif", border_style="cyan")
        table.add_column("Transaction ID", style="cyan")
        table.add_column("Decision", justify="center")
        table.add_column("Temps (s)", justify="right")
        for r in results_summary:
            d = r.get("decision", "")
            dec = "BLOQUE" if "BLOQUE" in d or "BLOQUE" in d.upper() else ("A_VERIFIER" if "VERIFIER" in d or "VERIFIER" in d.upper() else "VALIDE")
            table.add_row(r["transaction_id"], dec, str(r["elapsed_seconds"]))
        avg = sum(r["elapsed_seconds"] for r in results_summary) / len(results_summary)
        table.add_row("[bold]MOYENNE[/bold]", "", f"[bold]{avg:.2f}[/bold]")
        console.print(table)

    else:
        console.print("\n[cyan]Mode demo -- 3 transactions : fraude / ambigue / legitime[/cyan]")
        for tx in get_demo_transactions():
            console.print(f"\n[dim]--> Analyse {tx['transaction_id']}...[/dim]")
            result = analyze_fn(tx)
            display_result(result)


if __name__ == "__main__":
    main()