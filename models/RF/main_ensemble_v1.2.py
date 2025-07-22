import os
import joblib
import numpy as np
import pandas as pd

from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

from utils import gerar_features

# Definição dos checkpoints
CHECKPOINTS = {
    "draft": [],
    "10min": ['golddiffat10', 'void_grubs', 'opp_void_grubs', 'dragons', 'opp_dragons'],
    "15min": ['golddiffat10', 'golddiffat15', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons'],
    "20min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'void_grubs', 'opp_void_grubs'],
    "25min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'golddiffat25', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'atakhans', 'opp_atakhans'],
}

nova_partida = {
    'blue1': "Ambessa", 'blue2': 'Xin Zhao', 'blue3': "Taliyah", 'blue4': "Varus", 'blue5': "Renata Glasc",
    'red1':  "Rumble", 'red2': 'Vi', 'red3': "Sylas", 'red4' : "Senna", 'red5': "Braum",
    'league': '',
    'golddiffat10': 0.0, 'golddiffat15': 0.0, 'golddiffat20': 0.0, 'golddiffat25': 0.0,
    'firstmidtower': 0.5, 'firstherald': 0.5,
    'dragons': 0.0, 'opp_dragons': 0.0,
    'void_grubs': 0.0, 'opp_void_grubs': 0.0,
    'atakhans': 0.0, 'opp_atakhans': 0.0
}
df_nova = pd.DataFrame([nova_partida])

# Exibição
console = Console()
panels = []
total_weight = 0
ensemble_prob = 0

for cp in CHECKPOINTS:
    versao = "v2" if cp == "draft" else "v1"
    model_path = f"models/model_{versao}_{cp}.joblib"
    meta_path  = f"models/meta_{versao}_{cp}.joblib"

    model, scaler, thr, auc, acc = joblib.load(model_path)
    model_w2v, champ_freq, num_cols, league_encoding, _ = joblib.load(meta_path)

    league_encoding_pred = league_encoding if cp == "draft" else None
    X_nova = gerar_features(df_nova, model_w2v, num_cols, league_encoding_pred, champ_freq)

    if num_cols:
        start_idx = model_w2v.vector_size * (3 if cp == "draft" else 2)
        X_nova[:, start_idx:start_idx+len(num_cols)] = scaler.transform(X_nova[:, start_idx:start_idx+len(num_cols)])

    prob_blue = model.predict_proba(X_nova)[0, 1]
    prob_red = 1 - prob_blue
    weight = auc
    total_weight += weight
    ensemble_prob += weight * prob_blue

    odd_blue = round(1 / prob_blue, 2) if prob_blue > 0 else float('inf')
    odd_red = round(1 / prob_red, 2) if prob_red > 0 else float('inf')
    bar_blue = "█" * int(prob_blue * 20)
    bar_red = "█" * int(prob_red * 20)

    text = f"""
🎯[green] AUC CV {cp}:[/green] [cyan]{auc:.1%}[/cyan]
📈[green] Acurácia CV {cp}:[/green] [cyan]{acc:.1%}[/cyan]\n
[yellow]⚔ CONFRONTO:[/yellow] [blue]DRX.CH[/blue] vs [red]KT.CH[/red] | Checkpoint: [blue]{cp}[/blue]\n
📊 PROBABILIDADES: 
[blue]DRX.CH   [{bar_blue:<20}] {prob_blue:.1%}[/blue]
[red]KT.CH    [{bar_red:<20}] {prob_red:.1%}[/red]\n
💰 [yellow]ODDS[/yellow]:
[blue]DRX.CH → {odd_blue}[/blue]
[red]KT.CH  → {odd_red}[/red]"""
    panels.append(Panel(Text.from_markup(text), title=cp.upper()))

console.print(Columns(panels))
console.print(f"🤖 Ensemble ponderado DRX.CH: {ensemble_prob / total_weight:.1%}")
console.print(f"🤖 Ensemble ponderado KT.CH:  {(1 - ensemble_prob / total_weight):.1%}")
