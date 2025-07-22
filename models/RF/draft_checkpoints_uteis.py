import pandas as pd
import numpy as np
import random
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from gensim.models import Word2Vec

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns

random.seed(77)
np.random.seed(77)

CHECKPOINTS = {
    "draft": [],
    "10min": ['golddiffat10', 'void_grubs', 'opp_void_grubs', 'dragons', 'opp_dragons'],
    "15min": ['golddiffat10', 'golddiffat15', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons'],
    "20min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'void_grubs', 'opp_void_grubs'],
    "25min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'golddiffat25', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'atakhans', 'opp_atakhans'],
}

def compute_champ_freq(df):
    champs_cols = [f'blue{i}' for i in range(1,6)] + [f'red{i}' for i in range(1,6)]
    all_champs = df[champs_cols].stack()
    return all_champs.value_counts().to_dict()

def treinar_word2vec(df):
    champs_cols = [f'blue{i}' for i in range(1,6)] + [f'red{i}' for i in range(1,6)]
    sentences = [list(row) for row in df[champs_cols].itertuples(index=False)]  # <-- corrigido aqui
    model = Word2Vec(sentences=sentences, vector_size=128, window=10, min_count=5, workers=4, sg=1, epochs=100, seed=77)
    return model

def gerar_features(df, model_w2v, numeric_cols, league_encoding=None, champ_freq=None):
    X = []
    for idx, row in df.iterrows():
        def emb(champs):
            vecs, ws = [], []
            for champ in champs:
                if champ in model_w2v.wv:
                    vecs.append(model_w2v.wv[champ])
                    freq = champ_freq.get(champ, 1)
                    ws.append(np.log1p(freq))  # Peso suavizado
            return np.average(vecs, axis=0, weights=ws) if vecs else np.zeros(model_w2v.vector_size)

        emb_blue = emb([row[f'blue{i}'] for i in range(1, 6)])
        emb_red  = emb([row[f'red{i}'] for i in range(1, 6)])
        emb_diff = emb_blue - emb_red  # 🆕 diferença entre composições

        extras = row[numeric_cols].to_numpy(dtype=float) if numeric_cols else np.array([])
        league_feats = np.array([league_encoding.get(row['league'], 0.5)]) if league_encoding is not None else np.array([])

        full_vector = np.concatenate([emb_blue, emb_red, emb_diff, extras, league_feats])
        X.append(full_vector)

    return np.vstack(X)



def apply_dynamic_threshold(y_true, y_proba):
    thresholds = np.linspace(0.3, 0.7, 21)
    best_thr = max(((thr, f1_score(y_true, (y_proba >= thr).astype(int))) for thr in thresholds), key=lambda x: x[1])[0]
    return best_thr

def ajustar_variaveis_binarias(df):
    for col in ['firstmidtower', 'firstherald']:
        df[col] = df[col].map({1.0: 1.0, 0.0: -1.0, 0.5: 0.0}).fillna(0.0)
    return df

def treinar_pipeline(df, checkpoint, champ_freq=None, forcar_retreinamento=False):
    versao = "v2"  # ou "diff" se preferir
    model_path = f"models/model_{versao}_{checkpoint}.joblib"
    meta_path  = f"models/meta_{versao}_{checkpoint}.joblib"

    if not forcar_retreinamento and os.path.exists(model_path) and os.path.exists(meta_path):
        model, scaler, thr, auc, acc = joblib.load(model_path)
        w2v, _, _, league_encoding, _ = joblib.load(meta_path)
        return model, w2v, scaler, CHECKPOINTS[checkpoint], league_encoding, auc, acc

    df_cp = ajustar_variaveis_binarias(df.copy())
    numeric_cols = CHECKPOINTS[checkpoint]
    y = df_cp['result'].astype(int).to_numpy()

    league_encoding = None
    if checkpoint == 'draft':
        league_encoding = df_cp.groupby("league")["result"].mean().to_dict()

    champ_freq = compute_champ_freq(df_cp) if champ_freq is None else champ_freq
    model_w2v = treinar_word2vec(df_cp)

    X = gerar_features(df_cp, model_w2v, numeric_cols, league_encoding, champ_freq)

    scaler = StandardScaler()
    if numeric_cols:
        start = 3 * model_w2v.vector_size
        X[:, start:start+len(numeric_cols)] = scaler.fit_transform(X[:, start:start+len(numeric_cols)])

    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=77)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
    y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:,1]
    y_pred = (y_proba >= 0.5).astype(int)

    thr = apply_dynamic_threshold(y, y_proba)
    auc = roc_auc_score(y, y_proba)
    acc = accuracy_score(y, y_pred)

    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title(f"Matriz de Confusão - {checkpoint} (AUC: {auc:.2f})")
    plt.show()

    model.fit(X, y)
    joblib.dump((model, scaler, thr, auc, acc), model_path)
    joblib.dump((model_w2v, champ_freq, numeric_cols, league_encoding, auc), meta_path)
    return model, model_w2v, scaler, numeric_cols, league_encoding, auc, acc

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv('data/dados_cada_partida.csv')
    champ_freq = compute_champ_freq(df)

    # ✅ Força o reprocessamento completo (Word2Vec, modelo, scaler, etc.)
    forcar_retreinamento = True

    time_blue = 'DRX.CH'
    time_red = 'KT.CH '

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

    console = Console()
    panels = []
    total_weight = 0
    modelos = {}

    for cp in CHECKPOINTS:
        modelo = treinar_pipeline(df, cp, champ_freq, forcar_retreinamento)
        modelos[cp] = modelo
        total_weight += modelo[-2]  # AUC

    console.print('\n[cyan] == PROBABILIDADE DE VITÓRIA POR TEMPO DE JOGO - JOGO EVEN - SEM FAVORITISMO== [/cyan]\n')
    ensemble_prob = 0

    for cp, (model, w2v, scaler, num_cols, league_encoding, auc, acc) in modelos.items():
        league_encoding_pred = league_encoding if cp == "draft" else None
        X_nova = gerar_features(df_nova, w2v, num_cols, league_encoding_pred, champ_freq)
        if num_cols:
            start_idx = w2v.vector_size * 3
            X_nova[:, start_idx:start_idx+len(num_cols)] = scaler.transform(X_nova[:, start_idx:start_idx+len(num_cols)])

        prob_blue = model.predict_proba(X_nova)[0,1]
        prob_red  = 1 - prob_blue

        weight = auc / total_weight if total_weight > 0 else 0
        ensemble_prob += weight * prob_blue

        odd_blue = round(1/prob_blue,2) if prob_blue>0 else float('inf')
        odd_red  = round(1/prob_red,2)  if prob_red>0 else float('inf')
        bar_blue = "█" * int(prob_blue*20)
        bar_red  = "█" * int(prob_red*20)

        text = f"""
🎯[green] AUC CV {cp}:[/green] [cyan]{auc:.1%}[/cyan]
📈[green] Acurácia CV {cp}:[/green] [cyan]{acc:.1%}[/cyan]\n
[yellow]⚔ CONFRONTO:[/yellow] [blue]{time_blue}[/blue] vs [red]{time_red}[/red] | Checkpoint: [blue]{cp}[/blue]\n
📊 PROBABILIDADES: 
[blue]{time_blue}{" " * (10 - len(time_blue))}[{bar_blue:<20}] {prob_blue:.1%}[/blue]
[red]{time_red}{" " * (10 - len(time_red))}[{bar_red:<20}] {prob_red:.1%}[/red]\n
💰 [yellow]ODDS[/yellow]:
[blue]{time_blue} → {odd_blue}[/blue]
[red]{time_red} → {odd_red}[/red]"""
        panels.append(Panel(Text.from_markup(text), title=cp.upper()))

    console.print(Columns(panels))
    console.print(f"🤖 Ensemble ponderado {time_blue}: {ensemble_prob:.1%}")
    console.print(f"🤖 Ensemble ponderado {time_red}: {1 - ensemble_prob:.1%}")
