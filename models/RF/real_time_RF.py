import pandas as pd
import numpy as np
import random
import os
import joblib

from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from rich.console import Console
from rich.text import Text

random.seed(77)
np.random.seed(77)

CHECKPOINTS = {
    "draft": [],
    "10min": ['golddiffat10','void_grubs','opp_void_grubs','dragons','opp_dragons'],
    "15min": ['golddiffat10','golddiffat15','firstmidtower','firstherald','dragons','opp_dragons'],
    "20min": ['golddiffat10','golddiffat15','golddiffat20','firstmidtower','firstherald','dragons','opp_dragons','void_grubs','opp_void_grubs'],
    "25min": ['golddiffat10','golddiffat15','golddiffat20','golddiffat25','firstmidtower','firstherald','dragons','opp_dragons','atakhans','opp_atakhans'],
}

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


def compute_champ_freq(df):
    champs = pd.concat([
        df[[f'blue{i}' for i in range(1,6)]].stack(),
        df[[f'red{i}' for i in range(1,6)]].stack()
    ])
    return champs.value_counts().to_dict()


def gerar_features(df, model_w2v, numeric_cols, one_hot=None, champ_freq=None):
    X = []
    for idx, row in df.iterrows():
        def avg_emb(side):
            champs = [row[f'{side}{i}'] for i in range(1,6)]
            vecs, ws = [], []
            for c in champs:
                if c in model_w2v.wv:
                    vecs.append(model_w2v.wv[c])
                    ws.append(champ_freq.get(c,1))
            if not vecs:
                return np.zeros(model_w2v.vector_size)
            return np.average(np.stack(vecs), axis=0, weights=np.array(ws))
        emb_b = avg_emb('blue')
        emb_r = avg_emb('red')
        extras = row[numeric_cols].to_numpy(dtype=float) if numeric_cols else np.array([])
        league_feats = one_hot.loc[idx].to_numpy() if one_hot is not None else np.array([])
        X.append(np.concatenate([emb_b, emb_r, extras, league_feats]))
    return np.vstack(X)


def treinar_pipeline(df, chkpt, champ_freq):
    model_path = os.path.join(MODEL_DIR, f'model_{chkpt}.joblib')
    meta_path  = os.path.join(MODEL_DIR, f'meta_{chkpt}.joblib')
    if os.path.exists(model_path) and os.path.exists(meta_path):
        clf, scaler, thr, auc, acc = joblib.load(model_path)
        w2v, _, _, one_hot, _ = joblib.load(meta_path)
        return clf, w2v, scaler, CHECKPOINTS[chkpt], one_hot, thr, auc
    raise FileNotFoundError(f"Model for checkpoint {chkpt} not found. Execute the training script first.")


def prever_partida(model, model_w2v, scaler, numeric_cols, partida_dict, one_hot, thr, checkpoint, time_blue, time_red, auc):
    console = Console()
    df_nova = pd.DataFrame([partida_dict])
    df_nova.index = [0]

    if one_hot is not None:
        oh_nova = pd.get_dummies(df_nova['league']).reindex(columns=one_hot.columns, fill_value=0)
    else:
        oh_nova = None

    X_nova = gerar_features(df_nova, model_w2v, numeric_cols, oh_nova, compute_champ_freq(df_nova))
    if numeric_cols:
        si = 2 * model_w2v.vector_size
        X_nova[:, si:si+len(numeric_cols)] = scaler.transform(X_nova[:, si:si+len(numeric_cols)])

    prob_blue = model.predict_proba(X_nova)[0,1]
    prob_red  = 1 - prob_blue

    bar_blue = "█" * int(prob_blue * 20)
    bar_red  = "█" * int(prob_red * 20)
    odd_blue = round(1 / prob_blue, 2) if prob_blue > 0 else float('inf')
    odd_red  = round(1 / prob_red, 2)  if prob_red  > 0 else float('inf')
    console.print(f'    [green] === {checkpoint} ===  SEM FAVORITISMO = TEMPO REAL [/green]')
    console.print(f"🎯 [green]ROC AUC {round(auc*100,1)}%\n")
    console.print(
        f"[blue]{time_blue} [/blue][bold blue][{bar_blue:<20}][/bold blue] [cyan]{prob_blue:.1%}[/cyan]\n"
        f"[red]{time_red} [/red][bold red][{bar_red:<20}][/bold red] [bold red]{prob_red:.1%}[/bold red]\n"
    )
    console.print(f"[yellow]ODDS[/yellow]\n💰 [blue]{time_blue}→ {odd_blue}[/blue] \n💰 [red]{time_red}→ {odd_red}[/red]\n")


if __name__ == "__main__":
    df = pd.read_csv('data/dados_cada_partida.csv')
    champ_freq = compute_champ_freq(df)
    checkpoint = "10min"
    time_blue = 'tt '
    time_red  = 'edg'

    model, model_w2v, scaler, numeric_cols, one_hot, thr, auc = treinar_pipeline(df, checkpoint, champ_freq)


    nova_partida = {
        'blue1': "Rumble", 'blue2': 'Jhin', 'blue3': "Annie", 'blue4': "Alistar", 'blue5': "Trundle",
        'red1':  "Rakan", 'red2': 'Galio', 'red3': "Aurora", 'red4' : "Wukong", 'red5': "Varus",
        'league': 'LPL',
        'golddiffat10': -500.0, 'golddiffat15': 0.0, 'golddiffat20': 0.0, 'golddiffat25': 0.0,
        'firstmidtower': 0.5, 'firstherald': 0.5,
        'dragons': 1.0, 'opp_dragons': 0.0,
        'void_grubs': 0.0, 'opp_void_grubs': 0.0,
        'atakhans': 0.0, 'opp_atakhans': 0.0
    }

    prever_partida(
        model, model_w2v, scaler, numeric_cols,
        nova_partida, one_hot, thr, checkpoint,
        time_blue, time_red, auc
    )
