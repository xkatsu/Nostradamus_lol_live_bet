import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

from utils import gerar_features

# Checkpoints definidos
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
    from gensim.models import Word2Vec
    champs_cols = [f'blue{i}' for i in range(1,6)] + [f'red{i}' for i in range(1,6)]
    sentences = [list(row) for row in df[champs_cols].itertuples(index=False)]
    model = Word2Vec(sentences=sentences, vector_size=128, window=10, min_count=5, workers=4, sg=1, epochs=100, seed=77)
    return model

def ajustar_variaveis_binarias(df):
    for col in ['firstmidtower', 'firstherald']:
        df[col] = df[col].map({1.0: 1.0, 0.0: -1.0, 0.5: 0.0}).fillna(0.0)
    return df

def apply_dynamic_threshold(y_true, y_proba):
    thresholds = np.linspace(0.3, 0.7, 21)
    best_thr = max(((thr, f1_score(y_true, (y_proba >= thr).astype(int))) for thr in thresholds), key=lambda x: x[1])[0]
    return best_thr

# ========== MAIN TREINAMENTO ==========

os.makedirs("models", exist_ok=True)
df = pd.read_csv("data/dados_cada_partida.csv")
df = ajustar_variaveis_binarias(df)
champ_freq = compute_champ_freq(df)

for cp in CHECKPOINTS:
    print(f"\n🔁 Treinando checkpoint: {cp}")
    versao = "v2" if cp == "draft" else "v1"

    model_path = f"models/model_{versao}_{cp}.joblib"
    meta_path  = f"models/meta_{versao}_{cp}.joblib"

    df_cp = df.copy()
    numeric_cols = CHECKPOINTS[cp]
    y = df_cp["result"].astype(int).to_numpy()

    league_encoding = None
    if cp == "draft":
        league_encoding = df_cp.groupby("league")["result"].mean().to_dict()

    model_w2v = treinar_word2vec(df_cp)
    X = gerar_features(df_cp, model_w2v, numeric_cols, league_encoding, champ_freq)

    # Escalonamento
    scaler = StandardScaler()
    if numeric_cols:
        start = model_w2v.vector_size * (3 if cp == "draft" else 2)
        X[:, start:start+len(numeric_cols)] = scaler.fit_transform(X[:, start:start+len(numeric_cols)])

    # Treinar modelo com CV
    model = RandomForestClassifier(n_estimators=100, random_state=77)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:,1]
    y_pred = (y_proba >= 0.5).astype(int)

    thr = apply_dynamic_threshold(y, y_proba)
    auc = roc_auc_score(y, y_proba)
    acc = accuracy_score(y, y_pred)

    print(f"✅ AUC: {auc:.3f} | Acurácia: {acc:.3f} | Threshold ótimo: {thr:.2f}")
    cm = confusion_matrix(y, y_pred)
    plt.figure()
    plt.title(f"Matriz de Confusão - {cp}")
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.savefig(f"models/matriz_{cp}.png")
    plt.close()

    # Fit final
    model.fit(X, y)
    joblib.dump((model, scaler, thr, auc, acc), model_path)
    joblib.dump((model_w2v, champ_freq, numeric_cols, league_encoding, auc), meta_path)
