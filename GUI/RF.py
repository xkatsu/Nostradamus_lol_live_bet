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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def run_prediction_interface(team_blue, team_red):
    CHECKPOINTS = {
        "draft": [],
        "10min": ['golddiffat10', 'void_grubs', 'opp_void_grubs', 'dragons', 'opp_dragons'],
        "15min": ['golddiffat10', 'golddiffat15', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons'],
        "20min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'void_grubs', 'opp_void_grubs'],
        "25min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'golddiffat25', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'atakhans', 'opp_atakhans'],
    }

    def load_model(cp):
        model, scaler, thr, auc, acc = joblib.load(f"models/model_{cp}.joblib")
        w2v, champ_freq, numeric_cols, one_hot, _ = joblib.load(f"models/meta_{cp}.joblib")
        return model, w2v, scaler, numeric_cols, one_hot, auc, acc, champ_freq

    def gerar_features(df, model_w2v, numeric_cols, one_hot_leagues=None, champ_freq=None):
        X = []
        for idx, row in df.iterrows():
            def emb(champs):
                vecs, ws = [], []
                for champ in champs:
                    if champ in model_w2v.wv:
                        vecs.append(model_w2v.wv[champ])
                        ws.append(champ_freq.get(champ, 1))
                return np.average(vecs, axis=0, weights=ws) if vecs else np.zeros(model_w2v.vector_size)
            emb_blue = emb([row[f'blue{i}'] for i in range(1,6)])
            emb_red  = emb([row[f'red{i}'] for i in range(1,6)])
            extras = row[numeric_cols].to_numpy(dtype=float) if numeric_cols else np.array([])
            league_feats = one_hot_leagues.loc[idx].to_numpy() if one_hot_leagues is not None else np.array([])
            X.append(np.concatenate([emb_blue, emb_red, extras, league_feats]))
        return np.vstack(X)

    df_nova = pd.DataFrame([{  # Dados placeholders
        'blue1': "Pantheon", 'blue2': 'Rumble', 'blue3': 'Azir', 'blue4': "Neeko", 'blue5': 'Ezreal',
        'red1':  "Taliyah", 'red2': 'Poppy', 'red3': "Aurora", 'red4' : "Corki", 'red5': "Nautilus",
        'league': '',
        'golddiffat10': 0.0, 'golddiffat15': 0.0, 'golddiffat20': 0.0, 'golddiffat25': 0.0,
        'firstmidtower': 0.5, 'firstherald': 0.5,
        'dragons': 0.0, 'opp_dragons': 0.0,
        'void_grubs': 0.0, 'opp_void_grubs': 0.0,
        'atakhans': 0.0, 'opp_atakhans': 0.0
    }])

    total_weight = 0
    modelos = {}
    for cp in CHECKPOINTS:
        modelo = load_model(cp)
        modelos[cp] = modelo
        total_weight += modelo[-3]  # AUC

    result_text = ""
    odds_data = []
    x_vals = []
    y_vals = []
    ensemble_prob = 0

    for cp, (model, w2v, scaler, num_cols, one_hot, auc, acc, champ_freq) in modelos.items():
        oh_nova = None
        if one_hot is not None:
            oh_nova = pd.get_dummies(df_nova['league'].astype(str))
            oh_nova = oh_nova.reindex(columns=one_hot.columns, fill_value=0)

        X_nova = gerar_features(df_nova, w2v, num_cols, oh_nova, champ_freq)
        if num_cols:
            start_idx = w2v.vector_size * 2
            X_nova[:, start_idx:start_idx+len(num_cols)] = scaler.transform(X_nova[:, start_idx:start_idx+len(num_cols)])

        prob_blue = model.predict_proba(X_nova)[0,1]
        prob_red  = 1 - prob_blue

        weight = auc / total_weight if total_weight > 0 else 0
        ensemble_prob += weight * prob_blue

        odd_blue = round(1/prob_blue,2) if prob_blue>0 else float('inf')
        odd_red  = round(1/prob_red,2)  if prob_red>0 else float('inf')

        result_text += f"""
⏱️ Checkpoint: {cp}
🔷 {team_blue}: {prob_blue:.1%} (Odds: {odd_blue})
🔺 {team_red}:  {prob_red:.1%} (Odds: {odd_red})
📊 AUC: {auc:.2%} | Accuracy: {acc:.2%}
"""

        odds_data.append((cp, prob_blue, prob_red, (odd_blue, odd_red)))
        x_vals.append(cp)
        y_vals.append(prob_blue)

    result_text += f"""
🤖 Ensemble Ponderado
🔷 {team_blue}: {ensemble_prob:.1%}
🔺 {team_red}:  {1 - ensemble_prob:.1%}
"""

    return result_text.strip(), odds_data, x_vals, y_vals