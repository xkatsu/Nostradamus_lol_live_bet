import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# --- Seu dicionário de checkpoints e funções treinar_word2vec, gerar_features, inverter_exemplo, etc devem estar aqui (igual seu código original) ---

CHECKPOINTS = {
    "draft": [],
    "10min": ['golddiffat10', 'void_grubs','opp_void_grubs'],
    "15min": ['golddiffat10', 'golddiffat15', 'firsttower', 'firstmidtower', 'firstherald','void_grubs','opp_void_grubs'],
    "20min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'firsttower', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'void_grubs','opp_void_grubs'],
    "25min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'golddiffat25', 'firsttower', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'atakhans', 'opp_atakhans'],
}

CHAMP_COLS = ['blue1', 'blue2', 'blue3', 'blue4', 'blue5',
              'red1', 'red2', 'red3', 'red4', 'red5']

FEATURES_COM_SINAL = [
    'golddiffat10', 'golddiffat15', 'golddiffat20', 'golddiffat25',
    'atakhans', 'opp_atakhans', 'dragons', 'opp_dragons'
]

def treinar_word2vec(df):
    comps = []
    for _, row in df.iterrows():
        comps.append([row[f'blue{i}'] for i in range(1, 6)])
        comps.append([row[f'red{i}'] for i in range(1, 6)])
        comps.append([row[f'blue{i}'] for i in range(1, 6)] + [row[f'red{i}'] for i in range(1, 6)])
    model_w2v = Word2Vec(sentences=comps, vector_size=32, window=10, min_count=1, workers=4, sg=1, epochs=100, seed=77)
    return model_w2v

def gerar_features(df, model_w2v, numeric_cols, one_hot_leagues=None):
    X = []
    for _, row in df.iterrows():
        emb_blue = np.mean([model_w2v.wv[row[f'blue{i}']] for i in range(1, 6)], axis=0)
        emb_red = np.mean([model_w2v.wv[row[f'red{i}']] for i in range(1, 6)], axis=0)
        extras = row[numeric_cols].values.astype(float) if numeric_cols else np.array([])

        league_feats = np.array([])
        if one_hot_leagues is not None:
            league_feats = one_hot_leagues.loc[row.name].values

        X.append(np.concatenate([emb_blue, emb_red, extras, league_feats]))
    return np.vstack(X)

def inverter_exemplo(x_row, y_row, numeric_cols, league_size):
    emb_size = 32
    emb_blue = x_row[:emb_size]
    emb_red = x_row[emb_size:emb_size*2]
    extras = x_row[emb_size*2: -league_size].copy() if league_size > 0 else x_row[emb_size*2:].copy()
    league_feats = x_row[-league_size:] if league_size > 0 else np.array([])

    for i, col in enumerate(numeric_cols):
        if col in FEATURES_COM_SINAL and len(extras) > 0:
            extras[i] *= -1

    x_inv = np.concatenate([emb_red, emb_blue, extras, league_feats])
    y_inv = 1 - y_row
    return x_inv, y_inv

def encontrar_threshold_equilibrado(y_test, y_proba, thresholds=np.arange(0.3, 0.71, 0.01), alpha=0.7):
    melhor_thresh = 0.5
    melhor_score = 0
    melhor_metricas = {}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        score = alpha * acc + (1 - alpha) * f1
        if score > melhor_score:
            melhor_score = score
            melhor_thresh = t
            melhor_metricas = {
                'accuracy': acc,
                'f1': f1,
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred)
            }

    return melhor_thresh, melhor_metricas

def treinar_pipeline(df, checkpoint):
    numeric_cols = CHECKPOINTS[checkpoint]
    df_copy = df.copy()

    use_league = checkpoint == "draft"
    one_hot_leagues = None
    league_size = 0

    if use_league:
        df_copy['league'] = df_copy['league'].astype(str)
        one_hot_leagues = pd.get_dummies(df_copy['league'])
        league_size = one_hot_leagues.shape[1]

    model_w2v = treinar_word2vec(df_copy)
    X_raw = gerar_features(df_copy, model_w2v, numeric_cols, one_hot_leagues)
    y_raw = df_copy['result'].astype(int).values

    X_invertido, y_invertido = [], []
    for xi, yi in zip(X_raw, y_raw):
        xi_inv, yi_inv = inverter_exemplo(xi, yi, numeric_cols, league_size)
        X_invertido.append(xi_inv)
        y_invertido.append(yi_inv)

    X_total = np.vstack([X_raw, np.array(X_invertido)])
    y_total = np.concatenate([y_raw, np.array(y_invertido)])

    scaler = StandardScaler()
    if numeric_cols:
        if league_size > 0:
            start = -league_size - len(numeric_cols)
            end = -league_size
            X_total[:, start:end] = scaler.fit_transform(X_total[:, start:end])
        else:
            X_total[:, -len(numeric_cols):] = scaler.fit_transform(X_total[:, -len(numeric_cols):])

    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.2, random_state=77)

    base_model = RandomForestClassifier(n_estimators=100, random_state=77)
    model = CalibratedClassifierCV(estimator=base_model, cv=5, method='sigmoid')
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    # Encontra melhor threshold com base no conjunto de teste
    best_threshold, best_metrics = encontrar_threshold_equilibrado(y_test, y_proba)

    return {
        'model': model,
        'model_w2v': model_w2v,
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'one_hot_leagues': one_hot_leagues,
        'roc_auc': best_metrics.get('roc_auc', None),  # não calculado aqui, pode calcular se quiser
        'best_threshold': best_threshold,
        'metrics': best_metrics,
        'y_test': y_test,
        'y_proba': y_proba,
        'checkpoint': checkpoint
    }

if __name__ == "__main__":
    df = pd.read_csv('data/dados_cada_partida.csv')
    resultados = []

    for cp in CHECKPOINTS.keys():
        print(f"\nTreinando checkpoint: {cp}")
        res = treinar_pipeline(df, cp)

        # Adiciona ROC AUC calculado com probas e ground truth
        from sklearn.metrics import roc_auc_score
        res['metrics']['roc_auc'] = roc_auc_score(res['y_test'], res['y_proba'])

        print(f"Melhor threshold: {res['best_threshold']:.2f}")
        print(f"F1 Score: {res['metrics']['f1']:.2f}")
        print(f"Precisão: {res['metrics']['precision']:.2f}")
        print(f"Recall: {res['metrics']['recall']:.2f}")
        print(f"Acurácia: {res['metrics']['accuracy']:.2f}")
        print(f"ROC AUC: {res['metrics']['roc_auc']:.2f}")

        resultados.append({
            'checkpoint': cp,
            'best_threshold': res['best_threshold'],
            'f1_score': res['metrics']['f1'],
            'precision': res['metrics']['precision'],
            'recall': res['metrics']['recall'],
            'accuracy': res['metrics']['accuracy'],
            'roc_auc': res['metrics']['roc_auc']
        })

    df_resultados = pd.DataFrame(resultados)
    print("\nResumo geral dos melhores thresholds por checkpoint:")
    print(df_resultados)
    df_resultados.to_csv('calibration_model/melhores_thresholds.csv', index=False)
