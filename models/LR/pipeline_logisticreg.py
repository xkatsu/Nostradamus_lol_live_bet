import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

# === Checkpoints com features numéricas disponíveis em cada momento ===
CHECKPOINTS = {
    "draft": [],  # só composição
    "10min": ['golddiffat10','void_grubs'],
    "15min": ['golddiffat10', 'golddiffat15', 'firsttower', 'firstmidtower', 'firstherald'],
    "20min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'firsttower', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'void_grubs'],
    "25min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'golddiffat25', 'firsttower', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'atakhans', 'opp_atakhans'],
}

CHAMP_COLS = ['blue1','blue2','blue3','blue4','blue5',
              'red1','red2','red3','red4','red5']

# === 1. Treina Word2Vec para composições (apenas uma vez para todo o dataset) ===
def treinar_word2vec(df):
    comps = []
    for _, row in df.iterrows():
        comps.append([row[f'blue{i}'] for i in range(1, 6)])
        comps.append([row[f'red{i}'] for i in range(1, 6)])
        comps.append([row[f'blue{i}'] for i in range(1, 6)] + [row[f'red{i}'] for i in range(1, 6)])
    model_w2v = Word2Vec(sentences=comps, vector_size=8, window=5, min_count=1, workers=4, sg=1, epochs=50, seed=77)
    return model_w2v

# === 2. Gera features: embeddings + features numéricas selecionadas, com normalização ===
def gerar_features(df, model_w2v, numeric_cols, scaler=None):
    X = []
    for _, row in df.iterrows():
        emb_blue = np.mean([model_w2v.wv[row[f'blue{i}']] for i in range(1, 6)], axis=0)
        emb_red = np.mean([model_w2v.wv[row[f'red{i}']] for i in range(1, 6)], axis=0)
        extras = row[numeric_cols].values.astype(float) if numeric_cols else np.array([])
        X.append(np.concatenate([emb_blue, emb_red, extras]))
    X = np.vstack(X)
    if scaler is not None and numeric_cols:
        X[:, -len(numeric_cols):] = scaler.transform(X[:, -len(numeric_cols):])
    return X

# === 3. Treina pipeline para um checkpoint específico ===
def treinar_pipeline(df, checkpoint, model_w2v):
    numeric_cols = CHECKPOINTS[checkpoint]

    # Cópia para evitar sobrescrever dados originais
    df_copy = df.copy()

    scaler = None
    if numeric_cols:
        scaler = StandardScaler()
        df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])

    X = gerar_features(df_copy, model_w2v, numeric_cols, scaler)
    y = df_copy['result'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

    base_model = LogisticRegression(max_iter=1000, random_state=77)
    model = CalibratedClassifierCV(estimator=base_model, cv=5, method='sigmoid')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"\n=== Checkpoint: {checkpoint} ===")
    print(f"Acurácia: {acc:.2f}")
    print(f"ROC AUC: {roc:.2f}")

    return model, scaler, numeric_cols

# === 4. Prever resultado para nova partida no checkpoint escolhido ===
def prever_partida(model, model_w2v, scaler, numeric_cols, partida_dict):
    df_nova = pd.DataFrame([partida_dict])
    X_nova = gerar_features(df_nova, model_w2v, numeric_cols, scaler)
    prob_azul = model.predict_proba(X_nova)[0][1]
    prob_vermelho = 1 - prob_azul
    print(f"\nProbabilidade do time AZUL vencer: {prob_azul:.2%}")
    print(f"Probabilidade do time VERMELHO vencer: {prob_vermelho:.2%}")

# === Exemplo de uso ===
if __name__ == "__main__":
    df = pd.read_csv('dados_cada_partida.csv')

    # Treina Word2Vec uma única vez
    model_w2v = treinar_word2vec(df)

    # Escolha o checkpoint (momento do jogo) para treinar e prever
    checkpoint = "10min"  # pode ser "draft", "10min", "15min", "20min", "25min"

    model, scaler, numeric_cols = treinar_pipeline(df, checkpoint, model_w2v)

    nova_partida = {
        'blue1': 'Sett', 'blue2': 'Maokai', 'blue3': 'Tristana', 'blue4': "Varus", 'blue5': 'Rakan',
        'red1': 'Yorick', 'red2': 'Trundle', 'red3': 'Hwei', 'red4': 'Jhin', 'red5': "Bard",
        'golddiffat10': -5000.0, 'golddiffat15': 0.0, 'golddiffat20': 0.0, 'golddiffat25': 0.0,
        'firsttower': 0.0, 'firstmidtower': 0.0, 'firstherald': 0.0,
        'dragons': 0.0, 'opp_dragons': 0.0, 'void_grubs': 3.0,
        'atakhans': 0.0, 'opp_atakhans': 0.0,
    }

    prever_partida(model, model_w2v, scaler, numeric_cols, nova_partida)
