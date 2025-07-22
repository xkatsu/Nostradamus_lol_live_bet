import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score


# ====================
# 1. Leitura e Pré-processamento
# ====================
df = pd.read_csv('dados_cada_partida.csv')


colunas_numericas = ['atakhans','opp_atakhans',
    'golddiffat25','golddiffat20', 'golddiffat15','golddiffat10', 'firstmidtower', 'firsttower',
    'firstherald', 'dragons','opp_dragons', 'void_grubs']

colunas_string = [
    'blue1','blue2','blue3','blue4','blue5',
    'red1','red2','red3','red4','red5'
]

scaler = StandardScaler()
df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

# ====================
# 2. Geração dos embeddings
# ====================
comps = []
for _, row in df.iterrows():
    comps.append([row[f'blue{i}'] for i in range(1, 6)])
    comps.append([row[f'red{i}'] for i in range(1, 6)])
    comps.append([row[f'blue{i}'] for i in range(1, 6)] + [row[f'red{i}'] for i in range(1, 6)])

model_w2v = Word2Vec(sentences=comps, vector_size=8, window=5, min_count=1, workers=4, sg=1, epochs=50, seed=77)

def gerar_embedding_composicao(row):
    emb_blue = np.mean([model_w2v.wv[row[f'blue{i}']] for i in range(1, 6)], axis=0)
    emb_red = np.mean([model_w2v.wv[row[f'red{i}']] for i in range(1, 6)], axis=0)
    extras = row[colunas_numericas].values.astype(float)
    return np.concatenate([emb_blue, emb_red, extras])

def inverter_exemplo(x_row, y_row):
    blue = x_row[:8]  # emb_blue (8 dimensões)
    red = x_row[8:16]  # emb_red (8 dimensões)
    extras = x_row[16:]  # features numéricas (já normalizadas)
    
    x_invertido = np.concatenate([red, blue, extras])
    y_invertido = 1 - y_row
    return x_invertido, y_invertido

X = np.vstack(df.apply(gerar_embedding_composicao, axis=1))
y = df['result'].astype(int).values

X_invertido = []
y_invertido = []

for xi, yi in zip(X, y):
    xi_inv, yi_inv = inverter_exemplo(xi, yi)
    X_invertido.append(xi_inv)
    y_invertido.append(yi_inv)

# Junta original + invertido
X_total = np.vstack([X, np.array(X_invertido)])
y_total = np.concatenate([y, np.array(y_invertido)])

# ====================
# 3. Treinamento com Calibração
# ====================
X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.2, random_state=77)

base_model = RandomForestClassifier(n_estimators=100, random_state=77)
model = CalibratedClassifierCV(estimator=base_model, cv=5, method='sigmoid')
model.fit(X_train, y_train)

# ====================
# 4. Avaliação
# ====================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"Acurácia: {acc:.2f}")
print(f"ROC AUC: {roc:.2f}")

# ====================
# 5. Predição nova partida
# ====================
nova_partida = {
    'blue1': 'Yorick', 'blue2': 'Maokai', 'blue3': 'Tristana', 'blue4': "Varus", 'blue5': 'Rakan',
    'red1':'Sett', 'red2': 'Trundle', 'red3': 'Hwei', 'red4': 'Jhin', 'red5': "Bard",
    'golddiffat10': 0.0,
   'golddiffat15': 0,
    'golddiffat20':0,
    'golddiffat25': 0,
    'firsttower': 0.0,
    'firstmidtower': 0.0,
    'firstherald': 0.0,
    'dragons': 0.0,
    'opp_dragons':0.0,
    'void_grubs': 0.0,
    'atakhans':0.0,
    'opp_atakhans':0.0,
}

df_nova = pd.DataFrame([nova_partida])

def gerar_embedding_novo(row):
    emb_blue = np.mean([model_w2v.wv[row[f'blue{i}']] for i in range(1, 6)], axis=0)
    emb_red = np.mean([model_w2v.wv[row[f'red{i}']] for i in range(1, 6)], axis=0)
    extras = row[colunas_numericas].values.astype(float)
    return np.concatenate([emb_blue, emb_red, extras])

vetor = gerar_embedding_novo(df_nova.iloc[0]).reshape(1, -1)
vetor[:, -len(colunas_numericas):] = scaler.transform(vetor[:, -len(colunas_numericas):])

# Predição calibrada

prob_azul = model.predict_proba(vetor)[0][1]
prob_vermelho = 1 - prob_azul

print()
print(f"Probabilidade do time AZUL vencer: {prob_azul:.2%}")
print(f"Probabilidade do time VERMELHO vencer: {prob_vermelho:.2%}")
print()