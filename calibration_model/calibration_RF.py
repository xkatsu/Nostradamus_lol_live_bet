import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# Carregar os dados
df = pd.read_csv("data/dados_cada_partida.csv")

# Checkpoints e colunas numéricas
CHAMP_COLS = ['blue1', 'blue2', 'blue3', 'blue4', 'blue5',
              'red1', 'red2', 'red3', 'red4', 'red5']

CHECKPOINTS = {
    "draft": [],
    "10min": ['golddiffat10', 'void_grubs','opp_void_grubs','dragons', 'opp_dragons'],
    "15min": ['golddiffat10', 'golddiffat15', 'firstmidtower', 'firstherald','dragons', 'opp_dragons'],
    "20min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'void_grubs','opp_void_grubs'],
    "25min": ['golddiffat10', 'golddiffat15', 'golddiffat20', 'golddiffat25', 'firstmidtower', 'firstherald', 'dragons', 'opp_dragons', 'atakhans', 'opp_atakhans'],
}

# Loop por checkpoint
for checkpoint, cols in CHECKPOINTS.items():
    if not cols:
        continue  # pular "draft", sem features numéricas

    df_cp = df[CHAMP_COLS + cols + ['result']].dropna()
    if df_cp.empty:
        print(f"[{checkpoint}] Nenhum dado disponível. Pulando.\n")
        continue

    print(f"\n========== CHECKPOINT: {checkpoint.upper()} ==========")

    # Separar dados
    X = df_cp[cols]
    y = df_cp['result']
    
    # Padronizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Treinar modelo
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    base_model = RandomForestClassifier(n_estimators=300, random_state=42)
    model = CalibratedClassifierCV(base_model, cv=3)
    model.fit(X_train, y_train)

    # Previsões
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    threshold = 0.38
    y_pred = (y_proba >= threshold).astype(int)

    # ==== Curva ROC ====
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"[{checkpoint.upper()}] Curva ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ==== Matriz de Confusão ====
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Derrota', 'Vitória'], yticklabels=['Derrota', 'Vitória'])
    plt.title(f"[{checkpoint.upper()}] Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

    # ==== Distribuição das Probabilidades ====
    plt.figure(figsize=(6, 4))
    sns.histplot(y_proba, bins=20, kde=True, color='skyblue')
    plt.title(f"[{checkpoint.upper()}] Distribuição das Probabilidades")
    plt.xlabel("Probabilidade prevista de vitória")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # ==== Distribuição do GoldDiff (se existir) ====
    if 'golddiffat20' in cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df_cp, x='result', y='golddiffat20', palette='Set2')
        plt.title(f"[{checkpoint.upper()}] GoldDiff aos 20min por Resultado")
        plt.xlabel("Vitória (1) vs Derrota (0)")
        plt.ylabel("Gold Diff aos 20min")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        # ==== Salvar Falsos Positivos e Falsos Negativos ====
    X_test_df = pd.DataFrame(X_test, columns=cols)
    X_test_df['real'] = y_test.values
    X_test_df['predito'] = y_pred
    X_test_df['probabilidade'] = y_proba

    campeoes_df = df_cp.iloc[y_test.index][CHAMP_COLS].reset_index(drop=True)
    X_test_df = pd.concat([X_test_df.reset_index(drop=True), campeoes_df], axis=1)

    # Falsos Positivos: previu vitória (1), mas era derrota (0)
    fp = X_test_df[(X_test_df['real'] == 0) & (X_test_df['predito'] == 1)]

    # Falsos Negativos: previu derrota (0), mas era vitória (1)
    fn = X_test_df[(X_test_df['real'] == 1) & (X_test_df['predito'] == 0)]

    # Salvar em CSV
    fp.to_csv(f"analises/falsos_positivos_{checkpoint}.csv", index=False)
    fn.to_csv(f"analises/falsos_negativos_{checkpoint}.csv", index=False)

    print(f"Salvos: falsos_positivos_{checkpoint}.csv e falsos_negativos_{checkpoint}.csv")
