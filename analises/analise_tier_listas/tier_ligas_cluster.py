import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Carregar dados ===
df = pd.read_csv("data/data_completa.csv",low_memory=False)

# Filtrar só partidas com dados completos (opcional)
df = df[df["datacompleteness"] == 1]

# === 2. Criar métricas relevantes ===

# Calcular gold_diff no minuto 20 para identificar "vantagem grande"
df["big_gold_adv"] = df["golddiffat20"] > 5000

# Calcular kill diff no minuto 20 (se não existir, podemos usar kills e opp_kills)
df["kill_diff_20"] = df["killsat20"] - df["opp_killsat20"]

# Filtrar só partidas vencidas (result == 1 pode ser vitória, confirme no seu dataset)
df_win = df[df["result"] == 1]

# Agrupar por liga e calcular as métricas

gold_adv_pct = df.groupby("league")["big_gold_adv"].mean()
avg_gamelength_win = df_win.groupby("league")["gamelength"].mean()

# Média de torres por vitória (usar 'towers')
avg_towers_win = df_win.groupby("league")["towers"].mean()

# Média de barons por vitória ('barons')
avg_barons_win = df_win.groupby("league")["barons"].mean()

# Média de dragões por vitória ('dragons')
avg_dragons_win = df_win.groupby("league")["dragons"].mean()

# Média kill diff por vitória no minuto 20
avg_kill_diff_win = df_win.groupby("league")["kill_diff_20"].mean()

# Montar dataframe final
league_stats = pd.DataFrame({
    "gold_adv_pct": gold_adv_pct,
    "avg_gamelength_win": avg_gamelength_win,
    "avg_towers_win": avg_towers_win,
    "avg_barons_win": avg_barons_win,
    "avg_dragons_win": avg_dragons_win,
    "avg_kill_diff_win": avg_kill_diff_win
}).dropna()

# === 3. Normalizar para ponderar ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(league_stats)

# === 4. Calcular força estimada ponderada ===
weights = np.array([1.0, -1.0, 0.8, 0.7, 0.7, 0.9])
league_stats["forca_estimada"] = X_scaled.dot(weights)

# === 5. Clustering em 3 tiers com KMeans ===
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
tier_labels = kmeans.fit_predict(X_scaled)
league_stats["tier"] = tier_labels

# Mapear tiers para Tier 1 (mais forte) a Tier 3 (mais fraco)
centroids = kmeans.cluster_centers_.mean(axis=1)
order = np.argsort(-centroids)
tier_map = {old: f"Tier {i+1}" for i, old in enumerate(order)}
league_stats["tier"] = league_stats["tier"].map(tier_map)

# === 6. Plot - ranking por força estimada ===
ranking = league_stats.sort_values("forca_estimada", ascending=False).reset_index()

plt.figure(figsize=(12, 10))
sns.barplot(data=ranking, x="forca_estimada", y="league", hue="tier", palette="Set2")
plt.title("Ranking de Ligas por Força Estimada (3 Tiers)")
plt.xlabel("Força Estimada")
plt.ylabel("Liga")
plt.legend(title="Tier")
plt.tight_layout()
plt.show()

# === 7. Plot - PCA 2D para visualização ===
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)
league_stats["PCA1"] = pca_coords[:, 0]
league_stats["PCA2"] = pca_coords[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(data=league_stats, x="PCA1", y="PCA2", hue="tier", style="tier", s=120, palette="Set2")

for i, league in enumerate(league_stats.index):
    plt.text(
        league_stats.loc[league, "PCA1"] + 0.05,
        league_stats.loc[league, "PCA2"],
        league,
        fontsize=9
    )

plt.title("Distribuição das Ligas em 2D (PCA)")
plt.tight_layout()
plt.show()

# === 8. Plot - Heatmap das médias das métricas por Tier ===
heatmap_data = league_stats.groupby("tier").mean()[[
    "gold_adv_pct", "avg_gamelength_win", "avg_towers_win",
    "avg_barons_win", "avg_dragons_win", "avg_kill_diff_win"
]]

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".3f")
plt.title("Médias das Métricas por Tier")
plt.tight_layout()
plt.show()
