import numpy as np

def gerar_features(df, model_w2v, numeric_cols, league_encoding=None, champ_freq=None):
    """
    Gera os vetores de entrada para os modelos de previsão.
    
    - Para modelo v1: usa emb_blue + emb_red + extras [+ league]
    - Para modelo v2: usa emb_blue + emb_red + emb_diff + extras [+ league]

    A detecção é feita com base na presença de league_encoding (só usada no draft/v2).
    """
    X = []
    for idx, row in df.iterrows():
        def emb(champs):
            vecs, ws = [], []
            for champ in champs:
                if champ in model_w2v.wv:
                    vecs.append(model_w2v.wv[champ])
                    freq = champ_freq.get(champ, 1)
                    ws.append(np.log1p(freq))  # Peso suavizado por frequência
            return np.average(vecs, axis=0, weights=ws) if vecs else np.zeros(model_w2v.vector_size)

        emb_blue = emb([row[f'blue{i}'] for i in range(1, 6)])
        emb_red  = emb([row[f'red{i}'] for i in range(1, 6)])
        
        # Se estamos usando o modelo v2 (detecção via league_encoding), aplicamos emb_diff
        usar_diff = league_encoding is not None
        emb_diff = emb_blue - emb_red if usar_diff else np.array([])

        extras = row[numeric_cols].to_numpy(dtype=float) if numeric_cols else np.array([])
        league_feats = np.array([league_encoding.get(row['league'], 0.5)]) if league_encoding is not None else np.array([])

        full_vector = np.concatenate([emb_blue, emb_red, emb_diff, extras, league_feats])
        X.append(full_vector)

    return np.vstack(X)
