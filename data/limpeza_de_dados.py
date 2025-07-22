import pandas as pd
import numpy as np 
#import matplotlib as plt
# Carregar dados e filtrar
df_times = pd.read_csv('data/composições.csv', low_memory=False)
#print(df_times[['golddiffat10', 'result']].groupby(pd.cut(df_times['golddiffat10'], bins=10)).mean())
colunas_para_manter = [ 'pick1', 'pick2', 'pick3', 'pick4', 'pick5', 'result',
    'league','atakhans','opp_atakhans',
    'golddiffat25','golddiffat20', 'golddiffat15','golddiffat10', 'firstmidtower',
    'firstherald', 'dragons','opp_dragons', 'void_grubs','opp_void_grubs']


'''embeddings = df_times[colunas_para_manter]
embeddings.to_csv('composições.csv', index=False)'''

numeric_df = df_times.select_dtypes(include='number')
corr_matrix = numeric_df.corr()

resultados_da_correlação = corr_matrix['result'].sort_values(ascending =False)
#print(resultados_da_correlação.head(50))

#Outros dados úteis: tempo de jogo, elo, lado do mapa 


def unir_duas_linhas_com_extras(df_times):
    rows = []
    for i in range(0, len(df_times), 2):
        time1 = df_times.iloc[i]
        time2 = df_times.iloc[i+1]

        nova_linha = {}

        # Picks do time azul (linha i)
        for j in range(1, 6):
            nova_linha[f'blue{j}'] = time1[f'pick{j}']

        # Picks do time vermelho (linha i+1)
        for j in range(1, 6):
            nova_linha[f'red{j}'] = time2[f'pick{j}']

        # Resultado da partida (1 se azul venceu)
        nova_linha['result'] = time1['result']

        # Colunas extras
        extras = ['league','atakhans','opp_atakhans',
    'golddiffat25','golddiffat20', 'golddiffat15','golddiffat10', 'firstmidtower',
    'firstherald', 'dragons','opp_dragons', 'void_grubs','opp_void_grubs']

        for col in extras:
            nova_linha[col] = time1.get(col, None)  # evita erro caso col não exista

        rows.append(nova_linha)

    return pd.DataFrame(rows)

# Supondo que seu DataFrame original seja df:
df_filtrado = df_times[colunas_para_manter]  # usa apenas as colunas necessárias
df_unido = unir_duas_linhas_com_extras(df_filtrado)
df_unido = df_unido.dropna()
print(df_unido)

df_unido.to_csv('data/dados_cada_partida.csv', index=False)



