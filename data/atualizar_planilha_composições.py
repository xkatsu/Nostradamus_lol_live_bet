import pandas as pd

df_times = pd.read_csv('data/data_completa.csv', low_memory=False)
#x = df_times['league'].value_counts()
#print(x)
#ligas_excluidas = ['LPL', 'LCK', 'LEC', 'LCP', 'LTA N', 'MSI', 'LTA S', 'NEXO','LJL','VCS','CD','LFL2','NACL','LCKC']


ligas_recomendadas = ['LTA S', 'LPL', 'LCK', 'LEC', 'LCP', 'LTA N','MSI','EWC','LCKC']

ligas_relevantes = [
    "LPL", "LCK", "LJL", "LCKC", "PCS", "EM", "LEC", "PRM", "AL", "CD", "LCP", "LFL",
    "RL", "LVP SL", "NACL", "TCL", "HLL", "NLC", "VCS", "ROL", "LTA S", "LTA N",
    "HW", "EBL", "HC", "LRN", "HM", "LIT", "LRS", "LPLOL", "MSI"
]

liga_alvo = [ 'LPL', 'LCK', 'LEC', 'LCP', 'LTA N']

df_limpo = df_times[
    (df_times.position == 'team') &
    (df_times.patch >= 15.09) &
    (df_times.league.isin(ligas_recomendadas))
]



colunas_para_manter = [ 'pick1', 'pick2', 'pick3', 'pick4', 'pick5', 'result',
    'league','atakhans','opp_atakhans',
    'golddiffat25','golddiffat20', 'golddiffat15','golddiffat10', 'firstmidtower',
    'firstherald', 'dragons','opp_dragons', 'void_grubs','opp_void_grubs']



embeddings = df_limpo[colunas_para_manter].dropna()
print(embeddings.head())

embeddings.to_csv('data/composições.csv', index=False)
