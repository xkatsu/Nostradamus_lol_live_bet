import pandas as pd

df = pd.read_csv('analises/falsos_positivos_25min.csv')
df1 = pd.read_csv('analises/falsos_positivos_20min.csv')
df2 = pd.read_csv('analises/falsos_positivos_15min.csv')

df_ = pd.read_csv('analises/falsos_negativos_25min.csv')
df1_ = pd.read_csv('analises/falsos_negativos_20min.csv')
df2_ = pd.read_csv('analises/falsos_negativos_15min.csv')


dfteste_fp = pd.concat([df , df1 , df2], ignore_index=True)
dfteste_fn = pd.concat([df_, df1_, df2_], ignore_index=True)

red_cols = ['red1', 'red2', 'red3', 'red4', 'red5']
blue_cols =['blue1','blue2', 'blue3','blue4','blue5']

# Empilhar todos os campeões das colunas red em uma única série
todos_reds_fp = dfteste_fp[red_cols].values.ravel()
todos_blue_fp = dfteste_fp[blue_cols].values.ravel()

todos_reds_fn = dfteste_fn[red_cols].values.ravel()
todos_blue_fn = dfteste_fn[blue_cols].values.ravel()


# Contar ocorrências
contagem_red_fp = pd.Series(todos_reds_fp).value_counts()
contagem_blue_fp = pd.Series(todos_blue_fp).value_counts()

contagem_red_fn = pd.Series(todos_reds_fn).value_counts()
contagem_blue_fn = pd.Series(todos_blue_fn).value_counts()

#contagem_baits =


# Mostrar os 10 mais frequentes fp
print("Campeões mais frequêntes em Vitorias IMPROVAVEIS:")
print(contagem_red_fp.head(10))

print("Campeões mais frequêntes em Derrotas IMPROVAVEIS:")
print(contagem_blue_fp.head(10))

# Mostrar os 10 mais frequentes fn
print("Campeões mais frequêntes em Derrotas IMPROVAVEIS:")
print(contagem_red_fn.head(10))

print("Campeões mais frequêntes em Vitorias IMPROVAVEIS:")
print(contagem_blue_fn.head(10))