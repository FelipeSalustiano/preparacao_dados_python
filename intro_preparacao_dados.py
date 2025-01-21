import pandas as pd

# Análise Exploratória de Dados (AED)

df = pd.read_csv('clientes-v2.csv')

print(df.head().to_string())
print(df.tail().to_string())

df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce') # Convertendo os campos de data para o tipo DataTime 

# Verificando as informações do DataFrame 
print('Verificação inicial:')
print(df.info())

# Analisando campos nulos
print('Análise de dados nulos: \n', df.isnull().sum())
print('Porcentagem de dados nulos: \n', df.isnull().mean() * 100)
df.dropna(inplace=True)
print('Confirmar remoção dos dados nulos: \n', df.isnull().sum().sum())

# Analisando campos duplicados 
print('Análise de dados duplicados: \n', df.duplicated().sum())

# Analisando capos unicos 
print('Análise de dados nulos: \n', df.nunique().sum())

# Analisando a estatística dos dados 
print('Estatísticas dos dados: \n', df.describe())

df = df[['idade', 'data', 'estado', 'salario', 'nivel_educacao', 'numero_filhos', 'estado_civil', 'area_atuacao']]
print(df.head().to_string())

df.to_csv('clientes-v2-tratados.csv', index=False)
