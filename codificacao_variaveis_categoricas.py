import pandas as pd 
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.width', None)

df = pd.read_csv('clientes-v2-tratados.csv')

print(df.head())

# Codificação one-hot 'estado_civil'. (Obs.: o método one-hot é ideal para variáveis categóricas sem relação ordinal)
df = pd.concat([df, pd.get_dummies(df['estado_civil'], prefix='estado_civil')], axis=1)
print('\nDataFrame após codificação one-hot para "estado_civil":\n', df.head())

# Codificação ordinal para 'nível_educacao'. (Obs.: o método ordinal é ideal para variáveis com ordem natural, capturando a hierarquia dos dados)
educacao_ordem = {
    'Ensino Fundamental': 1, 
    'Ensino Médio':       2, 
    'Ensino Superior':    3, 
    'Pós-graducação':     4
}

df['nivel_educacao_ordinal'] = df['nivel_educacao'].map(educacao_ordem)
print('\nDataFrame após codificação ordinal para "nivel_educacao":\n', df.head())

# Tranformando 'area_atuacao' em categorias codificadas usando o método .cat.codes (Obs.: o método .cat.codes atribui números únicos para categorias. É útil para simplificar categorias que não tem uma ordem específica)
df['area_atuacao_cod'] = df['area_atuacao'].astype('category').cat.codes
print('\nDataFrame após transormar "area_atuacao" em códigos numéricos:\n', df.head())

# LabelEncorder para 'estado' (Obs.: Esse método converte cada valor único em números de 0 a n_classes-1. É igual ao .cat.codes, porém é mais acessivel a modelos de machine learning)
label_encoder = LabelEncoder()
df['estado_cod'] = label_encoder.fit_transform(df['estado'])
print('\nDataFrame após aplicar o LabelEncoder em "estado":\n', df.head())


