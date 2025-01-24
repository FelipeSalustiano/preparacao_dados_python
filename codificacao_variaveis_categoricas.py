import pandas as pd 
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.width', None)

df = pd.read_csv('clientes-v2-tratados.csv')

print(df.head())

# Codificação one-hot 'estado_civil'
df = pd.concat([df, pd.get_dummies(df['estado_civil'], prefix='estado_civil')], axis=1)

# Codificação ordinal para 'nível_educacao'
educacao_ordem = {
    'Ensino Fundamental': 1, 
    'Ensino Médio':       2, 
    'Ensino Superior':    3, 
    'Pós-graducação':     4
}

df['nivel_educacao_ordinal'] = df['nivel_educacao'].map(educacao_ordem)
