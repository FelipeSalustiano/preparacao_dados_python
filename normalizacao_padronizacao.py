from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import pandas as pd

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('clientes-v2-tratados.csv')

print(df.head())

df = df.drop(['data', 'estado', 'nivel_educacao', 'numero_filhos', 'estado_civil', 'area_atuacao'], axis=1) 

# Normalização - MinMaxScaler (Ajusta os valores para ficarem dentro de um intervalo específico)
scaler = MinMaxScaler()
df['idadeMinMaxScaler'] = scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler'] = scaler.fit_transform(df[['salario']])

min_max_scaler = MinMaxScaler(feature_range=(-1, 1)) # Altera o padrão de 0 a 1 por -1 a 1
df['idadeMinMaxScaler_mm'] = min_max_scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler_mm'] = min_max_scaler.fit_transform(df[['salario']])

# Padronização - StandardScaler (Padroniza os valores com média 0 e o desvio padrão com 1)
scaler = StandardScaler() 
df['idadeStandardScaler'] = scaler.fit_transform(df[['idade']]) 
df['salarioStandardScaler'] = scaler.fit_transform(df[['salario']])

# Padronização - RobustScaler (Ignora outliers ao usar mediana e quartir, sendo mais estável em dados com valores extremos)
scaler = RobustScaler()
df['idadeRobustScaler'] = scaler.fit_transform(df[['idade']])
df['salarioRobustScaler'] = scaler.fit_transform(df[['salario']])

print(df.head(15))

print('MinMaxScaler (De 0 a 1):')
print(f'Idade - Min: {df['idadeMinMaxScaler'].min():.4f} Max: {df['idadeMinMaxScaler'].max():.4f} Mean: {df['idadeMinMaxScaler'].mean():.4f} Std (Desvio Padrão): {df['idadeMinMaxScaler'].std():.4f}')
print(f'Salario - Min: {df["salarioMinMaxScaler"].min():.4f} Max: {df['salarioMinMaxScaler'].max():.4f} Mean: {df['salarioMinMaxScaler'].mean():.4f} Std (Desvio Padrão): {df['salarioMinMaxScaler'].std():.4f}')

print('\nMinMaxScaler (De -1 a 1):')
print(f'Idade - Min: {df['idadeMinMaxScaler_mm'].min():.4f} Max: {df['idadeMinMaxScaler_mm'].max():.4f} Mean: {df['idadeMinMaxScaler_mm'].mean():.4f} Std (Desvio Padrão): {df['idadeMinMaxScaler_mm'].std():.4f}')
print(f'Salario - Min: {df['salarioMinMaxScaler_mm'].min():.4f} Max: {df['salarioMinMaxScaler_mm'].max():.4f} Mean: {df['salarioMinMaxScaler_mm'].mean():.4f} Std (Desvio Padrão): {df['salarioMinMaxScaler_mm'].std():.4f}')

print('\nStandardScaler (Ajusta a média a 0 e desvio padrão a 1):')
print(f'Idade - Min: {df['idadeStandardScaler'].min():.4f} Max: {df['idadeStandardScaler'].max():.4f} Mean: {df['idadeStandardScaler'].mean():.4f} Std (Desvio Padrão): {df['idadeStandardScaler'].std():.4f}')
print(f'Salario - Min: {df['salarioStandardScaler'].min():.4f} Max: {df['salarioStandardScaler'].max():.4f} Mean: {df['salarioStandardScaler'].mean():.4f} Std (Desvio Padrão): {df['salarioStandardScaler'].std():.4f}')

print('\nRobustScaler (Ajusta a mediana e IQR):')
print(f'Idade - Min: {df['idadeRobustScaler'].min():.4f} Max: {df['idadeRobustScaler'].max():.4f} Mean: {df['idadeRobustScaler'].mean():.4f} Std (Desvio Padrão): {df['idadeRobustScaler'].std():.4f}')
print(f'Salario - Min: {df['salarioRobustScaler'].min():.4f} Max: {df['salarioRobustScaler'].max():.4f} Mean: {df['salarioRobustScaler'].mean():.4f} Std (Desvio Padrão): {df['salarioRobustScaler'].std():.4f}')
