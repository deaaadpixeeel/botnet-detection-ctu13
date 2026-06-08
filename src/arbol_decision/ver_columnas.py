import pandas as pd
import os

# Lee un archivo de ejemplo
ruta = "CTU13/1-Neris-20110810.binetflow.parquet"
df = pd.read_parquet(ruta)

print("Nombres reales de las columnas en tu dataset:")
print("="*50)
for i, col in enumerate(df.columns):
    print(f"{i+1}. '{col}'")

print("\n" + "="*50)
print("Primeras 2 filas para referencia:")
print(df.head(2))