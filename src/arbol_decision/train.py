"""
Entrenamiento de Arbol de Decision para Deteccion de Botnets
Dataset: CTU-13 (formato .parquet)
Metodologia: Entrenamiento con escenarios 1,2,3,4,5,6,8,12
Features alineadas con proyecto K-Means para comparativa directa.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURACION (alineada con K-Means)
# ============================================================

ESCENARIOS_ENTRENAMIENTO = [1, 2, 3, 4, 5, 6, 8, 12]

FILE_NAMES = {
    1:  "1-Neris-20110810.binetflow.parquet",
    2:  "2-Neris-20110811.binetflow.parquet",
    3:  "3-Rbot-20110812.binetflow.parquet",
    4:  "4-Rbot-20110815.binetflow.parquet",
    5:  "5-Virut-20110815-2.binetflow.parquet",
    6:  "6-Menti-20110816.binetflow.parquet",
    7:  "7-Sogou-20110816-2.binetflow.parquet",
    8:  "8-Murlo-20110816-3.binetflow.parquet",
    9:  "9-Neris-20110817.binetflow.parquet",
    10: "10-Rbot-20110818.binetflow.parquet",
    11: "11-Rbot-20110818-2.binetflow.parquet",
    12: "12-NsisAy-20110819.binetflow.parquet",
    13: "13-Virut-20110815-3.binetflow.parquet",
}

# Mismas columnas base que K-Means
BASE_FEAT = ["dur", "proto", "dir", "state", "stos", "dtos",
             "tot_pkts", "tot_bytes", "src_bytes"]
CAT_COLS  = ["proto", "dir", "state"]
NUM_COLS  = ["dur", "stos", "dtos", "tot_pkts", "tot_bytes", "src_bytes"]
ENG_COLS  = ["bytes_per_pkt", "src_ratio", "pps"]
# Columnas con distribucion sesgada — mismas que K-Means
LOG_COLS  = ["dur", "tot_pkts", "tot_bytes", "src_bytes", "bytes_per_pkt", "pps"]

TAMANO_MUESTRA = 150000
USAR_BALANCE   = True
RANDOM_STATE   = 42

# ============================================================
# 2. FUNCIONES DE CARGA
# ============================================================

def mostrar_progreso(actual, total, mensaje, inicio_tiempo=None):
    porcentaje = (actual / total) * 100
    filled = int(30 * actual // total)
    barra = '#' * filled + '-' * (30 - filled)
    if inicio_tiempo and actual > 0:
        t = time.time() - inicio_tiempo
        r = (t / actual) * (total - actual)
        tiempo_str = f"[trans: {t:.0f}s | rest: {r:.0f}s]"
    else:
        tiempo_str = ""
    print(f"\r[{barra}] {porcentaje:.1f}% - {mensaje} {tiempo_str}", end="", flush=True)


def is_botnet(s):
    s = str(s).lower()
    return "botnet" in s

def is_background(s):
    return "background" in str(s).lower()

def is_normal(s):
    return "normal" in str(s).lower()


def cargar_datos_ctu13(escenarios, carpeta="CTU13", muestrear=None):
    print("\n" + "="*60)
    print("PASO 1: CARGANDO ARCHIVOS")
    print("="*60)

    todos_datos = []
    for idx, escenario in enumerate(escenarios, 1):
        nombre = FILE_NAMES.get(escenario)
        ruta   = os.path.join(carpeta, nombre)
        print(f"\n[{idx}/{len(escenarios)}] Leyendo {nombre}...")
        t0 = time.time()
        try:
            df = pd.read_parquet(ruta)
            print(f"    -> {len(df):,} registros en {time.time()-t0:.1f}s")
            if 'label' in df.columns:
                bg = df['label'].apply(is_background).sum()
                if bg > 0:
                    df = df[~df['label'].apply(is_background)]
                    print(f"    -> Eliminados {bg:,} Background. Quedan {len(df):,}")
            todos_datos.append(df)
        except FileNotFoundError:
            print(f"    -> ERROR: no encontrado {ruta}")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    print("\n" + "-"*40)
    datos = pd.concat(todos_datos, ignore_index=True)
    print(f"Total concatenado: {len(datos):,} registros")

    if muestrear and USAR_BALANCE:
        datos = balancear_dataset(datos, muestrear)

    return datos


def balancear_dataset(df, tamano):
    print("\n" + "="*60)
    print("PASO 2: BALANCEANDO DATASET (50% botnet / 50% normal)")
    print("="*60)

    bot  = df[df['label'].apply(is_botnet)]
    norm = df[df['label'].apply(is_normal) & ~df['label'].apply(is_botnet)]

    print(f"  Botnet disponible: {len(bot):,}")
    print(f"  Normal disponible: {len(norm):,}")

    n2 = tamano // 2
    nb = min(n2, len(bot))
    nn = min(tamano - nb, len(norm))

    out = pd.concat([
        bot.sample(nb,  random_state=RANDOM_STATE),
        norm.sample(nn, random_state=RANDOM_STATE),
    ])
    out = out.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"  Dataset balanceado: {len(out):,} ({nb:,} botnet / {nn:,} normal)")
    return out


# ============================================================
# 3. PREPROCESAMIENTO (identico a K-Means)
# ============================================================

def add_engineered_features(df):
    """Agrega bytes_per_pkt, src_ratio y pps — identico al K-Means."""
    df = df.copy()
    df["bytes_per_pkt"] = df["tot_bytes"] / (df["tot_pkts"]  + 1e-6)
    df["src_ratio"]     = df["src_bytes"]  / (df["tot_bytes"] + 1e-6)
    df["pps"]           = df["tot_pkts"]   / (df["dur"]       + 1e-6)
    return df


def build_feature_matrix(df, encoders=None, scaler=None, medians=None, fit=True):
    """
    Pipeline de preprocesamiento identico al K-Means:
      - LabelEncoder en categoricas
      - log1p en columnas sesgadas
      - StandardScaler
    """
    df = add_engineered_features(df[BASE_FEAT])
    all_num = NUM_COLS + ENG_COLS
    df = df[CAT_COLS + all_num].copy()

    if fit:
        encoders = {}
        for col in CAT_COLS:
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna("_MISSING_")
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in CAT_COLS:
            le = encoders[col]
            df[col] = df[col].astype(str).fillna("_MISSING_")
            known = set(le.classes_)
            fb = "_MISSING_" if "_MISSING_" in known else le.classes_[0]
            df[col] = df[col].apply(lambda x: x if x in known else fb)
            df[col] = le.transform(df[col])

    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    if fit:
        medians = {c: float(df[c].median()) for c in all_num}
    for col in all_num:
        df[col] = df[col].fillna(medians.get(col, 0.0))

    X = df.astype(float).values

    if fit:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)

    return Xs, encoders, scaler, medians


# ============================================================
# 4. ENTRENAMIENTO
# ============================================================

def entrenar_arbol(X, y):
    print("\n" + "="*60)
    print("PASO 5: ENTRENANDO ARBOL DE DECISION")
    print("="*60)
    print(f"  Muestras: {X.shape[0]:,}   Features: {X.shape[1]}")
    print("  max_depth=15 | min_samples_split=50 | min_samples_leaf=25")
    print("  criterion=gini | class_weight=balanced")
    print("\nEntrenando...")

    modelo = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=25,
        criterion='gini',
        class_weight='balanced',
        random_state=RANDOM_STATE,
    )

    t0 = time.time()
    modelo.fit(X, y)
    print(f"Entrenamiento completado en {time.time()-t0:.1f}s")
    return modelo


# ============================================================
# 5. MAIN
# ============================================================

def main():
    t_total = time.time()

    print("="*60)
    print("ENTRENAMIENTO ARBOL DE DECISION - CTU-13")
    print("="*60)
    print(f"Escenarios: {ESCENARIOS_ENTRENAMIENTO}")
    print(f"Inicio: {time.strftime('%H:%M:%S')}")

    # Cargar y balancear
    datos = cargar_datos_ctu13(ESCENARIOS_ENTRENAMIENTO, "CTU13", TAMANO_MUESTRA)

    # Variable objetivo
    print("\n" + "="*60)
    print("PASO 3: CREANDO VARIABLE OBJETIVO")
    print("="*60)
    y = datos['label'].apply(is_botnet).astype(int)
    print(f"  Normal (0): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    print(f"  Botnet (1): {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

    # Preprocesamiento
    print("\n" + "="*60)
    print("PASO 4: PREPROCESAMIENTO (alineado con K-Means)")
    print("="*60)
    print(f"  Features base:      {BASE_FEAT}")
    print(f"  Features engineered: {ENG_COLS}")
    print(f"  Log1p en:           {LOG_COLS}")
    print(f"  Total features:     {len(CAT_COLS)+len(NUM_COLS)+len(ENG_COLS)}")

    t0 = time.time()
    X, encoders, scaler, medians = build_feature_matrix(datos, fit=True)
    print(f"  Preprocesamiento completado en {time.time()-t0:.1f}s")
    print(f"  Dimensiones: {X.shape[0]:,} x {X.shape[1]}")

    # Entrenar
    modelo = entrenar_arbol(X, y.values)

    # Guardar
    print("\n" + "="*60)
    print("PASO 6: GUARDANDO MODELO")
    print("="*60)
    os.makedirs('modelos', exist_ok=True)

    bundle = {
        "modelo":     modelo,
        "encoders":   encoders,
        "scaler":     scaler,
        "medians":    medians,
        "base_feat":  BASE_FEAT,
        "cat_cols":   CAT_COLS,
        "num_cols":   NUM_COLS,
        "eng_cols":   ENG_COLS,
        "log_cols":   LOG_COLS,
    }
    joblib.dump(bundle, 'modelos/arbol_decision_modelo.joblib')
    print("  Guardado: modelos/arbol_decision_modelo.joblib")

    # Evaluacion rapida en entrenamiento
    print("\n" + "="*60)
    print("EVALUACION EN DATOS DE ENTRENAMIENTO")
    print("="*60)
    y_pred = modelo.predict(X)
    print(classification_report(y, y_pred, target_names=['Normal', 'Botnet']))

    cm = confusion_matrix(y, y_pred)
    print("Matriz de confusion:")
    print(f"               Normal   Botnet")
    print(f"  Real Normal  {cm[0,0]:>7,}  {cm[0,1]:>7,}")
    print(f"  Real Botnet  {cm[1,0]:>7,}  {cm[1,1]:>7,}")

    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"  Tiempo total: {time.time()-t_total:.1f}s")
    print(f"  Fin: {time.strftime('%H:%M:%S')}")
    print("\nAhora ejecuta: python test.py")


if __name__ == "__main__":
    main()
