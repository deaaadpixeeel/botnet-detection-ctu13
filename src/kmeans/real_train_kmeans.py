import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

#columnas originales del dataset ctu13 en formato binetflow
BASE_FEAT = ["dur","proto","dir","state","stos","dtos","tot_pkts","tot_bytes","src_bytes"]
CAT_COLS  = ["proto","dir","state"]
NUM_COLS  = ["dur","stos","dtos","tot_pkts","tot_bytes","src_bytes"]
ENG_COLS  = ["bytes_per_pkt","src_ratio","pps"]
#columnas con distribucion muy sesgada que necesitan log1p antes de escalar
LOG_COLS  = ["dur","tot_pkts","tot_bytes","src_bytes","bytes_per_pkt","pps"]
RANDOM_STATE = 42
DEFAULT_OUT  = "botnet_kmeans_model.joblib"
STATS_FILE   = "train_stats.txt"


def load_files(patterns):
    #busca todos los parquets que coincidan con los patrones y los concatena en un dataframe
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    files = sorted(set(files))
    if not files:
        print("error: no se encontraron archivos")
        sys.exit(1)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def is_botnet(s):
    return "botnet" in str(s).lower()


def is_background(s):
    return "background" in str(s).lower()


def add_engineered_features(df):
    #agrega tres features derivadas que ayudan a separar el trafico botnet del normal
    #bytes_per_pkt captura el tamano promedio del paquete
    #src_ratio captura la asimetria del flujo indicando si es unidireccional
    #pps captura la intensidad del flujo en paquetes por segundo
    df = df.copy()
    df["bytes_per_pkt"] = df["tot_bytes"] / (df["tot_pkts"]  + 1e-6)
    df["src_ratio"]     = df["src_bytes"] / (df["tot_bytes"] + 1e-6)
    df["pps"]           = df["tot_pkts"]  / (df["dur"]       + 1e-6)
    return df


def sample_balanced(df, label_col, n_total, rs):
    #muestrea 50 por ciento botnet y 50 por ciento normal excluyendo background
    #es necesario porque ctu13 tiene solo 1 a 3 por ciento de botnet en el total
    #sin balanceo kmeans nunca forma clusters puros de botnet
    bot  = df[df[label_col].apply(is_botnet)]
    norm = df[~df[label_col].apply(is_botnet) & ~df[label_col].apply(is_background)]
    n2 = n_total // 2
    nb = min(n2, len(bot))
    nn = min(n_total - nb, len(norm))
    out = pd.concat([bot.sample(nb, random_state=rs), norm.sample(nn, random_state=rs)])
    return out.sample(frac=1, random_state=rs).reset_index(drop=True)


def build_feature_matrix(df, encoders=None, scaler=None, medians=None, fit=True):
    #construye la matriz numerica que kmeans necesita para entrenar
    #label encoder convierte texto a numeros en columnas categoricas
    #log1p comprime la escala de columnas con valores extremos
    #standardscaler iguala el peso de todas las columnas en la distancia euclidiana
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files",     nargs="+", required=True,
                        help="archivos parquet de entrenamiento acepta globs")
    parser.add_argument("--k",         type=int,   default=12,
                        help="numero de clusters default 12")
    parser.add_argument("--sample",    type=int,   default=150_000,
                        help="maximo de filas a usar default 150000")
    parser.add_argument("--balance",   action="store_true",
                        help="muestreo balanceado 50 por ciento botnet 50 por ciento normal")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="minimo ratio de botnet para etiquetar un cluster como botnet default 0.5")
    parser.add_argument("--output",    default=DEFAULT_OUT,
                        help=f"archivo de salida del modelo default {DEFAULT_OUT}")
    args = parser.parse_args()

    df_all = load_files(args.files)
    has_labels = "label" in df_all.columns

    #selecciona las filas de entrenamiento segun la estrategia de muestreo elegida
    if args.balance and has_labels:
        n = args.sample or 80_000
        df_train = sample_balanced(df_all, "label", n, RANDOM_STATE)
    elif args.sample and len(df_all) > args.sample:
        df_train = df_all.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        df_train = df_all

    bm = df_train["label"].apply(is_botnet) if has_labels else None
    X, encoders, scaler, medians = build_feature_matrix(df_train, fit=True)

    #entrena minibatchkmeans que es la version eficiente de kmeans para datasets grandes
    kmeans = MiniBatchKMeans(
        n_clusters=args.k, random_state=RANDOM_STATE,
        batch_size=8192, n_init=10, max_iter=300
    )
    kmeans.fit(X)
    cids = kmeans.labels_

    #calcula metricas de calidad del clustering sobre una submuestra para reducir el tiempo
    idx = np.random.choice(len(X), min(10_000, len(X)), replace=False)
    sil = silhouette_score(X[idx], cids[idx])
    db  = davies_bouldin_score(X[idx], cids[idx])

    #etiqueta cada cluster como botnet o normal segun su ratio de botnet en el entrenamiento
    cbot, cinterp = {}, {}
    for cl in np.unique(cids):
        if has_labels and bm is not None:
            r = float(bm[cids == cl].mean())
            cbot[int(cl)] = r
            cinterp[int(cl)] = "BOTNET" if r >= args.threshold else "NORMAL"
        else:
            cbot[int(cl)] = None
            cinterp[int(cl)] = "UNKNOWN"

    #guarda el modelo junto con todo el preprocesamiento necesario para clasificar datos nuevos
    bundle = {
        "kmeans":                 kmeans,
        "scaler":                 scaler,
        "encoders":               encoders,
        "medians":                medians,
        "feature_cols":           CAT_COLS + NUM_COLS + ENG_COLS,
        "cat_cols":               CAT_COLS,
        "num_cols":               NUM_COLS,
        "eng_cols":               ENG_COLS,
        "n_clusters":             args.k,
        "cluster_botnet_ratio":   cbot,
        "cluster_interpretation": cinterp,
        "silhouette":             sil,
        "davies_bouldin":         db,
        "trained_on":             args.files,
        "balanced_training":      args.balance,
        "botnet_threshold":       args.threshold,
    }
    joblib.dump(bundle, args.output)

    #escribe las estadisticas del entrenamiento en un archivo de texto plano
    lines = [
        f"k: {args.k}",
        f"threshold: {args.threshold}",
        f"balanced: {args.balance}",
        f"samples: {len(X)}",
        f"silhouette: {sil:.4f}",
        f"davies_bouldin: {db:.4f}",
        "",
        "clusters:",
    ]
    for cl in sorted(cinterp):
        r = cbot[cl]
        r_str = f"{r*100:.1f}%" if r is not None else "sin referencia"
        lines.append(f"  c{cl}: {cinterp[cl]}  botnet_ratio: {r_str}")

    with open(STATS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"modelo: {args.output}")
    print(f"estadisticas: {STATS_FILE}")


if __name__ == "__main__":
    main()
