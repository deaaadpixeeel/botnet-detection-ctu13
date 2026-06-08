import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from matplotlib.lines import Line2D

RANDOM_STATE = 42
BASE_FEAT = ["dur","proto","dir","state","stos","dtos","tot_pkts","tot_bytes","src_bytes"]
#columnas a las que se aplica log1p igual que en el entrenamiento
LOG_COLS  = ["dur","tot_pkts","tot_bytes","src_bytes","bytes_per_pkt","pps"]
STATS_FILE = "test_stats.txt"
PLOT_FILE  = "test_pca2d.png"


def is_botnet(s):
    return "botnet" in str(s).lower()


def load_files(patterns):
    #carga todos los parquets que coincidan con los patrones y los concatena
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    files = sorted(set(files))
    if not files:
        print("error: no se encontraron archivos")
        sys.exit(1)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def add_engineered_features(df):
    #calcula las mismas features derivadas usadas en el entrenamiento
    df = df.copy()
    df["bytes_per_pkt"] = df["tot_bytes"] / (df["tot_pkts"]  + 1e-6)
    df["src_ratio"]     = df["src_bytes"] / (df["tot_bytes"] + 1e-6)
    df["pps"]           = df["tot_pkts"]  / (df["dur"]       + 1e-6)
    return df


def preprocess_test(df, bundle):
    #aplica exactamente el mismo preprocesamiento que se uso al entrenar
    #usa los encoders escalador y medianas guardados dentro del modelo
    #si aparece una categoria desconocida la reemplaza por el valor de fallback
    encoders = bundle["encoders"]
    scaler   = bundle["scaler"]
    medians  = bundle["medians"]
    cat_cols = bundle["cat_cols"]
    num_cols = bundle["num_cols"]
    eng_cols = bundle.get("eng_cols", [])
    all_num  = num_cols + eng_cols

    df = add_engineered_features(df[BASE_FEAT])
    df = df[cat_cols + all_num].copy()

    for col in cat_cols:
        le = encoders[col]
        df[col] = df[col].astype(str).fillna("_MISSING_")
        known = set(le.classes_)
        fb = "_MISSING_" if "_MISSING_" in known else le.classes_[0]
        df[col] = df[col].apply(lambda x: x if x in known else fb)
        df[col] = le.transform(df[col])

    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    for col in all_num:
        df[col] = df[col].fillna(medians.get(col, 0.0))

    return scaler.transform(df.astype(float).values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         default="botnet_kmeans_model.joblib",
                        help="modelo guardado por real_train_kmeans.py o train_kmeans.py")
    parser.add_argument("--file",          nargs="+", required=True,
                        help="archivos parquet a evaluar acepta globs")
    parser.add_argument("--sample",        type=int, default=0,
                        help="muestrea n filas 0 significa todo")
    parser.add_argument("--no-background", action="store_true",
                        help="excluye flujos background antes de calcular metricas")
    parser.add_argument("--no-rules",      action="store_true",
                        help="desactiva las reglas post-proceso R1 y R2")
    parser.add_argument("--stats-file",    default=STATS_FILE,
                        help=f"archivo de salida de estadisticas default {STATS_FILE}")
    parser.add_argument("--plot-file",     default=None,
                        help="archivo de salida de la grafica; si se omite usa "
                             "test_pca2d-con-reglas.png (con reglas) o test_pca2d.png (sin reglas)")
    args = parser.parse_args()

    # nombre del PNG: depende de si las reglas están activas, salvo que se pase --plot-file
    if args.plot_file is None:
        args.plot_file = PLOT_FILE if args.no_rules else "test_pca2d-con-reglas.png"

    bundle = joblib.load(args.model)
    kmeans = bundle["kmeans"]
    interp = bundle["cluster_interpretation"]
    ratio  = bundle["cluster_botnet_ratio"]

    df = load_files(args.file)
    has_labels = "label" in df.columns

    #excluye el trafico background antes de evaluar para que las metricas sean comparables
    #background no es ni botnet ni trafico de usuario por lo que distorsiona precision y recall
    if args.no_background and has_labels:
        df = df[~df["label"].apply(lambda x: "background" in str(x).lower())].reset_index(drop=True)

    if has_labels:
        true_bot = df["label"].apply(is_botnet).astype(int)

    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)
        if has_labels:
            true_bot = df["label"].apply(is_botnet).astype(int)

    X = preprocess_test(df, bundle)

    #predice el cluster para cada flujo y lo convierte a binario botnet o normal
    pred_cl  = kmeans.predict(X)
    pred_bin = np.array([1 if interp.get(int(c), "UNKNOWN") == "BOTNET" else 0
                         for c in pred_cl])

    if not args.no_rules:
        # R1: UDP/CON con respuesta < 0.1 ms → loopback/caché local, imposible en C&C remoto
        # Neris DNS real tarda > 0.4 ms incluso en su percentil 25; loopback es < 0.1 ms
        r1 = ((df["proto"].values == "udp") &
              (df["state"].values == "CON") &
              (df["dur"].values < 0.0001))
        # R2: TCP largo de baja intensidad → keepalive IRC/botnet
        # sesiones > 5 min, < 200 bytes/pkt y < 0.5 pkts/s no ocurren en tráfico web normal
        bpp_raw = df["tot_bytes"].values / (df["tot_pkts"].values + 1e-6)
        pps_raw = df["tot_pkts"].values  / (df["dur"].values       + 1e-6)
        r2 = ((df["proto"].values == "tcp") &
              (df["dur"].values > 300) &
              (bpp_raw < 200) & (pps_raw < 0.5) &
              (pred_bin == 0))
        n_r1 = int(r1.sum())
        n_r2 = int(r2.sum())
        pred_bin[r1] = 0
        pred_bin[r2] = 1
        rules_line = f"reglas post-proceso: R1 (dns-local) corrigió {n_r1}, R2 (irc) activó {n_r2}"
    else:
        rules_line = "reglas post-proceso: desactivadas (--no-rules)"

    stat_lines = []

    if has_labels:
        cm = confusion_matrix(true_bot, pred_bin)
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(true_bot, pred_bin, zero_division=0)
        rec  = recall_score(true_bot, pred_bin, zero_division=0)
        f1   = f1_score(true_bot, pred_bin, zero_division=0)
        tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        stat_lines = [
            rules_line,
            "",
            f"precision: {prec:.4f}",
            f"recall: {rec:.4f}",
            f"f1: {f1:.4f}",
            f"tpr: {tpr:.4f}",
            f"fpr: {fpr:.4f}",
            "",
            "matriz de confusion:",
            f"                 pred normal  pred botnet",
            f"real normal      {tn:>11,}  {fp:>11,}",
            f"real botnet      {fn:>11,}  {tp:>11,}",
        ]

        for line in stat_lines:
            print(line)

    #genera la visualizacion pca 2d recortando outliers para que el scatter sea legible
    n_vis = min(25_000, len(X))
    idx   = np.random.choice(len(X), n_vis, replace=False)
    pca   = PCA(n_components=2, random_state=RANDOM_STATE)
    X2d   = pca.fit_transform(X[idx])
    var   = pca.explained_variance_ratio_ * 100
    labs  = pred_cl[idx]

    color_map = {"BOTNET": "tab:red", "NORMAL": "tab:green", "UNKNOWN": "tab:gray"}
    pt_colors = np.array([color_map.get(interp.get(int(c), "UNKNOWN"), "tab:gray")
                          for c in labs])

    #recorta el 0.5 por ciento de outliers en cada eje para que el grafico no se comprima
    xlo, xhi = np.percentile(X2d[:,0], [0.5, 99.5])
    ylo, yhi = np.percentile(X2d[:,1], [0.5, 99.5])
    mv = ((X2d[:,0]>=xlo)&(X2d[:,0]<=xhi)&(X2d[:,1]>=ylo)&(X2d[:,1]<=yhi))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X2d[mv,0], X2d[mv,1], c=pt_colors[mv], s=5, alpha=0.35, rasterized=True)

    legend_items = []
    for cl in sorted(interp):
        tag   = interp[cl]
        color = color_map.get(tag, "tab:gray")
        r_val = ratio.get(cl)
        r_str = f" ({r_val*100:.0f}% bot)" if r_val is not None else ""
        legend_items.append(
            Line2D([0],[0], marker="o", color="w", markerfacecolor=color,
                   markersize=9, label=f"C{cl} {tag}{r_str}")
        )
    centers_2d = pca.transform(kmeans.cluster_centers_)
    legend_items.append(
        Line2D([0],[0], marker="X", color="black", markersize=11,
               label="centroides", linestyle="None")
    )
    ax.scatter(centers_2d[:,0], centers_2d[:,1], s=220, marker="X",
               color="black", zorder=5)
    ax.legend(handles=legend_items, fontsize=8)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% varianza)")
    ax.set_title("KMeans Botnet Detector - PCA 2D")
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot_file, dpi=150)
    plt.close()

    with open(args.stats_file, "w") as f:
        f.write("\n".join(stat_lines) + "\n")

    print(f"\ngrafica: {args.plot_file}")
    print(f"estadisticas: {args.stats_file}")


if __name__ == "__main__":
    main()
