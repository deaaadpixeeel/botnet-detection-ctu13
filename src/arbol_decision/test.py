"""
Evaluacion del Arbol de Decision en escenarios no vistos.
Alineado con K-Means para comparativa directa:
  - Mismas features (BASE_FEAT + engineered)
  - Mismas reglas R1/R2 con mismos umbrales
  - Mismas metricas (precision, recall, f1, tpr, fpr)
  - Visualizacion PCA 2D identica
  - Soporte para escenarios sinteticos
"""

import argparse
import glob
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from matplotlib.lines import Line2D
import seaborn as sns

RANDOM_STATE = 42

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

# ============================================================
# CARGA
# ============================================================

def is_botnet(s):
    return "botnet" in str(s).lower()

def is_background(s):
    return "background" in str(s).lower()


def load_files(patterns_or_nums, carpeta="CTU13"):
    """
    Acepta:
      - lista de enteros (numeros de escenario)
      - lista de strings/globs (rutas directas)
    """
    frames = []
    for p in patterns_or_nums:
        if isinstance(p, int):
            ruta = os.path.join(carpeta, FILE_NAMES[p])
            files = [ruta]
        else:
            files = sorted(glob.glob(p))
            if not files:
                files = [p]  # ruta directa

        for f in files:
            print(f"  Leyendo {f}...")
            t0 = time.time()
            try:
                df = pd.read_parquet(f)
                print(f"    -> {len(df):,} registros en {time.time()-t0:.1f}s")
                frames.append(df)
            except FileNotFoundError:
                print(f"    -> ERROR: no encontrado")
            except Exception as e:
                print(f"    -> ERROR: {e}")

    if not frames:
        print("ERROR: no se encontraron archivos.")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


# ============================================================
# PREPROCESAMIENTO (identico a train.py y a K-Means)
# ============================================================

def add_engineered_features(df):
    df = df.copy()
    df["bytes_per_pkt"] = df["tot_bytes"] / (df["tot_pkts"]  + 1e-6)
    df["src_ratio"]     = df["src_bytes"]  / (df["tot_bytes"] + 1e-6)
    df["pps"]           = df["tot_pkts"]   / (df["dur"]       + 1e-6)
    return df


def preprocess_test(df, bundle):
    """Aplica exactamente el mismo preprocesamiento que en entrenamiento."""
    encoders = bundle["encoders"]
    scaler   = bundle["scaler"]
    medians  = bundle["medians"]
    cat_cols = bundle["cat_cols"]
    num_cols = bundle["num_cols"]
    eng_cols = bundle["eng_cols"]
    log_cols = bundle["log_cols"]
    base_feat = bundle["base_feat"]
    all_num   = num_cols + eng_cols

    df = add_engineered_features(df[base_feat])
    df = df[cat_cols + all_num].copy()

    for col in cat_cols:
        le = encoders[col]
        df[col] = df[col].astype(str).fillna("_MISSING_")
        known = set(le.classes_)
        fb = "_MISSING_" if "_MISSING_" in known else le.classes_[0]
        df[col] = df[col].apply(lambda x: x if x in known else fb)
        df[col] = le.transform(df[col])

    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    for col in all_num:
        df[col] = df[col].fillna(medians.get(col, 0.0))

    return scaler.transform(df.astype(float).values)


# ============================================================
# REGLAS POST-PROCESO (mismos umbrales que K-Means)
# ============================================================

def aplicar_reglas(df, pred_bin, no_rules=False):
    """
    R1: UDP/CON dur < 0.0001 s  -> NORMAL  (DNS loopback/cache local)
    R2: TCP dur>300s, bytes/pkt<200, pps<0.5, pred==NORMAL -> BOTNET (IRC keepalive)
    """
    if no_rules:
        return pred_bin.copy(), 0, 0

    pred = pred_bin.copy()

    # R1 — DNS loopback (mismo umbral exacto que K-Means: 0.0001 s)
    r1 = ((df["proto"].values == "udp") &
          (df["state"].values == "CON") &
          (df["dur"].values < 0.0001))

    # R2 — IRC keepalive (mismo umbral exacto que K-Means)
    bpp_raw = df["tot_bytes"].values / (df["tot_pkts"].values + 1e-6)
    pps_raw = df["tot_pkts"].values  / (df["dur"].values       + 1e-6)
    r2 = ((df["proto"].values == "tcp") &
          (df["dur"].values > 300) &
          (bpp_raw < 200) &
          (pps_raw < 0.5) &
          (pred == 0))

    n_r1 = int(r1.sum())
    n_r2 = int(r2.sum())
    pred[r1] = 0
    pred[r2] = 1

    return pred, n_r1, n_r2


# ============================================================
# VISUALIZACIONES
# ============================================================

def plot_pca2d(X, pred_bin, plot_file):
    """PCA 2D con colores rojo/verde — identico a K-Means."""
    n_vis = min(25_000, len(X))
    idx   = np.random.choice(len(X), n_vis, replace=False)
    pca   = PCA(n_components=2, random_state=RANDOM_STATE)
    X2d   = pca.fit_transform(X[idx])
    var   = pca.explained_variance_ratio_ * 100
    labels = pred_bin[idx]

    colors = np.where(labels == 1, "tab:red", "tab:green")

    xlo, xhi = np.percentile(X2d[:,0], [0.5, 99.5])
    ylo, yhi = np.percentile(X2d[:,1], [0.5, 99.5])
    mv = ((X2d[:,0]>=xlo)&(X2d[:,0]<=xhi)&
          (X2d[:,1]>=ylo)&(X2d[:,1]<=yhi))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X2d[mv,0], X2d[mv,1], c=colors[mv],
               s=5, alpha=0.35, rasterized=True)

    legend_items = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="tab:red",
               markersize=9, label="BOTNET (pred)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="tab:green",
               markersize=9, label="NORMAL (pred)"),
    ]
    ax.legend(handles=legend_items, fontsize=9)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% varianza)")
    ax.set_title("Arbol de Decision - PCA 2D")
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"  Guardada: {plot_file}")


def plot_confusion(tn, fp, fn, tp, out_file):
    """Matriz de confusion con heatmap."""
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred. NORMAL', 'Pred. BOTNET'],
                yticklabels=['Real NORMAL',  'Real BOTNET'])
    ax.set_title('Matriz de Confusion - Arbol de Decision', fontsize=13)
    ax.set_ylabel('Valor Real', fontsize=11)
    ax.set_xlabel('Prediccion',  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Guardada: {out_file}")


def plot_arbol(bundle, out_file="arbol_decision_visualizacion.png",
               out_importance="arbol_feature_importance.png"):
    """
    Genera dos graficas del arbol:
      1. Primeros 4 niveles del arbol (estructura de decision)
      2. Importancia de features (Gini)
    """
    from sklearn.tree import plot_tree

    modelo    = bundle["modelo"]
    cat_cols  = bundle["cat_cols"]
    num_cols  = bundle["num_cols"]
    eng_cols  = bundle["eng_cols"]

    feature_names = cat_cols + num_cols + eng_cols
    labels_display = {
        "proto":         "proto (cod)",
        "dir":           "dir (cod)",
        "state":         "state (cod)",
        "dur":           "dur [log]",
        "stos":          "stos",
        "dtos":          "dtos",
        "tot_pkts":      "tot_pkts [log]",
        "tot_bytes":     "tot_bytes [log]",
        "src_bytes":     "src_bytes [log]",
        "bytes_per_pkt": "bytes/pkt [log]",
        "src_ratio":     "src_ratio",
        "pps":           "pps [log]",
    }
    feat_labels = [labels_display.get(f, f) for f in feature_names]

    # --- Arbol (primeros 4 niveles) ---
    fig, ax = plt.subplots(figsize=(36, 18))
    plot_tree(
        modelo,
        max_depth=4,
        feature_names=feat_labels,
        class_names=["NORMAL", "BOTNET"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=9,
        ax=ax,
        precision=3,
    )
    ax.set_title(
        "Arbol de Decision — Deteccion de Botnets (CTU-13)\n"
        "Primeros 4 niveles  |  Entrenado en escenarios 1,2,3,4,5,6,8,12  |  max_depth=15",
        fontsize=14, fontweight="bold", pad=16,
    )
    plt.tight_layout()
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {out_file}")

    # --- Importancia de features ---
    importances = modelo.feature_importances_
    idx   = np.argsort(importances)[::-1]
    top_n = min(12, len(feat_labels))

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if importances[i] > 0.1 else "#1f77b4" for i in idx[:top_n]]
    bars = ax2.barh(
        [feat_labels[i] for i in idx[:top_n]][::-1],
        [importances[i] for i in idx[:top_n]][::-1],
        color=colors[::-1],
        edgecolor="white", linewidth=0.5,
    )
    for bar, val in zip(bars, [importances[i] for i in idx[:top_n]][::-1]):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)

    ax2.set_xlabel("Importancia (Gini)", fontsize=11)
    ax2.set_title(
        "Importancia de Features — Arbol de Decision\nDeteccion de Botnets CTU-13",
        fontsize=12, fontweight="bold",
    )
    ax2.axvline(0.1, color="red", linestyle="--", alpha=0.4, label="Umbral 0.10")
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_importance, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {out_importance}")


def plot_roc(y_true, y_proba, out_file):
    """Curva ROC con AUC."""
    fpr_c, tpr_c, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_c, tpr_c, 'b-', lw=2, label=f'Arbol Decision (AUC = {auc:.4f})')
    ax.plot([0,1], [0,1], 'r--', lw=1, label='Clasificador aleatorio')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('FPR (Tasa de Falsos Positivos)', fontsize=11)
    ax.set_ylabel('TPR / Recall',                   fontsize=11)
    ax.set_title('Curva ROC - Deteccion de Botnets', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Guardada: {out_file}")
    return auc


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluacion del Arbol de Decision sobre datos CTU-13 o sinteticos")
    parser.add_argument("--model",         default="modelos/arbol_decision_modelo.joblib",
                        help="modelo guardado por train.py")
    parser.add_argument("--escenarios",    nargs="+", type=int, default=[9, 10, 13],
                        help="numeros de escenario CTU-13 a evaluar (default: 9 10 13)")
    parser.add_argument("--file",          nargs="+", default=None,
                        help="rutas directas a parquet (alternativa a --escenarios)")
    parser.add_argument("--carpeta",       default="CTU13",
                        help="carpeta con archivos CTU-13 (default: CTU13)")
    parser.add_argument("--no-background", action="store_true",
                        help="excluir flujos Background antes de evaluar")
    parser.add_argument("--no-rules",      action="store_true",
                        help="desactivar reglas post-proceso R1 y R2")
    parser.add_argument("--stats-file",    default="test_stats.txt",
                        help="archivo de salida de metricas")
    parser.add_argument("--plot-file",     default=None,
                        help="nombre base para las graficas (sin extension)")
    args = parser.parse_args()

    # Nombre base para graficas
    if args.plot_file is None:
        sufijo = "sin-reglas" if args.no_rules else "con-reglas"
        base = f"arbol_pca2d-{sufijo}"
    else:
        base = args.plot_file

    t_total = time.time()
    print("="*60)
    print("EVALUACION ARBOL DE DECISION - ESCENARIOS NO VISTOS")
    print("="*60)
    print(f"Inicio: {time.strftime('%H:%M:%S')}")

    # Cargar modelo
    print(f"\nCargando modelo: {args.model}")
    try:
        bundle = joblib.load(args.model)
    except FileNotFoundError:
        print("ERROR: modelo no encontrado. Ejecuta primero: python train.py")
        sys.exit(1)

    modelo = bundle["modelo"]

    # Cargar datos
    print("\n" + "="*60)
    print("CARGANDO DATOS DE PRUEBA")
    print("="*60)

    if args.file:
        df = load_files(args.file)
    else:
        df = load_files(args.escenarios, args.carpeta)

    has_labels = "label" in df.columns

    # Quitar background
    if args.no_background and has_labels:
        bg = df["label"].apply(is_background).sum()
        df = df[~df["label"].apply(is_background)].reset_index(drop=True)
        print(f"  Background eliminado: {bg:,} flujos")

    if has_labels:
        true_bot = df["label"].apply(is_botnet).astype(int)
        print(f"\n  Normal: {(true_bot==0).sum():,} | Botnet: {(true_bot==1).sum():,}")

    # Preprocesar
    print("\n" + "="*60)
    print("PREPROCESANDO (pipeline identico al entrenamiento)")
    print("="*60)
    t0 = time.time()
    X = preprocess_test(df, bundle)
    print(f"  Completado en {time.time()-t0:.1f}s  |  Shape: {X.shape}")

    # Predecir
    print("\nGenerando predicciones...")
    t0 = time.time()
    pred_raw  = modelo.predict(X)
    pred_proba = modelo.predict_proba(X)[:, 1]
    print(f"  Completado en {time.time()-t0:.1f}s")

    # Reglas post-proceso
    pred_bin, n_r1, n_r2 = aplicar_reglas(df, pred_raw, no_rules=args.no_rules)

    if args.no_rules:
        rules_line = "reglas post-proceso: desactivadas (--no-rules)"
    else:
        rules_line = f"reglas post-proceso: R1 (dns-local) corrigio {n_r1}, R2 (irc) activo {n_r2}"

    print(f"\n  {rules_line}")

    # Metricas
    stat_lines = []
    auc = None

    if has_labels:
        cm = confusion_matrix(true_bot, pred_bin)
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(true_bot, pred_bin, zero_division=0)
        rec  = recall_score(   true_bot, pred_bin, zero_division=0)
        f1   = f1_score(       true_bot, pred_bin, zero_division=0)
        tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        auc  = roc_auc_score(true_bot, pred_proba)

        stat_lines = [
            rules_line,
            "",
            f"precision:  {prec:.4f}",
            f"recall:     {rec:.4f}",
            f"f1:         {f1:.4f}",
            f"tpr:        {tpr:.4f}",
            f"fpr:        {fpr:.4f}",
            f"auc-roc:    {auc:.4f}",
            "",
            "matriz de confusion:",
            f"                 pred normal  pred botnet",
            f"real normal      {tn:>11,}  {fp:>11,}",
            f"real botnet      {fn:>11,}  {tp:>11,}",
        ]

        print("\n" + "="*60)
        print("RESULTADOS")
        print("="*60)
        for line in stat_lines:
            print(line)

    # Graficas
    print("\n" + "="*60)
    print("GENERANDO GRAFICAS")
    print("="*60)

    plot_pca2d(X, pred_bin, f"{base}.png")
    plot_arbol(bundle,
               out_file="arbol_decision_visualizacion.png",
               out_importance="arbol_feature_importance.png")

    if has_labels:
        plot_confusion(tn, fp, fn, tp, "matriz_confusion.png")
        plot_roc(true_bot, pred_proba, "curva_roc.png")

    # Guardar stats
    if stat_lines:
        with open(args.stats_file, "w") as f:
            f.write("\n".join(stat_lines) + "\n")
        print(f"\n  Metricas guardadas: {args.stats_file}")

    print("\n" + "="*60)
    print("EVALUACION COMPLETADA")
    print("="*60)
    print(f"  Tiempo total: {time.time()-t_total:.1f}s")
    print(f"  Fin: {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
