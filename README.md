# Botnet Detection Pipeline: K-Means vs Decision Tree

[English version below](#english-version)

Este proyecto consolida un pipeline de aprendizaje máquina para la detección de botnets y lo eleva a un marco de evaluación comparativa con estándares de ingeniería. Evalúa el rendimiento de un modelo no supervisado (K-Means) y un modelo supervisado (Árbol de Decisión) sobre exactamente el mismo dataset y preprocesamiento, de modo que las diferencias en las métricas reflejan únicamente la naturaleza algorítmica de cada enfoque.

Los modelos fueron entrenados y evaluados utilizando capturas reales de tráfico en formato binetflow correspondientes al dataset CTU-13 (Universidad Técnica Checa, 2011).

## Diferencia Fundamental

| Aspecto | K-Means | Árbol de Decisión |
|---|---|---|
| Tipo de aprendizaje | No supervisado | Supervisado |
| Uso de etiquetas en entrenamiento | No | Sí |
| Interpretabilidad | Clusters y centroides | Reglas explícitas (978 hojas) |
| Uso en producción sin etiquetas | Viable | Requiere datos etiquetados |

## Metodología Compartida

Ambos modelos operan bajo estrictas condiciones de control para asegurar una comparativa justa:

* *Datos de Entrenamiento:* Escenarios 1, 2, 3, 4, 5, 6, 8, 12.
* *Datos de Prueba (No Vistos):* Escenarios 9, 10, 13.
* *Balanceo:* Muestreo 50/50 (botnet/normal) con n=150,000 en entrenamiento. Los flujos "Background" se excluyen en la evaluación final.
* *Ingeniería de Características:* Se utilizan 9 features base y 3 diseñadas (bytes_per_pkt, src_ratio, pps). Transformación log1p en columnas sesgadas y StandardScaler global.
* *Reglas Post-Proceso (R1 y R2):* Ambos modelos aplican reglas determinísticas idénticas para corregir falsos positivos (R1: DNS loopback) y recuperar falsos negativos (R2: IRC keepalive).

## Resultados Comparativos (Escenarios 9, 10, 13)

| Modelo | Precision | Recall | F1-Score | FPR |
|---|---|---|---|---|
| K-Means + R1 + R2 | 93.2 % | 69.6 % | 0.797 | 18.9 % |
| Árbol de Decisión + R1 + R2 | 97.5 % | 66.5 % | 0.791 | 6.3 % |

Nota: El Árbol de Decisión reduce la tasa de falsos positivos (FPR) a la mitad y logra un AUC-ROC de 0.9021 frente al 0.7356 del K-Means.

## Requisitos e Instalación

Se requiere Python 3.8+ y las dependencias listadas en el archivo requirements.txt.

bash
pip install -r requirements.txt


## Estructura del Repositorio
 * /data/: Archivos .parquet del dataset CTU-13 (ignorado en Git por peso).
 * /docs/: Evidencias visuales (matrices de confusión, curvas ROC, PCA 2D).
 * /models/: Modelos exportados en formato .joblib.
 * /src/arbol_decision/: Scripts de entrenamiento y evaluación del modelo supervisado.
 * /src/kmeans/: Scripts del modelo no supervisado y generador de escenarios sintéticos.
## Ejecución
Nota importante: Colocar los archivos .parquet del dataset CTU-13 en /data/ antes de ejecutar.
Para evaluar el Árbol de Decisión:
bash
cd src/arbol_decision/
python train.py
python test.py --escenarios 9 10 13 --no-background


Para evaluar el K-Means:
bash
cd src/kmeans/
python real_train_kmeans.py
python real_test_kmeans.py --model ../../models/botnet_kmeans_model.joblib --no-background


<a name="english-version"></a>
# English Version
This project consolidates a machine learning pipeline for botnet detection, elevating it to a comparative evaluation framework with engineering standards. It evaluates the performance of an unsupervised model (K-Means) and a supervised model (Decision Tree) over the exact same dataset and preprocessing pipeline.
The models were trained and evaluated using real network traffic captures in binetflow format from the CTU-13 dataset.
## Core Differences
| Feature | K-Means | Decision Tree |
|---|---|---|
| Learning Type | Unsupervised | Supervised |
| Label Usage | No | Yes |
| Interpretability | Clusters & Centroids | Explicit Rules |
| Unlabeled Production Use | Viable | Requires Labeled Data |
## Shared Methodology
Both models operate under strict control conditions to ensure a fair comparison:
 * *Training Data:* Scenarios 1, 2, 3, 4, 5, 6, 8, 12.
 * *Test Data (Unseen):* Scenarios 9, 10, 13.
 * *Feature Engineering:* 9 base features and 3 engineered (bytes_per_pkt, src_ratio, pps). log1p transformation on skewed columns and StandardScaler applied globally.
 * *Post-Processing Rules (R1 & R2):* Both models apply identical deterministic rules to correct false positives (R1: DNS loopback) and recover false negatives (R2: IRC keepalive).
## Comparative Results (Scenarios 9, 10, 13)
| Model | Precision | Recall | F1-Score | FPR |
|---|---|---|---|---|
| K-Means + R1 + R2 | 93.2 % | 69.6 % | 0.797 | 18.9 % |
| Decision Tree + R1 + R2 | 97.5 % | 66.5 % | 0.791 | 6.3 % |
## Requirements & Installation
Python 3.8+ is required along with the dependencies in requirements.txt.
bash
pip install -r requirements.txt


## Execution
Important Note: Place the CTU-13 .parquet files in the /data/ directory before running.
Decision Tree Execution:
bash
cd src/arbol_decision/
python train.py
python test.py --escenarios 9 10 13 --no-background


K-Means Execution:
bash
cd src/kmeans/
python real_train_kmeans.py
python real_test_kmeans.py --model ../../models/botnet_kmeans_model.joblib --no-background




```
