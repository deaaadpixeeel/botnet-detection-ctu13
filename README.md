# Botnet Detection Pipeline: K-Means vs Decision Tree

*[English version below](#english-version)*

Este proyecto consolida un pipeline de aprendizaje máquina para la detección de botnets y lo eleva a un marco de evaluación comparativa con estándares de ingeniería[cite: 1]. Evalúa el rendimiento de un modelo no supervisado (K-Means) y un modelo supervisado (Árbol de Decisión) sobre exactamente el mismo dataset y preprocesamiento[cite: 1].

Los modelos fueron entrenados y evaluados utilizando capturas reales de tráfico en formato binetflow correspondientes al dataset CTU-13[cite: 1].

## Requisitos e Instalación

Para ejecutar los scripts de entrenamiento y evaluación, se requiere Python 3.8+ y las dependencias listadas en el archivo requirements.txt.

Instala las dependencias ejecutando:
```bash
pip install -r requirements.txt


## Estructura del Repositorio

El repositorio está organizado en módulos para facilitar su ejecución y auditoría:

* /data/: Directorio destinado a alojar los archivos .parquet del dataset CTU-13 (no incluidos por restricciones de peso).
* /docs/: Contiene las evidencias visuales del análisis, incluyendo la matriz de confusión, curva ROC y visualizaciones PCA generadas en el reporte técnico.
* /models/: Almacena los modelos pre-entrenados exportados en formato .joblib.
* /src/arbol_decision/: Scripts fuente para el entrenamiento, evaluación y extracción de características (feature engineering) del modelo supervisado.
* /src/kmeans/: Scripts fuente para el pipeline de agrupamiento, pruebas en escenarios específicos y generación de tráfico sintético.

## Ejecución

Nota importante: Antes de ejecutar los scripts, asegúrate de colocar los archivos .parquet del dataset CTU-13 dentro del directorio /data/.

Para el Árbol de Decisión:
Navega al directorio correspondiente y ejecuta el entrenamiento o la prueba:

```bash
cd src/arbol_decision/
python train.py
python test.py

```

Para el modelo K-Means:
Navega al directorio correspondiente y ejecuta el entorno de evaluación:

```bash
cd src/kmeans/
python real_train_kmeans.py
python real_test_kmeans.py

```

---

# English Version

This project consolidates a machine learning pipeline for botnet detection, elevating it to a comparative evaluation framework with engineering standards. It evaluates the performance of an unsupervised model (K-Means) and a supervised model (Decision Tree) over the exact same dataset and preprocessing pipeline.

The models were trained and evaluated using real network traffic captures in binetflow format from the CTU-13 dataset.

## Requirements & Installation

To run the training and evaluation scripts, Python 3.8+ is required along with the dependencies listed in the requirements.txt file.

Install dependencies by running:

```bash
pip install -r requirements.txt

```

## Repository Structure

The repository is modularized to facilitate execution and auditing:

* /data/: Directory intended to host the .parquet files from the CTU-13 dataset (not included due to file size limits).
* /docs/: Contains visual evidence of the analysis, including confusion matrices, ROC curves, and PCA visualizations generated in the technical report.
* /models/: Stores the pre-trained models exported in .joblib format.
* /src/arbol_decision/: Source scripts for training, evaluation, and feature engineering of the supervised model.
* /src/kmeans/: Source scripts for the clustering pipeline, specific scenario testing, and synthetic traffic generation.

## Execution

Important Note: Before running the scripts, ensure that the .parquet files from the CTU-13 dataset are placed inside the /data/ directory.

For the Decision Tree:
Navigate to the directory and run the training or testing scripts:

```bash
cd src/arbol_decision/
python train.py
python test.py

```

For the K-Means model:
Navigate to the directory and run the evaluation environment:

```bash
cd src/kmeans/
python real_train_kmeans.py
python real_test_kmeans.py

```


