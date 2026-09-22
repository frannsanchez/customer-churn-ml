# Laboratorio de Minería de Datos - Customer Churn

## Descripción del proyecto

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning capaz de predecir el abandono de clientes (Customer Churn). El modelo busca predecir si un cliente abandonará o no el servicio.

## Objetivo de Machine Learning

La variable objetivo (target) es `Churn`, que representa si el cliente abandonó o no el servicio.

El modelo genera una predicción binaria:

- `No`: el cliente no abandona.
- `Yes`: el cliente abandona.

El proyecto contempla diferentes experimentos y modelos de clasificación, comparando sus resultados mediante distintas métricas de evaluación.

## Instalación

### Requisitos

- Python
- Git
- DVC

### Instalación de dependencias

El proyecto utiliza las siguientes librerías y herramientas:

- DVC
- MLflow
- pandas
- scikit-learn

Para instalar las dependencias, ejecutar:
pip install -r requirements.txt

## Ejecución

El entrenamiento se encuentra implementado en `src/training/train.py` y puede ejecutarse desde la consola sin depender del notebook.

### Ejecutar un experimento

El entrenamiento permite ejecutar un experimento específico. Los experimentos desarrollados durante la etapa de exploración se conservan y se registran mediante MLflow para su comparación y trazabilidad.

Para ejecutar un experimento específico:
python -m src.training.train --experiment "Nombre"
*Donde 'Nombre' corresponde al nombre del experimento definido en MODEL_CONFIGS.*

Para ejecutar todos los experimentos:
python -m src.training.train

## Estructura del proyecto

El proyecto se organiza separando los datos, la experimentación y el código reutilizable:

customer-churn-ml/
│
├── app/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
│
├── models/
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── evaluation/
│   ├── inference/
│   └── training/
│
├── tests/
│
├── .dvc/
├── .dvcignore
├── .gitignore
├── README.md
└── requirements.txt

## Descripción de las carpetas

data/: contiene los datos utilizados por el proyecto. Los datos se gestionan mediante DVC.
notebooks/: contiene los notebooks utilizados para la exploración y experimentación.
src/: contiene el código fuente reutilizable del proyecto.
    src/data/: carga y preparación inicial de los datos.
    src/features/: construcción del preprocesamiento y transformación de variables.
    src/evaluation/: cálculo y presentación de las métricas de evaluación.
    src/training/: entrenamiento de los modelos y seguimiento de experimentos.
    src/inference/: espacio destinado a la lógica de inferencia.
models/: espacio destinado a los artefactos de los modelos.
tests/: espacio destinado a las pruebas automatizadas.
app/: espacio destinado a la aplicación del proyecto.
.dvc/: configuración utilizada para la gestión de datos mediante DVC.
requirements.txt: dependencias del proyecto.

## Reproducibilidad y gestión de datos

El proyecto utiliza Git y DVC para mantener separados el código y los datos.

- **Git/GitHub**: versiona el código, la configuración y los metadatos del proyecto.
- **DVC**: permite versionar y recuperar el dataset.
- **DagsHub**: se utiliza como remote para el almacenamiento de los datos gestionados mediante DVC.
- **MLflow**: se utiliza para registrar y realizar el seguimiento de los experimentos de Machine Learning.

### Recuperación de los datos

Luego de clonar el repositorio, los datos se recuperan mediante DVC:
dvc pull

El flujo de reproducción es:

Git → código y metadatos
        ↓
DVC → datos versionados
        ↓
Entrenamiento → modelo y métricas
        ↓
MLflow → seguimiento de experimentos

De esta manera, el proyecto puede reconstruir el entorno de trabajo a partir del código versionado y de la versión correspondiente de los datos.

## Crear el entorno virtual

python -m venv .venv
.venv\Scripts\activate







