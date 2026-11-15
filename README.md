# Laboratorio de Minería de Datos - Customer Churn

Servicio de Machine Learning que estima la probabilidad de que un cliente de una empresa de telecomunicaciones abandone el servicio (*churn*) y devuelve un nivel de riesgo utilizable por otros sistemas (por ejemplo, un CRM o una campaña de retención).

Proyecto integrador de ISTEA, segundo cuatrimestre 2026.

## Contenido

1. [Problema y objetivo de ML](#1-problema-y-objetivo-de-ml)
2. [Arquitectura](#2-arquitectura)
3. [Estructura del proyecto](#3-estructura-del-proyecto)
4. [Instalación](#4-instalación)
5. [Datos y preprocesamiento](#5-datos-y-preprocesamiento)
6. [Entrenamiento y experimentos (MLflow)](#6-entrenamiento-y-experimentos-mlflow)
7. [Resultados y selección del modelo](#7-resultados-y-selección-del-modelo)
8. [Del Model Registry a la API](#8-del-model-registry-a-la-api)
9. [API REST](#9-api-rest)
10. [Tests](#10-tests)
11. [Docker y Docker Compose](#11-docker-y-docker-compose)
12. [Trazabilidad](#12-trazabilidad)
13. [Logging](#13-logging)
14. [Limitaciones conocidas](#14-limitaciones-conocidas)
15. [Problemas frecuentes](#15-problemas-frecuentes)

---

## 1. Problema y objetivo de ML

| | |
|---|---|
| **Tipo de problema** | Clasificación binaria supervisada |
| **Variable objetivo** | `Churn` (`Yes` / `No`) |
| **Salida principal** | Probabilidad estimada de abandono |
| **Salida derivada** | Nivel de riesgo `LOW` / `MEDIUM` / `HIGH` |
| **Fuera de alcance** | El CRM y las acciones comerciales posteriores a la predicción |

Ejemplo de respuesta del servicio:

```json
{
  "prediction": "CHURN",
  "probability": 0.974,
  "risk_level": "HIGH",
  "model_version": "1",
  "request_id": "71158d52-82b6-4beb-a5ed-975c5605cb4b"
}
```

## 2. Arquitectura

```
GitHub (código) ──► DVC + DagsHub (dataset y modelo exportado)
       │
       └──────────► MLflow + DagsHub (experimentos y Model Registry)
                          │
                          ▼
              customer-churn-model v1  ──export──►  models/ (DVC)
                                                        │
                                                        ▼
                                    FastAPI + Pydantic ─► pytest ─► Docker ─► Docker Compose
```

## 3. Estructura del proyecto

```
customer-churn-ml/
├── app/                    API FastAPI
│   ├── main.py             endpoints, logging y request_id
│   └── schemas.py          contratos de entrada/salida (Pydantic)
├── data/raw/               dataset (versionado con DVC, no con Git)
├── models/                 modelo exportado del Model Registry (DVC)
├── notebooks/              exploración inicial y EDA
├── scripts/
│   └── export_model.py     baja una versión del Model Registry a models/
├── src/
│   ├── data/               carga y partición train/test
│   ├── features/           preprocesamiento (ColumnTransformer)
│   ├── training/           experimentos y registro en MLflow
│   ├── evaluation/         métricas
│   └── inference/          carga del modelo, umbral y nivel de riesgo
├── tests/                  tests de inferencia y de la API
├── Dockerfile
├── compose.yaml
├── requirements.txt        entorno completo (desarrollo, entrenamiento, tests)
└── requirements-api.txt    dependencias mínimas de la API (imagen Docker)
```

## 4. Instalación

Repositorios: [GitHub](https://github.com/frannsanchez/customer-churn-ml) (código) y [DagsHub](https://dagshub.com/frannsanchez/customer-churn-ml) (datos, experimentos y modelos).

### Requisitos

| Requisito | Verificación | Notas |
|---|---|---|
| Git | `git --version` | |
| Python 3.14 | `python3 --version` | Es la versión con la que se entrenó el modelo |
| Docker Engine + Docker Compose | `docker --version`, `docker compose version` | En Windows: Docker Desktop. En Linux, para usarlo sin `sudo`: `sudo usermod -aG docker $USER` y volver a iniciar sesión |
| Cuenta de DagsHub y token | — | El token se genera en DagsHub → *User Settings* → *Tokens*. Para registrar runs o hacer `dvc push` hay que ser colaborador del repositorio |

El token es personal: no se comparte, no se sube al repositorio y no se muestra en el video.

### Entorno

```bash
git clone https://github.com/frannsanchez/customer-churn-ml.git
cd customer-churn-ml

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` es el entorno completo: API, entrenamiento, DVC, MLflow y tests. `requirements-api.txt` tiene solo lo que necesita la API, y es lo que instala la imagen Docker.

Todos los comandos de este README se ejecutan desde la raíz del repositorio y con el entorno virtual activado.

### Credenciales

No hay credenciales en el repositorio ni en la imagen Docker. Cada integrante usa su propio token de DagsHub.

**MLflow** (para entrenar o exportar modelos):

```bash
cp .env.example .env              # Windows: copy .env.example .env
```

Completar `.env`:

```
MLFLOW_TRACKING_URI=https://dagshub.com/frannsanchez/customer-churn-ml.mlflow
MLFLOW_TRACKING_USERNAME=<usuario de DagsHub>
MLFLOW_TRACKING_PASSWORD=<token de DagsHub>
```

**DVC** (para bajar datos y modelo). En DagsHub, el token se usa como access key y también como secret key:

```bash
dvc remote modify origin --local access_key_id <token>
dvc remote modify origin --local secret_access_key <token>
```

`.env` y `.dvc/config.local` están excluidos de Git. Aunque el repositorio de DagsHub sea público, `dvc pull` requiere un token.

### Recuperar datos y modelo

```bash
dvc pull
```

Descarga los archivos que no están en Git: el dataset (`data/raw/customer_churn_historical.csv`) y el modelo exportado (`models/customer-churn-model/`).

Para verificar: `dvc status` tiene que responder `Data and pipelines are up to date.`, y `models/customer-churn-model/` tiene que contener `model.skops` y `metadata.json`.

## 5. Datos y preprocesamiento

- **Dataset:** 7.043 clientes y 20 columnas: datos demográficos, servicios contratados, tipo de contrato, método de pago y cargos.
- **Target desbalanceado:** el 26,4 % de los clientes hace churn.
- **Limpieza** (`src/data/crear_dataset.py`): se descarta `customerID`, que no es un predictor válido, y se convierte `TotalCharges` a numérico. Tiene 26 nulos, que corresponden a clientes nuevos.
- **Partición:** 80/20 estratificada por `Churn`, con `random_state=42`. El set de test no se usa para entrenar.
- **Pipeline de scikit-learn** (`src/features/build_features.py`), el mismo en entrenamiento y en inferencia:

```
Categóricas: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
Numéricas:   SimpleImputer(median)        → StandardScaler
          → Estimador
```

El preprocesamiento y el estimador se guardan juntos como un único `Pipeline`. Por eso la API aplica exactamente las mismas transformaciones que se usaron al entrenar.

El EDA (dimensiones, tipos, nulos, distribución del target y asociación de variables con `Churn`) está en `notebooks/`.

## 6. Entrenamiento y experimentos (MLflow)

El entrenamiento se ejecuta desde Python, sin notebooks. Los experimentos están definidos en `MODEL_CONFIGS` (`src/training/train.py`) y cada ejecución queda registrada en MLflow, en DagsHub.

```bash
python -m src.training.train                                      # todos los experimentos
python -m src.training.train --experiment exp12_logistic_regression_all_balanced
python -m src.training.train --experiment <nombre> --register     # además lo registra en el Model Registry
```

Requieren el `.env` configurado. Cada ejecución crea runs nuevos en DagsHub. `--register` crea una versión nueva en el Model Registry: usarlo solo cuando el equipo decida cambiar de modelo.

Cada run registra:
- **Parámetros:** las variables usadas y los hiperparámetros del estimador.
- **Métricas:** precision, recall, F1, ROC-AUC y la matriz de confusión desglosada (`tn`, `fp`, `fn`, `tp`).
- **Modelo:** el pipeline completo, en formato `skops`.
- **Tags de trazabilidad:** `mlflow.source.git.commit` (commit del código) y `data_md5` (hash DVC del dataset).

Los experimentos se pueden ver en DagsHub, en la pestaña **Experiments**: https://dagshub.com/frannsanchez/customer-churn-ml

## 7. Resultados y selección del modelo

### Métrica prioritaria

Un **falso negativo** es un cliente que estaba por irse y el modelo lo clasificó como estable: se pierde sin que se haya intentado retenerlo. Un **falso positivo**, en cambio, cuesta una acción de retención (un llamado o un descuento) dirigida a alguien que no se iba a ir.

Como perder un cliente suele costar más que una acción de retención, se prioriza el **recall** (la proporción de churners detectados). Se lo acompaña con el **F1**, para no aceptar modelos que logren buen recall marcando a todos como churn, y con el **ROC-AUC**, que mide qué tan bien el modelo ordena a los clientes según su riesgo. Accuracy no se usa: con un 74 % de clientes que no hacen churn, un modelo que siempre responde "No" ya obtiene un 74 %.

### Comparación de experimentos (set de test, 1.409 clientes, 372 churners)

| Experimento | Modelo | Variables | Precision | Recall | F1 | ROC-AUC | FN |
|---|---|---|---|---|---|---|---|
| exp2 | DummyClassifier (baseline) | 3 | 0.000 | 0.000 | 0.000 | 0.500 | 372 |
| exp1 | DecisionTree | 3 | 0.000 | 0.000 | 0.000 | 0.550 | 372 |
| exp3 | DecisionTree | 4 | 0.300 | 0.290 | 0.295 | 0.524 | 264 |
| exp4 | LogisticRegression | 4 | 0.000 | 0.000 | 0.000 | 0.540 | 372 |
| exp5 | DecisionTree | 5 | 0.419 | 0.422 | 0.420 | 0.606 | 215 |
| exp6 | DecisionTree (+tenure) | 6 | 0.352 | 0.387 | 0.369 | 0.566 | 228 |
| exp7 | DecisionTree (+Contract) | 6 | 0.421 | 0.403 | 0.412 | 0.602 | 222 |
| exp8 | DecisionTree (+InternetService) | 6 | 0.382 | 0.387 | 0.385 | 0.581 | 228 |
| exp9 | DecisionTree (Contract en lugar de PaperlessBilling) | 5 | 0.420 | 0.395 | 0.407 | 0.600 | 225 |
| exp10 | RandomForest | 5 | 0.441 | 0.320 | 0.371 | 0.663 | 253 |
| exp11 | LogisticRegression | todas (19) | **0.663** | 0.449 | 0.535 | **0.812** | 205 |
| **exp12** | **LogisticRegression, balanced** | **todas (19)** | 0.480 | 0.720 | 0.576 | **0.812** | 104 |
| exp13 | DecisionTree (max_depth=5), balanced | todas (19) | 0.465 | **0.731** | 0.568 | 0.790 | **100** |
| exp14 | RandomForest, balanced | todas (19) | 0.507 | 0.688 | **0.584** | 0.805 | 116 |
| exp15 | RandomForest ajustado, balanced_subsample | todas (19) | 0.485 | 0.726 | 0.581 | 0.802 | 102 |

### Qué muestran los experimentos

1. **Con pocas variables (exp1 a exp10), ningún modelo supera ampliamente al baseline.** Varios predicen "No" para todos los clientes (recall 0), igual que el Dummy.
2. **Usar todas las variables es el cambio más importante**: el ROC-AUC sube de 0.66 a 0.81 (exp10 → exp11).
3. **`class_weight="balanced"` compensa el desbalance de clases**: con el mismo modelo, el recall pasa de 0.45 a 0.72 y los falsos negativos bajan de 205 a 104 (exp11 → exp12).
4. Con todas las variables y balanceo, **los cuatro modelos rinden parecido** (F1 entre 0.57 y 0.58).

### Modelo seleccionado: exp12 (LogisticRegression con todas las variables y `class_weight="balanced"`)

- **Mejor ROC-AUC (0.812)**: es el que mejor ordena a los clientes por riesgo, y eso es lo que usan los niveles LOW/MEDIUM/HIGH.
- **Detecta al 72 % de los clientes que se van**. Está a menos de 1,1 puntos del mejor recall (exp13), que tiene peor ROC-AUC.
- **Es más simple e interpretable** que los modelos de ensamble: los coeficientes permiten explicar qué variables aumentan el riesgo. Ante un rendimiento similar, se elige la solución más simple.
- **Matriz de confusión en test:** TN 747, FP 290, FN 104, TP 268.

Está registrado en el Model Registry como **`customer-churn-model` versión 1**.

## 8. Del Model Registry a la API

La API no se conecta a MLflow en tiempo de ejecución. La versión registrada se exporta una vez a `models/` y se versiona con DVC.

Para cambiar el modelo que sirve la API:

```bash
python -m src.training.train --experiment <nombre> --register   # 1. registrar (crea la versión N)
python -m scripts.export_model --version <N>                    # 2. exportar a models/
dvc add models/customer-churn-model && dvc push                 # 3. versionar el modelo
git add models/customer-churn-model.dvc && git commit -m "..."  # 4. fijar la versión en Git
docker compose up --build -d                                    # 5. reconstruir la imagen
```

El export genera, junto al modelo, un `metadata.json` con la versión registrada, el `run_id`, el commit de Git y el hash de los datos.

Esta decisión hace que los tests y la imagen Docker funcionen sin credenciales y que la imagen no contenga secretos. Además, la versión del modelo que sirve la API queda fijada en Git, a través del archivo `.dvc`.

## 9. API REST

La documentación interactiva (Swagger) está en **http://localhost:8000/docs**. La raíz `/` redirige ahí.

### Probar desde Swagger

1. Desplegar un endpoint, hacer click en **Try it out** y después en **Execute**. La respuesta aparece debajo, en **Responses**.
2. En `POST /predict` ya viene cargado un cliente de ejemplo (contrato mensual, fibra óptica, 2 meses de antigüedad), que devuelve riesgo `HIGH`.
3. Algunas pruebas:
   - `"Contract": "Two year"`, `"tenure": 60`, `"TotalCharges": 5000` → el riesgo baja a `LOW`.
   - `"Contract": "Weekly"` → error **422**: Pydantic rechaza el valor.
4. Al final de la página, en **Schemas**, están los contratos con los valores permitidos para cada campo.

En Windows conviene probar desde Swagger antes que con `curl`, porque PowerShell maneja las comillas de otra forma.

### `GET /health`

Indica si el servicio está operativo y qué modelo tiene cargado.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "customer-churn-model",
  "model_version": "1",
  "run_id": "4261771b72fa42aca608cf1fa3867b3e"
}
```

Devuelve **503** con `"status": "unavailable"` si el modelo no se pudo cargar.

### `POST /predict`

Recibe las 19 características de un cliente, con los mismos nombres que las columnas del dataset.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No",
       "tenure":2,"PhoneService":"Yes","MultipleLines":"No",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
       "StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check","MonthlyCharges":95.5,"TotalCharges":190.2}'
```

| Campo de respuesta | Descripción |
|---|---|
| `prediction` | `CHURN` si `probability ≥ 0.50`; si no, `NO_CHURN` |
| `probability` | Probabilidad estimada de churn (de 0 a 1) |
| `risk_level` | `LOW`, `MEDIUM` o `HIGH` (ver tabla siguiente) |
| `model_version` | Versión del Model Registry que generó la predicción |
| `request_id` | Identificador del request. Se toma del header `X-Request-ID` si viene; si no, se genera uno |

### Niveles de riesgo

Los umbrales se definieron observando cómo se distribuyen las probabilidades del modelo en el set de test:

| Nivel | Probabilidad | % de clientes | Churn real del grupo | % del total de churners |
|---|---|---|---|---|
| LOW | < 0.40 | 52 % | 9 % | 19 % |
| MEDIUM | 0.40 – 0.70 | 26 % | 32 % | 32 % |
| HIGH | ≥ 0.70 | 21 % | 61 % | 50 % |

El grupo **HIGH** concentra la mitad de los clientes que se van en solo un 21 % de la base. Es el grupo prioritario para una campaña de retención. El grupo LOW tiene un churn real de 9 %, tres veces menor que el promedio.

El umbral de decisión de 0.50 se mantiene porque el modelo se entrenó con `class_weight="balanced"`: con ese umbral, el modelo ya favorece el recall (0.72).

### Validación y errores

Los contratos están definidos con Pydantic en `app/schemas.py`. Las variables categóricas solo aceptan los valores del dataset, las numéricas tienen rangos (`tenure` entre 0 y 120, cargos ≥ 0) y se rechazan los campos desconocidos (por ejemplo, `customerID`). `TotalCharges` es opcional: si falta, el pipeline lo imputa.

| Código | Cuándo |
|---|---|
| 200 | Predicción correcta |
| 422 | La entrada no cumple el contrato: falta un campo, el tipo es incorrecto, el valor no está permitido o hay un campo extra |
| 503 | El modelo no está cargado |
| 500 | Error inesperado durante la inferencia (queda registrado en el log con su `request_id`) |

Ejemplo de error 422 con `"Contract": "Weekly"`:

```json
{
  "detail": [{
    "type": "literal_error",
    "loc": ["body", "Contract"],
    "msg": "Input should be 'Month-to-month', 'One year' or 'Two year'",
    "input": "Weekly"
  }]
}
```

## 10. Tests

```bash
pytest
```

Resultado esperado: `26 passed`. Los tests usan el modelo real, así que requieren haber hecho `dvc pull`.

| Archivo | Qué cubre |
|---|---|
| `tests/test_predictor.py` | Carga del modelo y de su metadata; error si falta el modelo; estructura y tipos de una predicción válida; coherencia entre el umbral y la predicción; imputación de `TotalCharges`; sanity check de negocio (un contrato largo implica menos riesgo); límites de los niveles de riesgo |
| `tests/test_api.py` | `/health`; `/predict` válido y estructura de la respuesta; propagación del `request_id`; coincidencia de versión entre `/health` y `/predict`; Swagger; entradas inválidas (categoría inexistente, valor fuera de rango, tipo incorrecto, campo faltante, campo extra, body vacío) → 422; servicio sin modelo → 503 |

## 11. Docker y Docker Compose

La imagen incluye solo lo necesario para servir el modelo: Python 3.14 slim, `requirements-api.txt`, `app/`, `src/inference/` y el modelo exportado. No incluye datos, credenciales ni el código de entrenamiento. Corre con un usuario sin privilegios y tiene `HEALTHCHECK` sobre `/health`.

```bash
docker compose up --build -d      # construir y levantar
docker compose ps                 # estado (debe decir "healthy")
docker compose logs -f            # logs en vivo
docker compose down               # detener
```

También se puede construir la imagen por separado con `docker build -t customer-churn-api:local .`. Después de cambiar el código hay que volver a correr `docker compose up --build -d`.

Para levantar la API sin Docker, durante el desarrollo: `uvicorn app.main:app --reload`.

| Variable de entorno | Default | Uso |
|---|---|---|
| `API_PORT` | `8000` | Puerto publicado en el host (`API_PORT=8080 docker compose up -d`) |
| `LOG_LEVEL` | `INFO` | Nivel de logging |
| `MODEL_DIR` | `/app/models/customer-churn-model` | Carpeta del modelo dentro del contenedor |

Flujo completo, desde cero:

```
git clone → pip install → dvc pull → pytest → docker compose up --build → POST /predict
```

## 12. Trazabilidad

Desde cualquier predicción se puede reconstruir su origen:

```
Respuesta de /predict        model_version = "1"
        ↓
/health                      run_id = 4261771b72fa42aca608cf1fa3867b3e
        ↓
MLflow (DagsHub)             run exp12_logistic_regression_all_balanced
        ↓                    tags: mlflow.source.git.commit = 4ed5c0a
Git                          commit 4ed5c0a (código con el que se entrenó)
        ↓                    tags: data_md5 = 178571792c566f067e3d16b5fb913026
DVC                          data/raw/customer_churn_historical.csv.dvc (mismo md5)
```

Esa misma información está en `models/customer-churn-model/metadata.json`.

## 13. Logging

La API registra el inicio del servicio (con el modelo y la versión cargados) y, por cada request, el `request_id`, el método, la ruta, el código HTTP y la latencia. En cada predicción registra además el resultado, la probabilidad, el nivel de riesgo y la versión del modelo. Los errores se registran con su traceback. El header `X-Request-ID` de la respuesta permite asociar cada predicción con su línea de log.

```
INFO - churn-api - Servicio iniciado - modelo customer-churn-model v1 (run_id=4261771b...)
INFO - churn-api - request_id=16e87779-... prediction=CHURN probability=0.9740 risk_level=HIGH model_version=1
INFO - churn-api - request_id=16e87779-... method=POST path=/predict status=200 latency_ms=44.0
```

## 14. Limitaciones conocidas

- **Probabilidades no calibradas:** `class_weight="balanced"` hace que el modelo sobreestime la probabilidad de churn. Las probabilidades sirven para ordenar a los clientes por riesgo, pero no deben leerse como probabilidades reales.
- **Umbrales definidos sobre el set de test:** lo ideal sería un set de validación separado. Con más datos se podrían ajustar usando validación cruzada.
- **Costos estimados en forma cualitativa:** la priorización del recall no se basa en costos reales de retención ni en el valor de cada cliente.
- **Un solo dataset histórico:** todavía no se midió el desempeño con datos nuevos ni el drift. Ambas cosas están previstas para la entrega final.

## 15. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `permission denied ... docker.sock` | El usuario no está en el grupo `docker`, o todavía no volvió a iniciar sesión después de agregarlo. Ver [Requisitos](#requisitos). No usar `sudo docker`. |
| `dvc pull` responde `Unable to locate credentials`, 401 o 403 | Falta configurar el token de DVC (ver [Credenciales](#credenciales)): que el repositorio sea público no alcanza. Si el token ya está configurado, verificar que el usuario sea colaborador en DagsHub. |
| `port is already allocated` al levantar Docker | El puerto 8000 está ocupado. Usar otro: `API_PORT=8080 docker compose up -d` y abrir http://localhost:8080/docs |
| `pytest` falla con `FileNotFoundError` en `models/` | Falta hacer `dvc pull`. |
| `/health` devuelve 503 | El contenedor no encuentra el modelo. Hacer `dvc pull` y reconstruir la imagen con `docker compose up --build -d`. |
| La terminal queda trabada mostrando `(END)` después de `git log` | Es el paginador de Git: salir con `q`. |
