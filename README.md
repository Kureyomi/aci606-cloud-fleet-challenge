# ACI606 — Cloud Fleet Challenge

Servicio de telemetría persistente para una flota virtual de robots, con inferencia de condición mediante Machine Learning, API REST, métricas operacionales y dashboard web.

## Servicio desplegado

- **Dashboard:** https://aci606-cloud-fleet-matias.onrender.com/dashboard
- **API / OpenAPI:** https://aci606-cloud-fleet-matias.onrender.com/docs
- **Health check:** https://aci606-cloud-fleet-matias.onrender.com/health
- **Métricas:** https://aci606-cloud-fleet-matias.onrender.com/metrics

**Plataforma:** Render Web Service (Python 3, plan Free).  
**Base de datos:** Neon PostgreSQL, región AWS US East 2 (Ohio).  
**Modelo:** pipeline Random Forest serializado en `models/model.joblib`.

> La URL de conexión `DATABASE_URL` se configura como variable de entorno en Render y en el entorno de desarrollo. Nunca se debe guardar una contraseña ni una cadena de conexión real en Git.

## Arquitectura

```text
Simulador de flota (5 robots)
          |
          | HTTP POST /telemetry
          v
Render — FastAPI + Uvicorn
          |                 |
          | SQLAlchemy      | scikit-learn
          v                 v
Neon PostgreSQL        models/model.joblib
          |
          v
Dashboard / API / métricas
```

- El simulador actúa como cliente y generador de carga.
- FastAPI recibe y valida la telemetría.
- SQLAlchemy almacena los registros en PostgreSQL gestionado por Neon.
- El modelo clasifica la condición del robot como `NORMAL`, `WARNING` o `CRITICAL`.
- El dashboard consulta los endpoints de la API.

## Endpoints

| Método | Ruta | Propósito |
|---|---|---|
| GET | `/health` | Estado de servicio, base de datos y modelo |
| POST | `/telemetry` | Ingesta de una muestra de telemetría |
| GET | `/robots` | Última muestra de cada robot |
| GET | `/robots/{robot_id}/history` | Historial completo de un robot |
| POST | `/predict` | Predicción de condición del robot |
| GET | `/metrics` | Solicitudes, errores, latencia y uptime |
| GET | `/dashboard` | Panel web de operación |
| GET | `/docs` | Documentación interactiva OpenAPI |

La respuesta de `POST /predict` contiene `robot_id`, `predicted_status`, `confidence`, `model_version` e `inference_ms`. Los datos inválidos se rechazan mediante respuestas HTTP 4xx (por ejemplo, HTTP 422 para batería fuera de rango).

## Modelo de Machine Learning

### Datos y preparación

La evidencia registrada en `docs/evidencia_modelo.txt` corresponde a **15.000 filas**, procedentes de **5 flotas**. Se utilizaron seis características:

- `speed`
- `battery`
- `motor_temp`
- `vibration`
- `motor_current`
- `obstacle_distance`

Variable objetivo: `status`, con clases `NORMAL`, `WARNING` y `CRITICAL`.

Distribución observada:

| Clase | Filas | Proporción |
|---|---:|---:|
| CRITICAL | 10.258 | 68,4 % |
| NORMAL | 2.612 | 17,4 % |
| WARNING | 2.130 | 14,2 % |
| **Total** | **15.000** | **100 %** |

La clase CRITICAL es mayoritaria; por ello se utiliza **F1 macro** como métrica principal, para dar el mismo peso a las tres clases.

### Separación y comparación

Se utilizó `GroupShuffleSplit` agrupando por `fleet_id`, para evaluar el modelo con flotas no vistas durante el entrenamiento. La evidencia guardada registra 9.000 filas de entrenamiento y 6.000 de prueba.

| Clasificador | F1 macro |
|---|---:|
| Baseline de clase mayoritaria | 0,2455 |
| Regresión Logística | 0,9069 |
| Random Forest | **0,9076** |

Se seleccionó **Random Forest**, por obtener el mayor F1 macro en la comparación registrada. La diferencia frente a Regresión Logística es pequeña (0,0007), por lo que ambos modelos tuvieron un rendimiento muy cercano en esta prueba.

### Matrices de confusión

Orden de filas y columnas: `NORMAL`, `WARNING`, `CRITICAL`.

**Baseline de clase mayoritaria**

```text
[[   0,    0, 1334],
 [   0,    0, 1169],
 [   0,    0, 3497]]
```

**Regresión Logística**

```text
[[1177,  157,    0],
 [  97, 1041,   31],
 [   0,  114, 3383]]
```

**Random Forest**

```text
[[1169,  165,    0],
 [  96, 1015,   58],
 [   0,   65, 3432]]
```

La evidencia completa, incluidos los reportes por clase y la matriz de cada clasificador, está en `docs/evidencia_modelo.txt`. Los resultados resumidos están en `models/metrics.json`.

## Preparar el entorno de desarrollo

Requisitos: Python y acceso a una instancia PostgreSQL (Neon).

1. Clonar el repositorio y entrar en la carpeta del proyecto.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Configurar las variables de entorno en el entorno local o en Codespaces. La cadena debe usar el controlador Psycopg 3:

```text
DATABASE_URL=postgresql+psycopg://...
MODEL_PATH=models/model.joblib
```

No subir el valor real de `DATABASE_URL` al repositorio, a capturas ni a documentos compartidos.

4. Iniciar la API:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

5. Abrir `http://127.0.0.1:8000/dashboard` o `http://127.0.0.1:8000/docs`.

## Ejecutar el simulador

En otra terminal:

```bash
python simulator/fleet_simulator.py --api http://127.0.0.1:8000 --robots 5 --interval 1
```

El simulador envía telemetría de cinco robots. Para una ejecución limitada, se puede agregar `--steps 10`; para detener una ejecución continua, usar `Ctrl+C`.

## Reproducir el entrenamiento

El script `ml/train_model.py` lee los archivos `data/training*.csv`. Para construir cinco archivos de 3.000 muestras cada uno con semillas distintas:

```bash
python simulator/generate_dataset.py --samples 3000 --seed 606 --output data/training_fleet_01.csv
python simulator/generate_dataset.py --samples 3000 --seed 607 --output data/training_fleet_02.csv
python simulator/generate_dataset.py --samples 3000 --seed 608 --output data/training_fleet_03.csv
python simulator/generate_dataset.py --samples 3000 --seed 609 --output data/training_fleet_04.csv
python simulator/generate_dataset.py --samples 3000 --seed 610 --output data/training_fleet_05.csv
python ml/train_model.py
```

El entrenamiento compara baseline mayoritario, Regresión Logística y Random Forest, guarda el pipeline en `models/model.joblib` y los resultados resumidos en `models/metrics.json`. Si se regeneran los CSV con otras semillas, las métricas pueden cambiar.

## Pruebas de integración (smoke test)

Prueba local básica:

```bash
python tests/smoke_test.py --api http://127.0.0.1:8000
```

Pruebas adicionales del contrato de predicción, historial y validación de datos:

```bash
python tests/smoke_test.py --api http://127.0.0.1:8000 --expect-predict --check-history --check-invalid
```

Contra el servicio desplegado:

```bash
python tests/smoke_test.py --api https://aci606-cloud-fleet-matias.onrender.com --expect-predict --check-history --check-invalid
```

En la validación realizada contra Render, las ocho comprobaciones pasaron: `/health`, `/dashboard`, `/telemetry`, `/robots`, historial, `/predict`, rechazo de dato inválido (HTTP 422) y `/metrics`. También se verificaron 120 registros por cada robot R01–R05 en Neon antes y después de reiniciar FastAPI.

Las pruebas de smoke test insertan una muestra con el identificador `TEST-SMOKE-01`; por ello puede aparecer temporalmente como un robot adicional en el dashboard y en la base de datos.

## Despliegue en Render

Configuración utilizada para el Web Service:

- **Runtime:** Python 3.
- **Build Command:** `pip install -r requirements.txt`.
- **Start Command:** `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- **Variables de entorno:** `DATABASE_URL` y `MODEL_PATH=models/model.joblib`.
- **Almacenamiento:** Neon PostgreSQL; no se depende del disco efímero de Render para conservar telemetría.

Render ejecuta el servicio web; Neon proporciona la base PostgreSQL persistente. El plan gratuito de Render puede suspender el servicio después de inactividad, por lo que puede producirse un arranque en frío.

## Trazabilidad Git

- Rama de trabajo: `feature/matias-smoke-tests`.
- Pull Request #1: [Feature/dashboard — Agux](https://github.com/Kureyomi/aci606-cloud-fleet-challenge/pull/1).
- Pull Request #2: [Smoke tests — XogAeternum](https://github.com/Kureyomi/aci606-cloud-fleet-challenge/pull/2).

**Estado al redactar este README:** ambos Pull Requests están abiertos y pendientes de integración en `Kureyomi/main`. La rama personal contiene los cambios de pruebas; la integración final en el repositorio del grupo requiere aprobación de alguien con permisos de escritura.

## Estructura principal

- `app/`: aplicación FastAPI y conexión a PostgreSQL.
- `dashboard/`: interfaz web.
- `simulator/`: simulador de flota y generador de datasets.
- `ml/`: pipeline de entrenamiento.
- `models/`: modelo serializado y métricas.
- `docs/`: evidencia del entrenamiento.
- `tests/`: pruebas de integración.
