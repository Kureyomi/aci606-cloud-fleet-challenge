from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import requests

SAMPLE: dict[str, Any] = {
    "robot_id": "TEST-SMOKE-01",
    "timestamp": time.time(),
    "x": 1.2,
    "y": 3.4,
    "speed": 0.72,
    "battery": 81.0,
    "motor_temp": 49.0,
    "vibration": 0.42,
    "motor_current": 3.8,
    "obstacle_distance": 1.3,
}


def check_status(name: str, response: requests.Response, expected: int = 200) -> bool:
    ok = response.status_code == expected
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: HTTP {response.status_code} (esperado {expected})")
    if not ok:
        print(f"       Respuesta: {response.text[:500]}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pruebas básicas de contrato para Cloud Fleet Challenge."
    )
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="URL base de la API")
    parser.add_argument(
        "--expect-predict",
        action="store_true",
        help="Exige que POST /predict esté disponible y devuelva el contrato completo.",
    )
    parser.add_argument(
        "--check-invalid",
        action="store_true",
        help="Comprueba que un dato inválido sea rechazado con HTTP 4xx.",
    )
    parser.add_argument(
        "--check-history",
        action="store_true",
        help="Comprueba que el registro enviado aparezca en el historial del robot.",
    )
    args = parser.parse_args()
    base = args.api.rstrip("/")
    session = requests.Session()
    failures = 0

    print(f"Cloud Fleet smoke test -> {base}")

    # 1. Estado del servicio
    try:
        response = session.get(f"{base}/health", timeout=10)
        if not check_status("GET /health", response):
            failures += 1
        else:
            try:
                health = response.json()
                print(
                    "       service={service}, database={database}, model_loaded={model}".format(
                        service=health.get("service", "no informado"),
                        database=health.get("database", "no informado"),
                        model=health.get("model_loaded", "no informado"),
                    )
                )
            except ValueError:
                print("       Aviso: /health no devolvió JSON.")
                failures += 1
    except requests.RequestException as exc:
        print(f"[FAIL] GET /health: {exc}")
        return 1

    # 2. Ingesta de telemetría
    try:
        response = session.post(f"{base}/telemetry", json=SAMPLE, timeout=10)
        if not check_status("POST /telemetry", response):
            failures += 1
    except requests.RequestException as exc:
        print(f"[FAIL] POST /telemetry: {exc}")
        failures += 1

    # 3. Lista de robots
    try:
        response = session.get(f"{base}/robots", timeout=10)
        if not check_status("GET /robots", response):
            failures += 1
        else:
            try:
                robots = response.json()
                if not isinstance(robots, list):
                    print("[FAIL] GET /robots: se esperaba una lista JSON.")
                    failures += 1
                else:
                    print(f"       Robots devueltos: {len(robots)}")
            except ValueError:
                print("[FAIL] GET /robots: respuesta JSON inválida.")
                failures += 1
    except requests.RequestException as exc:
        print(f"[FAIL] GET /robots: {exc}")
        failures += 1

    # 4. Historial: verifica que la muestra enviada esté persistida.
    if args.check_history:
        try:
            response = session.get(
                f"{base}/robots/{SAMPLE['robot_id']}/history", timeout=10
            )
            if not check_status("GET /robots/{id}/history", response):
                failures += 1
            else:
                try:
                    history = response.json()
                    found = isinstance(history, list) and any(
                        row.get("robot_id") == SAMPLE["robot_id"]
                        and row.get("timestamp") == SAMPLE["timestamp"]
                        for row in history
                    )
                    if found:
                        print(f"[OK] Historial contiene la muestra enviada ({len(history)} registros).")
                    else:
                        print("[FAIL] La muestra enviada no aparece en el historial.")
                        failures += 1
                except (ValueError, AttributeError):
                    print("[FAIL] El historial no tiene el formato JSON esperado.")
                    failures += 1
        except requests.RequestException as exc:
            print(f"[FAIL] GET /robots/{SAMPLE['robot_id']}/history: {exc}")
            failures += 1

    # 5. Predicción: comprueba el contrato de respuesta definido por el profesor.
    if args.expect_predict:
        try:
            response = session.post(f"{base}/predict", json=SAMPLE, timeout=15)
            if not check_status("POST /predict", response):
                failures += 1
            else:
                try:
                    prediction = response.json()
                    required = {
                        "robot_id",
                        "predicted_status",
                        "confidence",
                        "model_version",
                        "inference_ms",
                    }
                    missing = required.difference(prediction.keys())
                    valid_status = prediction.get("predicted_status") in {
                        "NORMAL",
                        "WARNING",
                        "CRITICAL",
                    }
                    if missing or not valid_status:
                        print(
                            "[FAIL] Contrato de /predict incompleto o estado inválido. "
                            f"Campos faltantes: {sorted(missing)}"
                        )
                        failures += 1
                    else:
                        print(
                            "[OK] Predicción: {status} | confianza={confidence} | "
                            "modelo={version} | latencia={latency} ms".format(
                                status=prediction["predicted_status"],
                                confidence=prediction["confidence"],
                                version=prediction["model_version"],
                                latency=prediction["inference_ms"],
                            )
                        )
                except (ValueError, AttributeError):
                    print("[FAIL] /predict no devolvió el JSON esperado.")
                    failures += 1
        except requests.RequestException as exc:
            print(f"[FAIL] POST /predict: {exc}")
            failures += 1

    # 6. Robustez: un campo batería fuera de rango debe producir un 4xx.
    if args.check_invalid:
        invalid_sample = dict(SAMPLE)
        invalid_sample["robot_id"] = "TEST-INVALID"
        invalid_sample["battery"] = 150
        try:
            response = session.post(
                f"{base}/telemetry", json=invalid_sample, timeout=10
            )
            if 400 <= response.status_code < 500:
                check_status("POST /telemetry con batería inválida", response, response.status_code)
            else:
                print(
                    "[FAIL] POST /telemetry con batería inválida: "
                    f"se esperaba HTTP 4xx y se recibió {response.status_code}."
                )
                failures += 1
        except requests.RequestException as exc:
            print(f"[FAIL] Prueba de dato inválido: {exc}")
            failures += 1

    # 7. Métricas operacionales
    try:
        response = session.get(f"{base}/metrics", timeout=10)
        if not check_status("GET /metrics", response):
            failures += 1
        else:
            try:
                metrics = response.json()
                print(
                    "       solicitudes={requests}, errores 4xx={errors}, "
                    "uptime={uptime}s".format(
                        requests=metrics.get("http_requests", "no informado"),
                        errors=metrics.get("errors_4xx", "no informado"),
                        uptime=metrics.get("uptime_seconds", "no informado"),
                    )
                )
            except ValueError:
                print("[FAIL] /metrics no devolvió JSON.")
                failures += 1
    except requests.RequestException as exc:
        print(f"[FAIL] GET /metrics: {exc}")
        failures += 1

    print("-" * 60)
    if failures:
        print(f"RESULTADO: {failures} prueba(s) fallaron.")
        return 1
    print("RESULTADO: todas las pruebas solicitadas pasaron.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
