"""
Teste local da API — Rinha de Backend 2026.

Uso:
    python test_local.py                   # usa os payloads de exemplo do repo
    python test_local.py --requests 1000   # stress test
    python test_local.py --url http://localhost:9999  # URL customizada
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
import urllib.error
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:9999"

EXAMPLE_PAYLOADS = [
    {
        "id": "tx-test-001",
        "transaction": {"amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z"},
        "customer": {"avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001"]},
        "merchant": {"id": "MERC-001", "mcc": "5912", "avg_amount": 298.95},
        "terminal": {"is_online": False, "card_present": True, "km_from_home": 13.7090520965},
        "last_transaction": {"timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.8626479774},
    },
    {
        "id": "tx-test-002",
        "transaction": {"amount": 9999.99, "installments": 12, "requested_at": "2026-03-11T03:00:00Z"},
        "customer": {"avg_amount": 100.0, "tx_count_24h": 15, "known_merchants": []},
        "merchant": {"id": "MERC-999", "mcc": "7995", "avg_amount": 5000.0},
        "terminal": {"is_online": True, "card_present": False, "km_from_home": 800.0},
        "last_transaction": {"timestamp": "2026-03-11T02:55:00Z", "km_from_current": 600.0},
    },
    {
        "id": "tx-test-003",
        "transaction": {"amount": 50.0, "installments": 1, "requested_at": "2026-03-11T12:30:00Z"},
        "customer": {"avg_amount": 55.0, "tx_count_24h": 2, "known_merchants": ["MERC-010"]},
        "merchant": {"id": "MERC-010", "mcc": "5411", "avg_amount": 60.0},
        "terminal": {"is_online": False, "card_present": True, "km_from_home": 2.0},
        "last_transaction": None,
    },
]


def check_ready(url: str) -> bool:
    try:
        req = urllib.request.Request(f"{url}/ready")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status < 300
    except Exception as e:
        print(f"  /ready falhou: {e}")
        return False


def send_fraud_score(url: str, payload: dict) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/fraud-score",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as r:
        resp = json.loads(r.read())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return resp, elapsed_ms


def validate_response(resp: dict) -> list[str]:
    errors = []
    if "approved" not in resp:
        errors.append("campo 'approved' ausente")
    elif not isinstance(resp["approved"], bool):
        errors.append(f"'approved' deve ser bool, got {type(resp['approved'])}")
    if "fraud_score" not in resp:
        errors.append("campo 'fraud_score' ausente")
    else:
        fs = resp["fraud_score"]
        if not isinstance(fs, (int, float)):
            errors.append(f"'fraud_score' deve ser número, got {type(fs)}")
        elif fs not in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            errors.append(f"'fraud_score' inválido: {fs} (esperado: 0.0/0.2/0.4/0.6/0.8/1.0)")
    if "approved" in resp and "fraud_score" in resp:
        fs = resp.get("fraud_score", -1)
        approved = resp.get("approved")
        expected_approved = fs < 0.6
        if approved != expected_approved:
            errors.append(f"'approved'={approved} inconsistente com fraud_score={fs}")
    return errors


def stress_test(url: str, payloads: list, n: int, workers: int = 10):
    latencies = []
    errors = 0
    done = 0

    def send_one(i):
        p = payloads[i % len(payloads)]
        try:
            resp, ms = send_fraud_score(url, p)
            errs = validate_response(resp)
            return ms, len(errs) > 0
        except Exception:
            return None, True

    print(f"\n  Enviando {n} requisições ({workers} workers paralelos)...")
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(send_one, i) for i in range(n)]
        for f in as_completed(futs):
            ms, err = f.result()
            done += 1
            if err:
                errors += 1
            elif ms is not None:
                latencies.append(ms)
            if done % 100 == 0:
                print(f"    {done}/{n}...", end="\r", flush=True)
    total_s = time.perf_counter() - t_start

    if not latencies:
        print("  ERRO: nenhuma resposta válida recebida.")
        return

    latencies.sort()
    p50 = statistics.median(latencies)
    p99 = latencies[int(len(latencies) * 0.99)]
    p999 = latencies[int(len(latencies) * 0.999)]
    rps = n / total_s

    # Simulação do score de latência
    if p99 > 2000:
        score_lat = -3000
    else:
        score_lat = 1000 * math.log10(1000 / max(p99, 1))

    print(f"\n  {'='*50}")
    print(f"  Requisições:  {n} ({errors} erros)")
    print(f"  Throughput:   {rps:.0f} req/s")
    print(f"  Latência p50: {p50:.2f} ms")
    print(f"  Latência p99: {p99:.2f} ms")
    print(f"  Latência p99.9:{p999:.2f} ms")
    print(f"  Tempo total:  {total_s:.1f}s")
    print(f"  {'='*50}")
    print(f"  Score latência estimado: {score_lat:+.0f} pts")
    if p99 <= 1:
        print("  >> EXCELENTE: p99 ≤ 1ms → +3000 pts máximos!")
    elif p99 <= 10:
        print("  >> MUITO BOM: p99 ≤ 10ms")
    elif p99 <= 100:
        print("  >> BOM: p99 ≤ 100ms")
    elif p99 <= 2000:
        print("  >> ATENÇÃO: p99 > 100ms, otimize NPROBE ou hardware")
    else:
        print("  >> CRÍTICO: p99 > 2000ms → penalidade -3000!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    print(f"\nRinha de Backend 2026 — Teste Local")
    print(f"URL: {args.url}")
    print("-" * 40)

    # 1. Health check
    print("\n[1] GET /ready ...")
    if not check_ready(args.url):
        print("FALHOU. Verifique se o docker-compose está rodando (porta 9999).")
        sys.exit(1)
    print("  OK")

    # 2. Testes funcionais
    print("\n[2] POST /fraud-score — testes funcionais:")
    all_ok = True
    for i, payload in enumerate(EXAMPLE_PAYLOADS):
        try:
            resp, ms = send_fraud_score(args.url, payload)
            errs = validate_response(resp)
            status = "OK" if not errs else f"ERRO: {'; '.join(errs)}"
            print(f"  tx-{i+1}: {resp} ({ms:.1f}ms) → {status}")
            if errs:
                all_ok = False
        except Exception as e:
            print(f"  tx-{i+1}: EXCEÇÃO → {e}")
            all_ok = False

    if not all_ok:
        print("\nERRO: respostas inválidas detectadas. Verifique o código.")
        sys.exit(1)

    # 3. Stress test
    print(f"\n[3] Stress test ({args.requests} req, {args.workers} workers):")
    stress_test(args.url, EXAMPLE_PAYLOADS, args.requests, args.workers)


if __name__ == "__main__":
    main()
