# Rinha de Backend 2026 — Fraud Detection (Python)

Fraud-score API under 1 CPU / 350 MB. Challenge: k-NN (k=5, Euclidean distance) over **3,000,000** 14-dimensional reference vectors.

## Stack

| Layer | Choice | Why |
|---|---|---|
| HTTP | **granian + uvloop** | Rust-backed ASGI server with native event loop |
| JSON | **msgspec** | Decodes directly into `Struct`; faster than orjson, free validation |
| Search | **FAISS IVF + fp16** | ANN O(√N); fp16 preserves distance precision better than SQ8 |
| Output | **pre-computed responses** | Only 6 possible outputs (k=5); zero serialization per request |
| LB | **HAProxy + UDS** | TCP mode, Unix Domain Sockets — zero TCP loopback overhead |

Extra optimizations: **mmap read-only** index, **GC disabled** after warmup, **1 FAISS thread** per instance, **warmup** before `/ready`, reused query buffer, weekday via Sakamoto (no `datetime`).

## How to run

1. Download the official dataset and place it at `resources/references.json.gz` (3M vectors).
2. Build — the FAISS index is built at image build time, not at startup:
   ```bash
   docker compose build    # on Mac ARM: DOCKER_DEFAULT_PLATFORM=linux/amd64
   docker compose up
   ```
3. Test:
   ```bash
   curl localhost:9999/ready
   python test_local.py --requests 500 --workers 20
   ```

## Tuning: NPROBE

Detection score is nearly free (we replicate the reference algorithm exactly), so the competition is **latency**. `NPROBE` controls recall vs. speed:

- Higher `NPROBE` (e.g. 16) → better recall, higher latency.
- Lower `NPROBE` (e.g. 4) → faster, slightly lower recall.

Adjust via `NPROBE` env var in `docker-compose.yml` — no rebuild needed. To change index granularity, edit `NLIST` in `preprocess.py` (rebuild required).

## Scoring formula

```
score_final = score_p99 + score_det   (max +6000, min −6000)

score_p99:  1000 × log₁₀(1000 / p99)   if p99 ≤ 2000ms, else −3000
score_det:  1000 × log₁₀(1/ε) − 300 × log₁₀(1+E)
            where E = 1×FP + 3×FN + 5×HTTP_errors
```
