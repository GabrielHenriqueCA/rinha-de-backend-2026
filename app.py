# granian + uvloop + msgspec + faiss ivf fp16
import gc
import json
import os

import msgspec
import numpy as np
import faiss

D = 14

faiss.omp_set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")

NPROBE = int(os.environ.get("NPROBE", "4"))

index = faiss.read_index("index.faiss", faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
index.nprobe = NPROBE
LABELS = np.load("labels.npy", mmap_mode="r")

with open("normalization.json") as f:
    N = json.load(f)
with open("mcc_risk.json") as f:
    MCC = json.load(f)

INV_AMOUNT = 1.0 / N["max_amount"]
INV_INST   = 1.0 / N["max_installments"]
INV_RATIO  = 1.0 / N["amount_vs_avg_ratio"]
INV_MIN    = 1.0 / N["max_minutes"]
INV_KM     = 1.0 / N["max_km"]
INV_TX     = 1.0 / N["max_tx_count_24h"]
INV_MERCH  = 1.0 / N["max_merchant_avg_amount"]

# k=5, só 6 resultados possíveis — serializa uma vez e reutiliza
RESPONSES = [
    b'{"approved":true,"fraud_score":0.0}',
    b'{"approved":true,"fraud_score":0.2}',
    b'{"approved":true,"fraud_score":0.4}',
    b'{"approved":false,"fraud_score":0.6}',
    b'{"approved":false,"fraud_score":0.8}',
    b'{"approved":false,"fraud_score":1.0}',
]
SAFE = RESPONSES[0]
CT_JSON = [(b"content-type", b"application/json")]
CT_TEXT = [(b"content-type", b"text/plain")]


class Tx(msgspec.Struct):
    amount: float
    installments: int
    requested_at: str


class Cust(msgspec.Struct):
    avg_amount: float
    tx_count_24h: int
    known_merchants: list


class Merch(msgspec.Struct):
    id: str
    mcc: str
    avg_amount: float


class Term(msgspec.Struct):
    is_online: bool
    card_present: bool
    km_from_home: float


class Last(msgspec.Struct):
    timestamp: str
    km_from_current: float


class Payload(msgspec.Struct):
    transaction: Tx
    customer: Cust
    merchant: Merch
    terminal: Term
    last_transaction: Last | None = None
    id: str = ""


DECODER = msgspec.json.Decoder(Payload)

_SAK = (0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4)

# sem await entre vectorize e index.search, então isso é seguro
QBUF = np.empty((1, D), dtype=np.float32)


def _weekday(y: int, m: int, d: int) -> int:
    if m < 3:
        y -= 1
    return ((y + y // 4 - y // 100 + y // 400 + _SAK[m - 1] + d) % 7 - 1) % 7


def _civil_days(y: int, m: int, d: int) -> int:
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _minutes_between(cur: str, prev: str) -> float:
    def abs_min(s: str) -> float:
        days = _civil_days(int(s[0:4]), int(s[5:7]), int(s[8:10]))
        return days * 1440.0 + int(s[11:13]) * 60.0 + int(s[14:16]) + int(s[17:19]) / 60.0
    return abs_min(cur) - abs_min(prev)


def _c(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x


def vectorize(p: Payload, out: np.ndarray) -> None:
    t = p.transaction
    a = t.amount
    out[0] = _c(a * INV_AMOUNT)
    out[1] = _c(t.installments * INV_INST)
    avg = p.customer.avg_amount
    ratio = (a / avg) if avg > 0.0 else (10.0 if a > 0.0 else 0.0)
    out[2] = _c(ratio * INV_RATIO)
    ts = t.requested_at
    out[3] = int(ts[11:13]) / 23.0
    out[4] = _weekday(int(ts[0:4]), int(ts[5:7]), int(ts[8:10])) / 6.0
    lt = p.last_transaction
    if lt is None:
        out[5] = -1.0
        out[6] = -1.0
    else:
        out[5] = _c(_minutes_between(ts, lt.timestamp) * INV_MIN)
        out[6] = _c(lt.km_from_current * INV_KM)
    out[7] = _c(p.terminal.km_from_home * INV_KM)
    out[8] = _c(p.customer.tx_count_24h * INV_TX)
    out[9]  = 1.0 if p.terminal.is_online else 0.0
    out[10] = 1.0 if p.terminal.card_present else 0.0
    out[11] = 0.0 if p.merchant.id in p.customer.known_merchants else 1.0
    out[12] = MCC.get(p.merchant.mcc, 0.5)
    out[13] = _c(p.merchant.avg_amount * INV_MERCH)


def _warmup() -> None:
    dummy = np.zeros((1, D), dtype=np.float32)
    for _ in range(64):
        index.search(dummy, 5)


_warmup()
gc.disable()
print(f"[api] ready nprobe={NPROBE}", flush=True)


async def app(scope, receive, send):
    if scope["type"] != "http":
        return

    if scope["path"] == "/ready":
        await send({"type": "http.response.start", "status": 200, "headers": CT_TEXT})
        await send({"type": "http.response.body", "body": b"OK"})
        return

    if scope["path"] == "/fraud-score":
        body = b""
        while True:
            event = await receive()
            body += event.get("body", b"")
            if not event.get("more_body", False):
                break
        try:
            p = DECODER.decode(body)
            vectorize(p, QBUF[0])
            _, idx = index.search(QBUF, 5)
            frauds = int(LABELS[idx[0]].sum())
            await send({"type": "http.response.start", "status": 200, "headers": CT_JSON})
            await send({"type": "http.response.body", "body": RESPONSES[frauds]})
        except Exception:
            await send({"type": "http.response.start", "status": 200, "headers": CT_JSON})
            await send({"type": "http.response.body", "body": SAFE})
        return

    await send({"type": "http.response.start", "status": 404, "headers": []})
    await send({"type": "http.response.body", "body": b""})
