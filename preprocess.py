# roda uma vez no build, gera index.faiss + labels.npy que ficam na imagem
import gzip
import time

import msgspec
import numpy as np
import faiss

REFERENCES = "resources/references.json.gz"
OUT_INDEX = "index.faiss"
OUT_LABELS = "labels.npy"

D = 14
NLIST = 4096
TRAIN_SAMPLE = 256 * NLIST  # ~1M, recomendado pelo paper do FAISS


class Ref(msgspec.Struct):
    vector: list
    label: str


def main() -> None:
    t0 = time.time()
    print(f"[preprocess] lendo {REFERENCES} ...", flush=True)
    with open(REFERENCES, "rb") as f:
        raw = gzip.decompress(f.read())
    refs = msgspec.json.decode(raw, type=list[Ref])
    del raw
    n = len(refs)
    print(f"[preprocess] {n:,} vetores em {time.time()-t0:.1f}s", flush=True)

    X = np.empty((n, D), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    for i, r in enumerate(refs):
        X[i] = r.vector
        y[i] = 1 if r.label == "fraud" else 0
    del refs
    print(f"[preprocess] matriz montada ({X.nbytes/1e6:.0f} MB)", flush=True)

    quantizer = faiss.IndexFlatL2(D)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, D, NLIST, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2
    )

    rng = np.random.default_rng(42)
    sample = X if n <= TRAIN_SAMPLE else X[rng.choice(n, TRAIN_SAMPLE, replace=False)]
    print(f"[preprocess] treinando IVF (nlist={NLIST}, train={len(sample):,}) ...", flush=True)
    index.train(sample)
    del sample

    print("[preprocess] adicionando vetores ...", flush=True)
    index.add(X)
    del X

    faiss.write_index(index, OUT_INDEX)
    np.save(OUT_LABELS, y)
    print(f"[preprocess] OK em {time.time()-t0:.1f}s -> {OUT_INDEX} + {OUT_LABELS}", flush=True)


if __name__ == "__main__":
    main()
