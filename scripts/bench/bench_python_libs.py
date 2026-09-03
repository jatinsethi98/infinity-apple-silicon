"""Same-run comparison of Python-installable ANN libraries on SIFT1M (HNSW M=32, efC=200, efS=256, k=10).
Usage: uv run --python 3.12 --with numpy,hnswlib python scripts/bench/bench_python_libs.py hnswlib
       uv run --python 3.12 --with numpy,usearch  python scripts/bench/bench_python_libs.py usearch
       uv run --python 3.12 --with numpy,faiss-cpu python scripts/bench/bench_python_libs.py faiss
Datasets root: $DATASETS or <repo>/datasets (see fetch_datasets.py). Python wrapper overhead is included in QPS."""
import numpy as np, time, os, sys
import os; ROOT = os.environ.get("DATASETS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")); base = np.fromfile(os.path.join(ROOT, "sift1m/base.f32"), dtype=np.float32).reshape(-1, 128)
q = np.fromfile(os.path.join(ROOT, "sift1m/query.f32"), dtype=np.float32).reshape(-1, 128)
gt = np.fromfile(os.path.join(ROOT, "sift1m/groundtruth.i32"), dtype=np.int32).reshape(-1, 100)[:, :10]
n, d = base.shape; T = 10; M = 32; EFC = 200; EFS = 256; K = 10
def recall(ids): return float((ids[:, :K] == gt[:, None, :]).any(-1).mean()) if ids.ndim == 2 else float('nan')
def recall_sets(ids):
    return float(np.mean([len(set(ids[i][:K]) & set(gt[i])) / K for i in range(len(gt))]))
which = sys.argv[1]
if which == "hnswlib":
    import hnswlib
    idx = hnswlib.Index(space="l2", dim=d)
    t = time.perf_counter(); idx.init_index(max_elements=n, ef_construction=EFC, M=M); idx.set_num_threads(T)
    idx.add_items(base, np.arange(n)); build = time.perf_counter() - t
    idx.set_ef(EFS)
    idx.set_num_threads(1); t = time.perf_counter(); idx.knn_query(q[:1000], k=K); lat1 = (time.perf_counter() - t) / 1000
    idx.set_num_threads(12); t = time.perf_counter(); ids, _ = idx.knn_query(q, k=K); wall = time.perf_counter() - t
    print(f"hnswlib build={build:.2f}s vec/s={n/build:,.0f} qps12={len(q)/wall:,.0f} p50~{lat1*1e6:.0f}us recall@10(ef{EFS})={recall_sets(ids):.4f} version={hnswlib.__version__ if hasattr(hnswlib,'__version__') else '?'}")
elif which == "usearch":
    from usearch.index import Index
    import usearch
    idx = Index(ndim=d, metric="l2sq", dtype="f32", connectivity=M, expansion_add=EFC, expansion_search=EFS)
    t = time.perf_counter(); idx.add(np.arange(n), base, threads=T); build = time.perf_counter() - t
    t = time.perf_counter(); idx.search(q[:1000], K, threads=1); lat1 = (time.perf_counter() - t) / 1000
    t = time.perf_counter(); res = idx.search(q, K, threads=12); wall = time.perf_counter() - t
    ids = res.keys if hasattr(res, "keys") else np.array([r.keys for r in res])
    ids = np.asarray(ids).reshape(len(q), -1)
    print(f"usearch build={build:.2f}s vec/s={n/build:,.0f} qps12={len(q)/wall:,.0f} p50~{lat1*1e6:.0f}us recall@10(ef{EFS})={recall_sets(ids):.4f} version={usearch.__version__} hw={idx.hardware_acceleration}")
elif which == "faiss":
    import faiss
    faiss.omp_set_num_threads(T)
    idx = faiss.IndexHNSWFlat(d, M); idx.hnsw.efConstruction = EFC
    t = time.perf_counter(); idx.add(base); build = time.perf_counter() - t
    idx.hnsw.efSearch = EFS
    faiss.omp_set_num_threads(1); t = time.perf_counter(); idx.search(q[:1000], K); lat1 = (time.perf_counter() - t) / 1000
    faiss.omp_set_num_threads(12); t = time.perf_counter(); _, ids = idx.search(q, K); wall = time.perf_counter() - t
    print(f"faiss(pip) build={build:.2f}s vec/s={n/build:,.0f} qps12={len(q)/wall:,.0f} p50~{lat1*1e6:.0f}us recall@10(ef{EFS})={recall_sets(ids):.4f} version={faiss.__version__}")
