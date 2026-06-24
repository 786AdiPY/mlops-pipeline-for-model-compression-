"""Benchmark latency and accuracy for pkl / ONNX FP32 / TRT INT8."""
import os
import json
import pickle
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score, roc_auc_score

MODEL_PKL   = os.getenv("MODEL_PKL",   "/artifacts/model.pkl")
MODEL_ONNX  = os.getenv("MODEL_ONNX",  "/artifacts/model_fp32.onnx")
MODEL_TRT   = os.getenv("MODEL_TRT",   "/artifacts/model_int8.trt")
TEST_CSV    = os.getenv("TEST_CSV",    "/data/test.csv")
RESULTS_OUT = os.getenv("RESULTS_OUT", "/artifacts/benchmark_results.json")
WARMUP      = int(os.getenv("WARMUP",  "20"))
RUNS        = int(os.getenv("RUNS",    "200"))

FEATURE_COLS = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_products", "support_calls", "payment_delay_days",
    "contract_type", "internet_service", "online_security", "tech_support",
]


def load_test():
    df = pd.read_csv(TEST_CSV)
    X  = df[FEATURE_COLS].values.astype(np.float32)
    y  = df["churn"].values
    return X, y


def bench_pkl(X: np.ndarray, y: np.ndarray):
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    # Warmup
    for _ in range(WARMUP):
        model.predict_proba(X[:1])

    t0 = time.perf_counter()
    for _ in range(RUNS):
        proba = model.predict_proba(X)[:, 1]
    elapsed = (time.perf_counter() - t0) / RUNS * 1000  # ms per batch

    pred = (proba > 0.5).astype(int)
    return {
        "model":        "pkl_xgboost",
        "latency_ms":   round(elapsed, 3),
        "accuracy":     round(accuracy_score(y, pred), 4),
        "auc":          round(roc_auc_score(y, proba), 4),
        "model_size_kb": round(os.path.getsize(MODEL_PKL) / 1024, 1),
    }


def bench_onnx(X: np.ndarray, y: np.ndarray):
    sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0].name

    for _ in range(WARMUP):
        sess.run(None, {inp: X[:1]})

    t0 = time.perf_counter()
    for _ in range(RUNS):
        out = sess.run(None, {inp: X})
    elapsed = (time.perf_counter() - t0) / RUNS * 1000

    if isinstance(out[1], list):
        proba = np.array([d[1] for d in out[1]])
    elif hasattr(out[1], 'ndim') and out[1].ndim == 2:
        proba = out[1][:, 1]
    else:
        proba = out[1]
    pred  = (proba > 0.5).astype(int)
    return {
        "model":        "onnx_fp32",
        "latency_ms":   round(elapsed, 3),
        "accuracy":     round(accuracy_score(y, pred), 4),
        "auc":          round(roc_auc_score(y, proba), 4),
        "model_size_kb": round(os.path.getsize(MODEL_ONNX) / 1024, 1),
    }


def bench_trt(X: np.ndarray, y: np.ndarray):
    fallback_marker = MODEL_TRT + ".json"
    if os.path.exists(fallback_marker):
        with open(fallback_marker) as f:
            marker = json.load(f)
        if marker.get("fallback"):
            print("TRT engine not built (no GPU). Running ONNX fallback for v3.")
            result = bench_onnx(X, y)
            result["model"] = "trt_int8_fallback_onnx"
            return result

    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa

        with open(MODEL_TRT, "rb") as f:
            engine_bytes = f.read()

        runtime  = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine   = runtime.deserialize_cuda_engine(engine_bytes)
        context  = engine.create_execution_context()

        d_input  = cuda.mem_alloc(X.nbytes)
        out_shape = (len(X),)
        output   = np.empty(out_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)
        stream   = cuda.Stream()

        def infer():
            cuda.memcpy_htod_async(d_input, X, stream)
            context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
            cuda.memcpy_dtoh_async(output, d_output, stream)
            stream.synchronize()

        for _ in range(WARMUP):
            infer()

        t0 = time.perf_counter()
        for _ in range(RUNS):
            infer()
        elapsed = (time.perf_counter() - t0) / RUNS * 1000

        proba = output.copy()
        pred  = (proba > 0.5).astype(int)
        return {
            "model":        "trt_int8",
            "latency_ms":   round(elapsed, 3),
            "accuracy":     round(accuracy_score(y, pred), 4),
            "auc":          round(roc_auc_score(y, proba), 4),
            "model_size_kb": round(os.path.getsize(MODEL_TRT) / 1024, 1),
        }

    except Exception as e:
        print(f"TRT bench failed: {e}. Falling back to ONNX.")
        result = bench_onnx(X, y)
        result["model"] = "trt_int8_fallback_onnx"
        return result


def main():
    X, y = load_test()
    print(f"Test set: {len(X)} rows, {X.shape[1]} features")

    results = []
    for name, fn in [("pkl", bench_pkl), ("onnx", bench_onnx), ("trt", bench_trt)]:
        print(f"\n--- Benchmarking {name} ---")
        r = fn(X, y)
        results.append(r)
        print(json.dumps(r, indent=2))

    # Compute relative speedup vs pkl baseline
    baseline_lat = results[0]["latency_ms"]
    for r in results:
        r["speedup_vs_pkl"] = round(baseline_lat / r["latency_ms"], 2)

    os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {RESULTS_OUT}")

    # Summary table
    print("\n{:<30} {:>12} {:>10} {:>8} {:>12}".format(
        "model", "latency_ms", "accuracy", "auc", "speedup"))
    print("-" * 76)
    for r in results:
        print("{:<30} {:>12.3f} {:>10.4f} {:>8.4f} {:>12.2f}x".format(
            r["model"], r["latency_ms"], r["accuracy"], r["auc"], r["speedup_vs_pkl"]))


if __name__ == "__main__":
    main()
