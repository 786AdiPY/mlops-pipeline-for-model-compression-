"""FastAPI serving: /predict/v1 (pkl), /v2 (onnx), /v3 (trt/onnx-fallback)."""
import os
import json
import pickle
import time
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

MODEL_PKL  = os.getenv("MODEL_PKL",  "/artifacts/model.pkl")
MODEL_ONNX = os.getenv("MODEL_ONNX", "/artifacts/model_fp32.onnx")
MODEL_TRT  = os.getenv("MODEL_TRT",  "/artifacts/model_int8.trt")

FEATURE_COLS = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_products", "support_calls", "payment_delay_days",
    "contract_type", "internet_service", "online_security", "tech_support",
]

models: dict = {}


class ChurnRequest(BaseModel):
    tenure_months:      float
    monthly_charges:    float
    total_charges:      float
    num_products:       int
    support_calls:      int
    payment_delay_days: int
    contract_type:      int
    internet_service:   int
    online_security:    int
    tech_support:       int

    @field_validator("contract_type")
    @classmethod
    def validate_contract(cls, v):
        if v not in (0, 1, 2):
            raise ValueError("contract_type must be 0, 1, or 2")
        return v


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction:  int
    model_version:     str
    latency_ms:        float


class CompareResponse(BaseModel):
    input:   dict
    results: List[PredictResponse]


def _to_array(req: ChurnRequest) -> np.ndarray:
    return np.array(
        [[getattr(req, col) for col in FEATURE_COLS]], dtype=np.float32
    )


def _load_trt_or_fallback():
    fallback = MODEL_TRT + ".json"
    if os.path.exists(fallback):
        with open(fallback) as f:
            marker = json.load(f)
        if marker.get("fallback"):
            return ("onnx_fallback", ort.InferenceSession(
                marker["use_onnx"], providers=["CPUExecutionProvider"]
            ))

    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa
        with open(MODEL_TRT, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine  = runtime.deserialize_cuda_engine(engine_bytes)
        return ("trt", engine)
    except Exception:
        return ("onnx_fallback", ort.InferenceSession(
            MODEL_ONNX, providers=["CPUExecutionProvider"]
        ))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # v1 — pkl
    with open(MODEL_PKL, "rb") as f:
        models["v1"] = ("pkl", pickle.load(f))

    # v2 — ONNX FP32
    models["v2"] = ("onnx", ort.InferenceSession(
        MODEL_ONNX, providers=["CPUExecutionProvider"]
    ))

    # v3 — TRT or ONNX fallback
    models["v3"] = _load_trt_or_fallback()

    print("Models loaded: v1=pkl, v2=onnx, v3=" + models["v3"][0])
    yield
    models.clear()


app = FastAPI(title="XGB Churn API", version="1.0.0", lifespan=lifespan)


def _infer_pkl(model, X: np.ndarray):
    proba = model.predict_proba(X)[0, 1]
    return float(proba)


def _infer_onnx(sess: ort.InferenceSession, X: np.ndarray):
    inp  = sess.get_inputs()[0].name
    out  = sess.run(None, {inp: X})
    if isinstance(out[1], list):
        proba = out[1][0][1]
    elif hasattr(out[1], 'ndim') and out[1].ndim == 2:
        proba = out[1][0, 1]
    else:
        proba = float(out[0][0])
    return float(proba)


def _infer_trt(engine, X: np.ndarray):
    import pycuda.driver as cuda
    context = engine.create_execution_context()
    output  = np.empty((1,), dtype=np.float32)
    d_in    = cuda.mem_alloc(X.nbytes)
    d_out   = cuda.mem_alloc(output.nbytes)
    stream  = cuda.Stream()
    cuda.memcpy_htod_async(d_in, X, stream)
    context.execute_async_v2([int(d_in), int(d_out)], stream.handle)
    cuda.memcpy_dtoh_async(output, d_out, stream)
    stream.synchronize()
    return float(output[0])


def predict_version(version: str, req: ChurnRequest) -> PredictResponse:
    if version not in models:
        raise HTTPException(404, f"Version {version} not found")

    kind, model = models[version]
    X           = _to_array(req)

    t0 = time.perf_counter()
    if kind == "pkl":
        proba = _infer_pkl(model, X)
    elif kind in ("onnx", "onnx_fallback"):
        proba = _infer_onnx(model, X)
    else:
        proba = _infer_trt(model, X)
    latency = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        churn_probability=round(proba, 6),
        churn_prediction=int(proba > 0.5),
        model_version=f"{version}/{kind}",
        latency_ms=round(latency, 3),
    )


@app.post("/predict/v1", response_model=PredictResponse)
def predict_v1(req: ChurnRequest):
    return predict_version("v1", req)


@app.post("/predict/v2", response_model=PredictResponse)
def predict_v2(req: ChurnRequest):
    return predict_version("v2", req)


@app.post("/predict/v3", response_model=PredictResponse)
def predict_v3(req: ChurnRequest):
    return predict_version("v3", req)


@app.post("/compare", response_model=CompareResponse)
def compare(req: ChurnRequest):
    results = [predict_version(v, req) for v in ("v1", "v2", "v3")]
    return CompareResponse(input=req.model_dump(), results=results)


@app.get("/health")
def health():
    return {"status": "ok", "loaded_versions": list(models.keys())}
