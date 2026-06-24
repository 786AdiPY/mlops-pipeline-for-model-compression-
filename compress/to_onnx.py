"""Convert XGBoost pkl → ONNX FP32 using onnxmltools."""
import os
import pickle
import json
import numpy as np
import onnx
import onnxruntime as ort
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

MODEL_PKL  = os.getenv("MODEL_PKL",  "/artifacts/model.pkl")
MODEL_ONNX = os.getenv("MODEL_ONNX", "/artifacts/model_fp32.onnx")
META_PATH  = os.getenv("META_PATH",  "/artifacts/model_meta.json")

FEATURE_COLS = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_products", "support_calls", "payment_delay_days",
    "contract_type", "internet_service", "online_security", "tech_support",
]


def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {"feature_cols": FEATURE_COLS}


def convert(pkl_path: str, onnx_path: str):
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    meta       = load_meta()
    n_features = len(meta["feature_cols"])

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model   = convert_xgboost(model, initial_types=initial_type)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX saved → {onnx_path}  ({os.path.getsize(onnx_path)/1024:.1f} KB)")
    return onnx_path


def validate(pkl_path: str, onnx_path: str, n_samples: int = 100):
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    rng = np.random.default_rng(0)
    X   = rng.uniform(0, 1, (n_samples, len(FEATURE_COLS))).astype(np.float32)
    ref = model.predict_proba(X)[:, 1]

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out  = sess.run(None, {"float_input": X})
    # onnxmltools XGBoost output: [labels, probabilities_map]
    # probabilities_map is list of {0: p0, 1: p1} dicts
    if isinstance(out[1], list):
        pred = np.array([d[1] for d in out[1]])
    elif hasattr(out[1], 'ndim') and out[1].ndim == 2:
        pred = out[1][:, 1]
    else:
        pred = out[1]

    max_diff = float(np.abs(ref - pred).max())
    print(f"Max probability diff pkl vs ONNX: {max_diff:.6f}")
    assert max_diff < 1e-3, f"ONNX mismatch too large: {max_diff}"
    print("Validation passed.")


if __name__ == "__main__":
    onnx_path = convert(MODEL_PKL, MODEL_ONNX)
    validate(MODEL_PKL, onnx_path)
