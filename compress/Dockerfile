# CPU base for local/CI — TRT conversion gracefully skips without GPU
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    xgboost==2.0.3 \
    scikit-learn==1.4.2 \
    onnxmltools==1.12.0 \
    skl2onnx==1.16.0 \
    onnx==1.16.0 \
    onnxruntime==1.18.0 \
    numpy==1.26.4

COPY to_onnx.py to_trt.py ./

# Run both conversion steps
CMD ["bash", "-c", "python to_onnx.py && python to_trt.py"]
