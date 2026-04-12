# NVIDIA Model Optimizer ONNX PTQ — TensorRT container (see upstream onnx_ptq example)
# https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq
FROM nvcr.io/nvidia/tensorrt:26.02-py3

# Required inside TensorRT images for cuDNN discovery (per Model Optimizer docs)
ENV CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}"

# TREx (TensorRT Engine Explorer): NVIDIA/TensorRT sources + dedicated venv for TensorRT engine
# profiling / exploration (layer graphs, trtexec JSON, notebooks). Optional; not used by the
# model-opt-yolo PTQ CLI. Sources track branch release/10.15; see docs/docker-reference.md.
ENV TREX_HOME=/workspace/TREx
ENV TREX_VENV=/workspace/TREx/env

WORKDIR /workspace/model-opt-yolo

# PyPI nvidia-modelopt does not ship the full ONNX autotune CLI; install from GitHub (PEP 508 direct URL).
# onnxruntime-gpu from PyPI targets CUDA 12; this image ships CUDA 13 — align ORT with the toolkit:
# https://github.com/microsoft/onnxruntime/issues (ort-cuda-13-nightly)
ARG MODELOPT_GIT_REF=main
ARG ORT_CUDA13_INDEX="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"

# Install local CLI package (model-opt-yolo) + upstream Model Optimizer + ORT nightly
COPY pyproject.toml README.md LICENSE NOTICE /workspace/model-opt-yolo/
COPY model_opt_yolo /workspace/model-opt-yolo/model_opt_yolo/

# Avoid upgrading apt-managed wheel/setuptools on this base (can break pip uninstall).
# graphviz: system package for TREx (trt-engine-explorer) pydot/graphviz backends.
# python3-venv: ensures `python3 -m venv` works for /workspace/TREx/env across base image variants.
RUN apt-get update && apt-get install -y --no-install-recommends git graphviz python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -U \
        "nvidia-modelopt[onnx] @ git+https://github.com/NVIDIA/Model-Optimizer.git@${MODELOPT_GIT_REF}" \
    && pip install --no-cache-dir /workspace/model-opt-yolo \
    && pip install --no-cache-dir --upgrade --force-reinstall --pre \
        --index-url "${ORT_CUDA13_INDEX}" \
        onnxruntime-gpu --no-deps \
    && rm -rf /workspace/model-opt-yolo/*

# TREx: shallow clone TensorRT matching release/10.15; editable install of tools/experimental/trt-engine-explorer
# into ${TREX_VENV} with notebook extras (Jupyter workflows for engine analysis).
ARG TENSORRT_TREX_GIT_REF=release/10.15
RUN git clone --depth 1 --branch "${TENSORRT_TREX_GIT_REF}" \
        https://github.com/NVIDIA/TensorRT.git "${TREX_HOME}" \
    && python3 -m venv "${TREX_VENV}" \
    && "${TREX_VENV}/bin/pip" install --no-cache-dir -U pip setuptools wheel \
    && "${TREX_VENV}/bin/pip" install --no-cache-dir \
        -e "${TREX_HOME}/tools/experimental/trt-engine-explorer[notebook]"

# Default: shell in project dir; bind-mount host models/data/artifacts under this path
CMD ["/bin/bash"]
