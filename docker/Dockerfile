# NVIDIA Model Optimizer ONNX PTQ — TensorRT container (see upstream onnx_ptq example)
# https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq
FROM nvcr.io/nvidia/tensorrt:26.02-py3

# Required inside TensorRT images for cuDNN discovery (per Model Optimizer docs)
ENV CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}"

# TREx: isolated venv (pandas/trex pins) — avoids clashing with model-opt-yolo / cupy / numpy stacks.
ENV TREX_HOME=/workspace/TREx
ENV TREX_VENV=/workspace/TREx/tools/experimental/trt-engine-explorer/env_trex

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
# graphviz: TREx (install.sh also runs apt install graphviz). sudo: required by upstream install.sh.
RUN apt-get update && apt-get install -y --no-install-recommends git graphviz sudo \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -U \
        "nvidia-modelopt[onnx] @ git+https://github.com/NVIDIA/Model-Optimizer.git@${MODELOPT_GIT_REF}" \
    && pip install --no-cache-dir /workspace/model-opt-yolo \
    && pip install --no-cache-dir --upgrade --force-reinstall --pre \
        --index-url "${ORT_CUDA13_INDEX}" \
        onnxruntime-gpu --no-deps \
    && rm -rf /workspace/model-opt-yolo/*

# TREx: install.sh --venv --full in env_trex. model-opt-yolo stays on the image Python; trex-analyze
# prepends TREX_VENV site-packages so ``import trex`` works without installing the CLI into env_trex.
ARG TENSORRT_TREX_GIT_REF=release/10.15
RUN git clone --depth 1 --branch "${TENSORRT_TREX_GIT_REF}" \
        https://github.com/NVIDIA/TensorRT.git "${TREX_HOME}" \
    && touch "${TREX_HOME}/tools/experimental/trt-engine-explorer/bin/__init__.py" \
    && sed -i 's|sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "utils"))|sys.path.insert(0, os.path.dirname(SCRIPT_DIR))|' \
        "${TREX_HOME}/tools/experimental/trt-engine-explorer/bin/trex.py" \
    && bash -c "cd \"${TREX_HOME}/tools/experimental/trt-engine-explorer\" && source ./install.sh --venv --full" \
    && "${TREX_VENV}/bin/python" -c "import trex; print('trex', trex.__version__)"

# Default: shell in project dir; bind-mount host models/data/artifacts under this path
CMD ["/bin/bash"]
