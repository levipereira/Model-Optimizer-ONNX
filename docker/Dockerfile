# NVIDIA Model Optimizer ONNX PTQ — TensorRT container (see upstream onnx_ptq example)
# https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq
FROM nvcr.io/nvidia/tensorrt:26.02-py3

# Required inside TensorRT images for cuDNN discovery (per Model Optimizer docs)
ENV CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}"

# TREx: isolated venv (pandas/trex pins) — avoids clashing with modelopt-onnx-ptq / cupy / numpy stacks.
ENV TREX_HOME=/workspace/TREx
ENV TREX_VENV=/workspace/TREx/tools/experimental/trt-engine-explorer/env_trex

WORKDIR /workspace/modelopt-onnx-ptq

# nvidia-modelopt from NVIDIA PyPI (needs --extra-index-url). `[onnx]` for PTQ / modelopt.onnx.quantization.
# onnxruntime-gpu from PyPI targets CUDA 12; this image ships CUDA 13 — align ORT with the toolkit:
# https://github.com/microsoft/onnxruntime/issues (ort-cuda-13-nightly)
ARG MODELOPT_VERSION=0.43.0
ARG NVIDIA_PYPI_EXTRA_INDEX=https://pypi.nvidia.com
ARG ORT_CUDA13_INDEX="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"

# Install local CLI package (modelopt-onnx-ptq) + upstream Model Optimizer + ORT nightly
COPY pyproject.toml README.md LICENSE NOTICE /workspace/modelopt-onnx-ptq/
COPY modelopt_onnx_ptq /workspace/modelopt-onnx-ptq/modelopt_onnx_ptq/

# Avoid upgrading apt-managed wheel/setuptools on this base (can break pip uninstall).
# graphviz: TREx (install.sh also runs apt install graphviz). sudo: required by upstream install.sh.
RUN apt-get update && apt-get install -y --no-install-recommends git graphviz sudo \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -U \
        "nvidia-modelopt[onnx]==${MODELOPT_VERSION}" \
        --extra-index-url "${NVIDIA_PYPI_EXTRA_INDEX}" \
    && pip install --no-cache-dir /workspace/modelopt-onnx-ptq \
    && pip install --no-cache-dir --upgrade --force-reinstall --pre \
        --index-url "${ORT_CUDA13_INDEX}" \
        onnxruntime-gpu --no-deps \
    && rm -rf /workspace/modelopt-onnx-ptq/*

# TREx: install.sh --venv --full in env_trex. modelopt-onnx-ptq stays on the image Python; trex-analyze
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
