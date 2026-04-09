# License & attribution

## This project

- **Copyright © 2026 Levi Pereira** ([levi.pereira@gmail.com](mailto:levi.pereira@gmail.com))
- Licensed under the **Apache License, Version 2.0**
- Full text: [`LICENSE`](../LICENSE)
- Attribution file: [`NOTICE`](../NOTICE)

---

## Third-party components

This toolkit is designed for use with:

| Project | License | Notes |
|---------|---------|--------|
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) | Apache 2.0 | PTQ, optional ONNX autotune; install source may be PyPI and/or Git |
| [TensorRT](https://developer.nvidia.com/tensorrt) | Proprietary (SDK) / see NVIDIA | Engine build and inference in NGC container |
| [ONNX Runtime](https://onnxruntime.ai/) | MIT | Calibration and EP testing |
| [COCO dataset](https://cocodataset.org/) | [CC BY 4.0](https://cocodataset.org/#termsofuse) | Optional calibration / evaluation images and annotations |

Upstream examples that influenced workflow: [Model Optimizer `examples/onnx_ptq`](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq).

---

## Trademarks

NVIDIA, TensorRT, CUDA, and related marks are trademarks of NVIDIA Corporation. This project is not affiliated with or endorsed by NVIDIA beyond use of public tools and documentation as intended.
