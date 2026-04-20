# Agent Skills (`skills/`)

This folder ships **portable [Agent Skills](https://agentskills.io)** for **any** agentic environment—**Claude Code**, **Cursor**, **GitHub Copilot**, CLI agents, etc.—so workflows do not depend on editor-specific config. The same files can be copied into `.cursor/skills/` or a product’s skills directory locally; **`skills/` in this repo is the version published on GitHub.**

**Human-oriented docs:** [`docs/`](../docs/) — update `README.md`, `docs/*.md`, and CLI help when behavior changes (see umbrella skill).

---

## Agentic workflow (quick)

1. **Repository conventions & where things live** → [`modelopt-onnx-ptq-dev/SKILL.md`](modelopt-onnx-ptq-dev/SKILL.md) (umbrella).
2. **Run PTQ** (calib → quantize → build-trt) → [`onnx-ptq/SKILL.md`](onnx-ptq/SKILL.md) + [`onnx-ptq/reference.md`](onnx-ptq/reference.md).
3. **Benchmark / compare modes / profiles** → [`ptq-trt-performance/SKILL.md`](ptq-trt-performance/SKILL.md).
4. **`eval-trt` I/O, `--output-format auto`, DeepStream vs Ultralytics** → [`onnx-eval-io-autodetect/SKILL.md`](onnx-eval-io-autodetect/SKILL.md).
5. **CUDA, ORT, EP, autotune, TRT parse errors** → [`modelopt-troubleshooting/SKILL.md`](modelopt-troubleshooting/SKILL.md).

After code or flag changes, cross-check [`docs/cli-reference.md`](../docs/cli-reference.md) and `modelopt-onnx-ptq --help`.

---

## Files in this folder

| File | Role |
|------|------|
| [`modelopt-onnx-ptq-dev/SKILL.md`](modelopt-onnx-ptq-dev/SKILL.md) | **Umbrella:** layout, rules, full CLI index, Docker, coding standards, pointers to domain skills. |
| [`onnx-ptq/SKILL.md`](onnx-ptq/SKILL.md) | PTQ steps: env, `download-coco`, `calib`, `quantize`, `pipeline-e2e`, modes, autotune, `build-trt`. |
| [`onnx-ptq/reference.md`](onnx-ptq/reference.md) | `quantize()` signature, upstream `python -m modelopt.onnx.quantization` flags, opset, EPs. |
| [`ptq-trt-performance/SKILL.md`](ptq-trt-performance/SKILL.md) | mAP vs QPS, `pipeline-e2e`, `report-runs`, YAML profiles, backbone/neck/head `include_nodes`, `trex-analyze`. |
| [`onnx-eval-io-autodetect/SKILL.md`](onnx-eval-io-autodetect/SKILL.md) | `eval-trt` output layouts, `--output-format auto`, ONNX heuristics, roadmap for CLI renames. |
| [`modelopt-troubleshooting/SKILL.md`](modelopt-troubleshooting/SKILL.md) | Diagnostics: libcublas, EPs, calibration, autotune Concat, `QuantizeLinear` / trtexec issues. |

---

## `SKILL.md` format

Each skill starts with YAML frontmatter:

```yaml
---
name: onnx-ptq
description: >-
  One-line description for skill discovery (what triggers this skill).
---
```

The **`description`** field should name the CLI (`modelopt-onnx-ptq`), common tasks (PTQ, TensorRT, calibration), and error patterns so tools can attach the right file.

---

## Privacy and safety

Do not commit secrets, internal hostnames, or customer paths. Examples use generic names (`your_model.onnx`, `artifacts/…`).

---

## Documentation hygiene

When **behavior**, **CLI flags**, **defaults**, or **paths** change, update the matching **skill**, **`docs/*.md`**, and **`modelopt_onnx_ptq/`** help strings in the same change set.
