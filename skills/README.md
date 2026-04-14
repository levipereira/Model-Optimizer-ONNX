# AI coding agents & Agent Skills

This folder ships **structured context** for **AI coding agents**: assistants that edit code, run commands, and follow project conventions inside your editor or in agent-style CLIs. The format follows the **[Agent Skills](https://agentskills.io)** open format—portable `SKILL.md` files with frontmatter and focused workflows—so agents do not rely only on generic model knowledge.

**Also read:** **[`docs/`](../docs/)** — human-oriented documentation; agents should update these when behavior or flags change (see the umbrella skill).

---

## What each file is for

### Umbrella: [`model-opt-yolo-dev/SKILL.md`](model-opt-yolo-dev/SKILL.md)

Single entry point for **repository conventions**: layout, pipeline order, code style, docs policy, ONNX quantization notes, Docker/CUDA hints, calibration expectations, and performance-measurement steps. It links out to the domain skills below instead of duplicating every command.

Use it when the agent should **change code**, **add CLI flags**, or **align with maintainer rules**.

### Domain: [`onnx-ptq/SKILL.md`](onnx-ptq/SKILL.md)

Step-by-step **PTQ**: environment (Docker/local), `calib`, `model-opt-yolo quantize`, `pipeline-e2e`, mode/method tables (int8/fp8/int4), autotune presets, validation snippet, `build-trt`.

### Reference: [`onnx-ptq/reference.md`](onnx-ptq/reference.md)

**API-level** detail: full `modelopt.onnx.quantization.quantize()` signature, `python -m modelopt.onnx.quantization` CLI flags, opset requirements, execution-provider resolution, optional AutoCast helpers.

### Domain: [`ptq-trt-performance/SKILL.md`](ptq-trt-performance/SKILL.md)

**Benchmarking and tuning**: comparable mAP vs QPS, `pipeline-e2e` + `--quant-matrix`, `--session-id`, `report-runs` and session-only reports, manual A/B, **backbone / neck / head** Conv enumeration and `include_nodes` YAML whitelists, `trex-analyze`, TensorRT build fallbacks (`strongly-typed` vs `best`).

### Domain: [`modelopt-troubleshooting/SKILL.md`](modelopt-troubleshooting/SKILL.md)

**Diagnostics**: ORT/CUDA/cuDNN/TensorRT mismatches, EP errors, calibration shape issues, autotune Concat failures, `QuantizeLinear` / trtexec parse errors, OOM during calibration.

---

## `SKILL.md` format

Each `SKILL.md` starts with YAML frontmatter, for example:

```yaml
---
name: onnx-ptq
description: >-
  One-line description for skill discovery (what the agent should use this for).
---
```

The **`description`** field helps tools decide when to attach the skill. Body markdown is normal GitHub-flavored text: headings, tables, code fences, links to `docs/`.

---

## How agents should use this

1. **Implement or refactor** → read [`model-opt-yolo-dev/SKILL.md`](model-opt-yolo-dev/SKILL.md) first (conventions + rules).
2. **Run quantize / pipeline-e2e** → [`onnx-ptq/SKILL.md`](onnx-ptq/SKILL.md) + [`reference.md`](onnx-ptq/reference.md) as needed.
3. **Compare PTQ modes or profiles / tune latency** → [`ptq-trt-performance/SKILL.md`](ptq-trt-performance/SKILL.md).
4. **Errors from modelopt, ORT, or TRT** → [`modelopt-troubleshooting/SKILL.md`](modelopt-troubleshooting/SKILL.md).

Always cross-check **CLI flags** in [`docs/cli-reference.md`](../docs/cli-reference.md) after substantive changes.

---

## Privacy and safety

Do not commit **secrets**, **internal hostnames**, or **customer-specific paths** into skills. Examples in skills use generic names (e.g. `your_model.onnx`, profile placeholders). If you fork or mirror this repo publicly, review diffs for accidental environment-specific strings.

---

## Documentation hygiene

When **behavior**, **CLI flags**, or **defaults** change, update this umbrella and/or domain skills **and** user docs (`README.md`, `docs/*.md`, CLI help) so they stay consistent.
