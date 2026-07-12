# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StarVLA is a modular, plug-and-play research platform for Vision-Language-Action (VLA) models for generalist robots. It supports multiple VLA framework variants, VLM backends, action heads, and data pipelines, all composable via config-driven registration.

## Common Commands

### Installation
```bash
conda create -n starVLA python=3.10 -y && conda activate starVLA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

### Linting (scoped to changed files only — full-repo `make check` fails due to historical backlog)
```bash
FILES=$(git diff --name-only --diff-filter=ACMR origin/starVLA_dev | grep -E '\.py$')
black $FILES
python -m ruff check --fix $FILES
```

### Full-repo lint (reference only, expected to fail)
```bash
make check        # black --check + ruff check (read-only)
make autoformat   # black + ruff --fix-only (in-place)
```

### Tests
```bash
python -m pytest tests/
python -m unittest tests/test_single_process_dist_safety.py  # single test
```

### Training (launched via accelerate)
```bash
accelerate launch --config_file <deepspeed_yaml> starVLA/training/train_starvla.py --cfg <experiment_yaml>
```

## Lint / Format Config

- **Black**: line-length 121, target py310, preview mode
- **Ruff**: line-length 121, rules A/B/E/F/I/RUF/W, ignores F722. `__init__.py` ignores E402/F401.
- Configured in `pyproject.toml`.

## Architecture

### Framework Registry Pattern

All model frameworks register via `@FRAMEWORK_REGISTRY.register("Name")` (defined in `starVLA/model/tools.py`). The `build_framework(cfg)` function in `starVLA/model/framework/base_framework.py` auto-imports all framework modules under `starVLA/model/framework/` and instantiates by `cfg.framework.name`.

The base class `baseframework` extends HuggingFace `PreTrainedModel`. Subclasses implement:
- `forward(examples)` → returns dict with `"action_loss"` (training)
- `predict_action(examples)` → returns dict with `"normalized_actions"` (inference)

### VLA Framework Variants (in `starVLA/model/framework/VLM4A/`)

| Framework | Class | Action Head |
|-----------|-------|-------------|
| StarVLA-OFT | `QwenOFT` | MLP |
| StarVLA-FAST | `QwenFast` | Discrete autoregressive tokens |
| StarVLA-PI | `QwenPI_v3` | Flow-matching diffusion |
| StarVLA-GR00T | `QwenGR00T` | Dual-system (VLM System 2 + flow-matching System 1) |

World-Model-for-Action frameworks live in `starVLA/model/framework/WM4A/`.

### Training Entry Points (`starVLA/training/`)

- `train_starvla.py` — VLA-only (action data)
- `train_starvla_cotrain.py` — co-training VLA + VLM objectives
- `train_starvlm.py` — VLM-only
- `train_starvln.py` — VLN (vision-language navigation)

Training uses PyTorch + HuggingFace Accelerate + DeepSpeed (ZeRO-2/3). Supports Ascend NPU via `torch_npu`.

### Data Pipeline

Dataloaders return model-agnostic dicts with keys: `image` (list of PIL.Image), `lang` (str), `action` (ndarray), `state` (optional ndarray). The `build_dataloader()` in `starVLA/dataloader/__init__.py` dispatches to `lerobot_datasets` or `vlm_datasets` based on config. Datasets use LeRobot format. Data mixtures are defined in per-benchmark `data_config.py` files under `examples/<benchmark>/train_files/data_registry/`.

### Configuration

Single YAML config per experiment. CLI args override YAML via OmegaConf dotlist merging. Config is wrapped with `AccessTrackedConfig` for tracking used parameters.

### Evaluation

Client-server pattern via WebSocket. `deployment/model_server/server_policy.py` serves inference. Benchmark-specific `model2<benchmark>_interface.py` files handle observation/action translation.

### Examples Layout

- `examples/simBenchmarks/` — LIBERO, SimplerEnv, RoboTwin, Robocasa, DOMINO, Calvin, etc.
- `examples/modelExtensions/` — CoTrainVLM, Gemma4, MiniCPM, NeuralVLA
- `examples/realRobots/` — Franka, UnitreeG1, RoboChallenge

## Branching & PRs

- Two-branch model: `starVLA` (stable) and `starVLA_dev` (active development)
- All PRs target `starVLA_dev`, merged via squash merge
- Feature branch prefixes: `feat/`, `fix/`, `docs/`, `refactor/`, `exp/`, `hotfix/`
- Commit messages follow Conventional Commits format
- PR titles use `[type] (scope): description` format

## Conventions

- Files under any `**/bar/` directory are git-ignored (used for local/custom scripts)
- `playground/` is for local data/models/checkpoints (git-ignored)
- Each framework file in `starVLA/model/framework/VLM4A/` can be run standalone for smoke testing
- Adding a new framework: create module, register with `@FRAMEWORK_REGISTRY.register()`, add default config dataclass, add example YAML under `examples/`



# 撰写文档规范

* 图表用mermaid, 数学相关的用LaTex, 必要时可以用py脚本画一些更能帮助读者理解的图片.
* 分析要深入仔细, 既要包括纵向分析(算法或方法的由来与演进历史, 以及在该算法或方法的基础上又演进和优化出了些什么解决类似问题的方法, 新老方法各有什么优缺点, 各适合应用到什么场景), 纵向分析(同时期同类算法的对比分析, 不同算法或方法各有什么优缺点, 各适合应用到什么场景), 和 消融分析(算法或方法中哪些点是在benchmark实验或实践中被证明有效的, 哪些点相对来说更有效, 哪些没那么有效).
* 系统或程序的设计要包括静态架构(组件图,类图,组件和类的职责与关系等等)和动态架构(数据流图,序列图,工作流图,不同场景下的各组件或类的调用与协调图.如果是算法还会涉及forward阶段的数据流,模型组件间的调用,以及backwawrd阶段的数据流,gradient流,哪些权重冻结哪些会被更新,和模型组件间的调用等等).
* 解释要深入浅出, 图文并茂, 可以举一些易于理解的例子帮助说明, 对关键的逻辑也要进行深入的代码解读, 要用严谨的科普论文的风格.
