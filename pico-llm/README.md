# pico-llm numeric sequences: baseline vs. linear attention

This repo extends the starter pico-llm code with a small research pipeline on synthetic integer sequences (counting, arithmetic, geometric, alternating, random walk). Models are tiny decoder-only Transformers trained for next-token prediction on CPU-friendly workloads.

## 项目训练/数据/评测逻辑概览

### 主入口
- SFT + DPO 统一入口：`train/train_dpo.py`（命令行运行 `python -m train.train_dpo ...`）。
- 纯 SFT/对比实验：`train/train_baseline.py`、`train/train_linear.py`、`experiments/compare_baseline_linear.py`。

### 数据导入
- 默认数据源：TinyStories，通过 `datasets.load_dataset("roneneldan/TinyStories")` 读取。
- Tokenizer：默认 GPT-2（tiktoken）；如指定 `--hf_model_name gpt2`，则使用 HF tokenizer，自动补充 `pad_token`，并按 `seq_max_len` 及模型 `model_max_length` 截断。
- 划分：按比例拆分 train/val/test（默认 0.8/0.1/0.1）；`--tinystories_limit` 控制采样数量。
- 偏好数据（DPO）：从真实文本切分 prompt+chosen，rejected 通过扰动/打乱或引入其他样本续写生成。

### 模型
- 自定义小模型：`models/transformer_baseline.py`（标准注意力），`models/transformer_linear.py`（线性注意力）。
- HF 预训练模型：`--hf_model_name`（如 `gpt2`），封装于 `models/hf_wrapper.py`，可与自定义 tokenizer 一致。
- Checkpoint 读写：`models/checkpoint.py` 支持 `baseline/sft/dpo_policy/hf` 类型。

### 训练流程（train/train_dpo.py）
- SFT：在 TinyStories 上做下一词预测，参数 `--sft_epochs/--sft_learning_rate/--tinystories_limit/--seq_max_len` 等。
- DPO：加载 SFT 参考模型，初始化策略为参考权重；在偏好对上优化 DPO 损失（chosen vs rejected），支持早停 `--early_stop_patience/--early_stop_metric`。
- 设备：`--device cpu|cuda:0`；HF 模型建议 GPU。

### 评测与输出
- SFT：验证集 loss/acc，曲线 `sft_loss_curve.png`，指标 `metrics.json`。
- DPO：偏好准确率 `pref_acc`（chosen 胜率）和 DPO loss，历史记录在 `dpo_metrics.json`。
- 日志：所有进度使用 `print` 输出，需结合 `tee` 落盘，例如 `... | tee results/sft_only/sft.log`。
- 目录：SFT 结果在 `results/sft_only/run-*/`，DPO 结果在 `results/dpo_run/run-*/`。

### 示例命令
```bash
# SFT（GPT-2 基座，10 轮）
python -m train.train_dpo \
  --device cuda:0 \
  --hf_model_name gpt2 \
  --tinystories_limit 5000 \
  --seq_max_len 128 \
  --beta 0.1 --length_normalize \
  --epochs 0 --sft_epochs 10 \
  --output_dir results/sft_only \
  2>&1 | tee results/sft_only/sft.log

# DPO（基于上述 SFT ckpt，对齐偏好）
python -m train.train_dpo \
  --device cuda:0 \
  --hf_model_name gpt2 \
  --sft_ckpt results/sft_only/run-*/sft/sft_checkpoint.pt \
  --tinystories_limit 5000 \
  --seq_max_len 128 \
  --beta 0.1 --length_normalize \
  --train_pairs 800 --val_pairs 200 --test_pairs 200 \
  --epochs 5 \
  --early_stop_patience 3 --early_stop_metric pref_acc \
  --output_dir results/dpo_run \
  2>&1 | tee results/dpo_run/dpo.log
```

## Layout
- `data/` synthetic sequence generator and padding utilities.
- `models/` baseline softmax Transformer and linear-attention variant (+ checkpoint helpers).
- `train/` training scripts for each model, shared trainer + dataloader helpers.
- `analysis/` interpretability tools: embedding PCA, attention heatmaps, neuron activation stats.
- `experiments/` end-to-end comparison entrypoint.
- `results/` default artifact directory (figures, logs, checkpoints).

## Quick start
Install dependencies (PyTorch, matplotlib, numpy):
```bash
pip install torch matplotlib numpy
```

### Train baseline (softmax) model
```bash
python train/train_baseline.py --device cpu --epochs 10 --output_dir results/baseline
```
Checkpoints, loss curves, and metrics land under `results/baseline/run-*/`.

### Train linear-attention model
```bash
python train/train_linear.py --device cpu --epochs 10 --output_dir results/linear
```

### Interpretability
- Embedding PCA:
  ```bash
  python analysis/embedding_viz.py --checkpoint <path/to/checkpoint.pt> --output results/embedding_pca.png
  ```
- Attention heatmaps (works for both models; linear shows kernel-based weights):
  ```bash
  python analysis/attention_viz.py --checkpoint <checkpoint> --seq_type arithmetic --length 12 --output_dir results/attention
  ```
- Neuron activation selectivity across sequence types:
  ```bash
  python analysis/activations.py --checkpoint <checkpoint> --output_dir results/activations
  ```

### Baseline vs. linear comparison
```bash
python experiments/compare_baseline_linear.py --device cpu --epochs 10 --output_dir results/compare
```
This trains (or loads if you point `--baseline_ckpt/--linear_ckpt`), reports train/val/test metrics, evaluates longer sequences, plots combined loss curves, and re-runs interpretability snapshots for both models. Outputs live under `results/compare/run-*/`.

## DPO post-training (Lecture 9 extension)
- Install deps for text DPO: `pip install datasets tiktoken`.
- Default mode trains/loads an SFT baseline on TinyStories, then runs DPO on TinyStories-derived preference pairs (prompt + true continuation vs. corrupted continuation):
  ```bash
  python train/train_dpo.py --device cpu --tinystories_limit 5000 --seq_max_len 128 --beta 0.1 --length_normalize
  # or reuse an SFT checkpoint:
  python train/train_dpo.py --device cpu --sft_ckpt path/to/sft_checkpoint.pt --beta 0.1 --length_normalize
  ```
- What it does: loads TinyStories from HuggingFace, encodes with GPT-2 tokenizer (+pad token), trains an SFT reference (unless provided), creates preference pairs by splitting each story into prompt/continuation and corrupting the continuation for the rejected sample, then optimizes DPO against the frozen reference. Artifacts live under `results/dpo/run-*/`.
- Key knobs: `--tinystories_limit` (subset size), `--seq_max_len` (truncate tokens), `--beta` (preference sharpness), `--train_pairs/--val_pairs/--test_pairs` (preference counts), `--length_normalize` (divide logprobs by response length), `--max_steps` (cap steps per epoch for quick runs). Set `--data_source synthetic` to fall back to the toy numeric pipeline.
- Metrics: preference accuracy on val/test splits plus loss curves; SFT history is stored alongside if auto-trained.

## Notes
- Padding token is reserved as the last vocab id; real tokens stay in `[0, vocab_size-2]`.
- Default configs are tiny (d_model=128, 2 layers) to keep CPU runs fast (<1–2 minutes per run on typical laptops).
- Figures and JSON summaries are written into `results/`; feel free to clean this directory between runs.
