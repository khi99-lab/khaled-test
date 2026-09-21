# EMW Qwen v0.5 — English-only adapter placement pilot

Preregistered 2026-09-21 before GPU baseline or training. This document is a plan, not a completed-results report.

## Question
Does a small learned internal adapter improve explicit evidence-state extraction, and does a distributed adapter outperform an equally sized layer-selective adapter under this specific fixed training recipe?
This pilot does not establish broad psychiatric expertise, an exclusive psychiatric circuit, or the best possible placement.

## Immutable references
- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`.
- Model revision: `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
- Engine: `qwen_lora_pilot_v05.py`, commit `6d2faaaff24c586faab3d0351fe2eb675c5b173e`.
- Engine SHA256: `16980b9b8ac0d4d918440f3187e33ce34e211c4860deb44754b7d101e8fcd9f1`.
- Required entry point: `qwen_lora_pilot_v05_protocol.py`, commit `abdb54621170a231a6a64da292d379132161b80f`.
- Protocol SHA256: `70bdc464bc15dd64eacf77b4f012d338c5e21eb465e55db5659c1a797628bbac`.

The protocol independently balances evaluation source/timeframe metadata. It corrects an accidental coupling in the generic engine's dataset builder BEFORE any outputs are observed. Run the protocol entry point, not the engine's standalone --run entry point. Both sources are retained for audit. Training examples, status labels, evaluation answers and model results are not altered post-run.

## Arms and equal parameter budget
1. Distributed: rank 4 in all 28 transformer layers (indices 0–27).
2. Selective: rank 14 in layers 13–20 inclusive.

Both target q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj and down_proj. Both use alpha/rank = 2, dropout = 0, bias = none, and freeze embeddings, output head, norms, original matrices and all non-adapter parameters.

Each adapter has exactly 10,092,544 trainable parameters, checked at runtime, about 0.133% of the original 7,615,616,512-parameter model. Location and per-layer rank necessarily differ jointly in this equal-budget comparison; a result cannot be attributed solely to location. No claim that layers 13–20 are psychiatric-specialist layers is made. This is not an edit limited to heads H21/H24.

## Runtime and comparison reference
Use a NEW private Kaggle notebook requesting Tesla T4 hardware. Where two GPUs are provided, each arm runs in its own subprocess on one GPU, with its own freshly loaded model and baseline. One-GPU availability falls back to sequential execution.

Use QLoRA: NF4 4-bit working matrices with double quantization, FP16 computation, and FP32 adapters. PEFT preparation can change dtypes of other working-copy tensors BEFORE baseline hashing. The official source checkpoint is not overwritten. Both before/after comparisons use this same prepared quantized working model, not previous v0.4 FP16 results. No auto-sharded inference device_map is used for training. No paid Hugging Face Job is used.

Pinned execution packages: transformers 4.57.6, peft 0.18.1, bitsandbytes 0.49.0, accelerate 1.12.0, huggingface_hub 0.36.0, tokenizers 0.22.2. Kaggle's installed Torch/CUDA is retained and recorded. Dependencies are installed in a notebook-local virtual environment under /kaggle/temp, not on any other session.

## Data and leakage boundaries
English only. All examples are synthetic and authored for this experiment. No patient information, board-question material, production chart data or old evaluation output files are used.

- Train: 400 examples = 256 clinical evidence-state examples + 128 nonclinical examples + 16 simple general-instruction replay examples.
- Development: 48 examples = 24 clinical + 24 nonclinical.
- Held-out evaluation: 96 examples = 48 clinical + 48 nonclinical.
- General regression: 32 short deterministic tasks (arithmetic, text transformation, comparison, extraction); not a general-reasoning benchmark.
- Four equally represented evidence states: present, absent, not_assessed and uncertain.
- Evaluation timeframe: 48 current and 48 historical examples.

Train/dev/test have disjoint target vocabulary and disjoint wording-template families, with exact input-overlap checks. Multiple rows share templates: 96 rows are NOT 96 independent clinical cases. Source and timeframe are explicitly provided; correctness on those fields is not evidence of free-form speaker attribution or longitudinal reasoning. The same instruction/schema is deliberately used before and after training. General replay and regression have different examples but shared operations.

Frozen canonical JSON hashes:
- train: `69e541cd1453a5140e3b339071a60ad471dedd5a00110e8edc5f85d81bcbdfdd`
- dev: `7699511cef4e9c9b61f1461335a2bba757a823ca4ae39050e413e15244282039`
- test: `7cf6b9db7b00f4b82dd699aade40f1ed530f1badca97801413ce3a686e90a209`
- regression: `6cb960cb81f38f8af09f6515ba7c8461d2a2cf1c97c9d8ccf79ec928ac85dc9a`

The evaluation is held out from optimization, not secretly inaccessible to the experiment author. Once reviewed, it is an exposed development evaluation and cannot be reused as a fresh confirmatory test for later tuning.

## Training recipe fixed before results
One seed: 1705. Exactly 100 optimizer steps per arm, one training pass, microbatch 1, accumulation 4. Identical shuffled example order and assistant-only loss masking. Learning rate 1e-4 with the source's fixed warmup/decay schedule. AdamW with weight decay 0 and clip norm 1. No truncation above the 512-token cap: overlong data fails explicitly. FP16 GradScaler and nonfinite-gradient checks stop only the affected experiment process, not other notebooks.

Only the final fixed-step adapter is compared. Step-50/100 adapter snapshots are saved for recovery, not selected by test performance. No test-dependent early stopping, learning-rate tuning, checkpoint selection or automatic reruns. No merge or export of a full changed base model. Adapters stay separately switchable.

## Validation and endpoints
A tiny randomly initialized Qwen2 CPU smoke test checks PEFT layer targeting, equal parameter budget, and gradient isolation before loading the real model. It is a code check, not evidence about Qwen's capabilities.

Before training, evaluate the model with zero-initialized adapters. Disabling them must restore identical generated output on a fixed sentinel. After training, hash every frozen parameter and bitsandbytes quantization-state tensor again and require equality; verify adapter tensors did change. Disabling the trained adapter must restore the sentinel reference. Compare frozen-base hashes and before-training outputs between arms.

Score separately:
- exact JSON schema compliance (no silent repair),
- status, source and timeframe correctness,
- exact evidence quote and whether it occurs in the input,
- all-field correctness,
- dangerous not_assessed -> absent and uncertain -> present/absent upgrades,
- general regression exact-match accuracy.

Primary endpoint: correct held-out status count out of 96, always also broken down into 48 clinical and 48 nonclinical. Invalid outputs count wrong; quote/format success is not proof of clinical correctness.

A fixed exploratory 'pilot promising' flag requires at least +5 status-correct cases, no increase in either tracked upgrade error, no more than one exact-evidence loss and no more than one general-regression loss. This is a researcher-chosen development rule, NOT clinical validation. High baseline ceiling may make the +5 rule unattainable, which must be reported rather than silently relaxed. Report all metrics even if no arm passes. One seed and few template families do not establish a statistically reliable winner.

## Preservation and outputs
No production EM-AI change; no old notebook stopped, restarted or overwritten. Separate run directory and private notebook. Outputs include per-arm raw baseline and final generations, metrics, adapter files, training loss/gradient logs, dependency environment, all frozen-base hashes, data hashes and a comparison report. Model downloads and virtual environments remain outside /kaggle/working to avoid publishing huge caches as results.

## Primary method references
- https://huggingface.co/docs/peft/developer_guides/quantization
- https://huggingface.co/docs/peft/package_reference/lora
- https://arxiv.org/abs/2106.09685
- https://arxiv.org/abs/2305.14314

No improvement or successful real-model training is claimed by this preregistration.
