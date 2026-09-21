# v0.5 launch record — 2026-09-21

## Scope and authorization
Owner authorized starting the English-only distributed-versus-selective adapter training pilot. Prior Kaggle notebooks and production EM-AI are unchanged. No paid Hugging Face Job was launched.

## Source references
- Engine: 6d2faaaff24c586faab3d0351fe2eb675c5b173e, emw_qwen_research/qwen_lora_pilot_v05.py
- Dataset protocol entry point: abdb54621170a231a6a64da292d379132161b80f, emw_qwen_research/qwen_lora_pilot_v05_protocol.py
- Preregistration: 107e84d75edf5de74a6553ad8b78ec431a1a464d, emw_qwen_research/V05_PREREGISTRATION_AND_RUN.md

## First submission
https://www.kaggle.com/code/khamed87/emw-qwen-lora-placement-pilot-v0-5
Kernel ID 135244624, version 1.
It ended ERROR during venv creation: the ensurepip subprocess failed. Both source hash checks passed. It did not reach model loading or training. No result-driven adaptation of data, hyperparameters or checkpoint selection occurred.

## Separate retry
https://www.kaggle.com/code/khamed87/emw-qwen-lora-placement-pilot-v0-5-run-2
Kernel ID 135244777, version 1.
Only the dependency bootstrap changed: create the notebook-local virtual environment without ensurepip, then install pinned packages with the existing pip --python option into that environment.
The model, engine, protocol, datasets, seeds, scoring, training steps and parameter budget are unchanged.

Last status read at approximately 2026-09-21 13:10 UTC: RUNNING. This is a notebook execution state, not proof of a completed model load, optimizer update, trained adapter or improvement. No training-result artifacts were available at this read. Read current Kaggle status/output for later progress; this launch record is deliberately immutable.

## Planned experiment
- Distributed: rank 4 across layers 0–27.
- Selective: rank 14 across layers 13–20.
- Exactly 10,092,544 trainable adapter parameters per arm, asserted at runtime.
- 100 optimizer steps each; identical 400-example training order and assistant-only loss masking.
- 48 development, 96 held-out task examples (48 clinical, 48 nonclinical), and 32 general regression cases.
- English-only synthetic narrow evidence-state task, not clinical competency or broad psychiatric specialization.
- NF4 QLoRA working copy; new matched before-training baseline. No direct numerical comparison to earlier FP16 v0.4 is justified.
- Frozen-base and quantization-state hashing before/after, separately saved adapters, no merge or full-model overwrite.

Actual completions and metrics must be reported separately from these intended settings.
