# EMW Qwen v0.4 — measurement audit and head confirmation

## Execution reference
- Kaggle: https://www.kaggle.com/code/khamed87/emw-qwen-validation-v0-4
- Kernel ID: 135231064; version: 1.
- Submitted on 2026-09-21. Kaggle returned RUNNING after submission. This is not a completed-results report.
- Source commit: `9c534d9b0b130f4167dff315f1ff56ac23e18ce2`.
- Source file: `emw_qwen_research/qwen_validation_v04.py`.
- Source SHA256: `095f3c414fdfc217f2d7a94cb53fc91430f3f2faf67677e3412cb6f1b1aad8ef`.
- Canonical generated dataset SHA256 (JSON with ensure_ascii=False, sort_keys=True): `a2cbd4960e9542a91f6355ffa1415369b894cda107494608b4d418e8528d86c8`.
- Local syntax and 618 counted self-test assertions passed. These are code/data checks, not Qwen performance tests. Local Transformers execution was not available.

## Frozen reference
`Qwen/Qwen2.5-Coder-7B-Instruct`, revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
FP16 execution on the requested Kaggle Tesla T4 machine shape; SDPA attention. GPU devices, package versions, attention implementation and device map are saved by the running script. This run is not bit-identical BF16 inference and is not a directly matched comparison against v0.3 prompts or batching.
No training, optimizer, backward, LoRA, parameter editing, pruning or checkpoint writes. No production EM-AI changes. No existing Kaggle session is stopped or restarted by this experiment.

## Dataset and preregistered coordinates
120 synthetic statements / 60 matched pairs: 60 clinical, 60 nonclinical. 72 English statements and 48 Arabic statements. Three axes: assessed denial versus not assessed; certain versus uncertain reported value; source conflict versus agreement. These are template-based research probes, not a validated clinical competency benchmark or independent patients.
Candidate coordinates: layer 18, heads 21 and 24, separately and together. Head 8 is a previously low-effect comparison coordinate. No reranking on the new sample.

## Measurement audit
1. Normal and reversed A/B semantic assignments, with actual greedy label generation.
2. Score both bare and space-prefixed A/B tokens, record probability mass and top token.
3. Descriptive category sequence likelihoods through the category delimiter, plus category-and-evidence generation for every case.
4. Explicit position IDs for padded forward passes; single versus mixed-batch comparison on a predefined 24-statement panel in both mappings.
5. No-op self-patching with identical cached activations.
6. Invalid generated outputs count as invalid/wrong, not silently accepted.

Technical continuation gates are fixed in the script: no-op maximum margin difference <=0.001 with no prediction changes; single/batch maximum difference <=0.15 with no prediction changes. Failure produces `AUDIT_FAILED_NO_CAUSAL_SCAN`, preserving the audit and skipping the head confirmation stage of THIS notebook only.
Interpretation eligibility is separately reported for each domain/axis/language stratum: letter accuracy >=0.8; swap consistency, generated label validity, and generated/scored agreement each >=0.9. All strata and rows remain available, including failures.

## Head confirmation
Three positions: last prompt token, end of the evidence span, and a shared neutral prefix before the evidence. Zero each candidate separately, both together, and control head; opposite-state and same-state donor patching. Same-state donors are different topics within the same domain, language and axis; they are not perfect meaning-identical paraphrases. Do not overinterpret this control.
Descriptive label likelihood interventions are run on all 120 cases. Generated descriptive outputs under selected joint-head interventions use the predefined 24-statement sentinel panel. Generation is modified at prompt prefill only, not subsequent generated tokens. Evidence quotation correctness needs independent review.

## Integrity, outputs and limits
Hash every loaded FP16 parameter byte before and after; do not infer full integrity from samples. Record changes by tensor. Save raw outputs, token positions, audit, intervention rows, activations, paired bootstrap summaries, environment and output-file checksums.
Pair-level bootstrap treats the two statements and two letter assignments as one cluster. Intervals remain exploratory: templates are reused and no multiple-comparison-adjusted significance is claimed.
Kaggle notebook cap: 7200 seconds. Internal process cap: 6900 seconds. These are resource limits, not completion-time promises. The notebook uses the existing free Kaggle quota; no paid Hugging Face job was launched.

## Interpretation boundary
Successful execution is not clinical approval. Margin shifts do not establish psychiatric specificity, clinical accuracy, or the best site for LoRA. Read the measurement audit before interpreting any causal ranking. No claim of improvement is made by this run: the underlying model is unchanged.
