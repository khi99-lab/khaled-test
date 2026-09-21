# EMW Qwen Original Brain Baseline v0.1

## Status
PASS — first unmodified-model functional baseline completed on Kaggle.

## Model identity
- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Hugging Face revision: `c03e6d358207e414f1eca0bb1891e29f1db0e242`
- Parameter count: `7,615,616,512`
- Runtime: 2 × Tesla T4
- Runtime dtype: FP16
- Hidden-state stages captured per prompt: 29 (embedding + 28 transformer layers)

## No-training / no-weight-update controls
- training_performed: false
- optimizer_created: false
- backward_called: false
- adapter_loaded: false
- lora_loaded: false
- trainable_parameter_count: 0
- weights_modified_by_experiment: false

Selected tensor fingerprints were identical before/after for:
- `model.embed_tokens.weight`
- `model.layers.0.self_attn.q_proj.weight`
- `model.layers.13.mlp.up_proj.weight`
- `model.layers.27.self_attn.o_proj.weight`
- `lm_head.weight`

Important: the checkpoint was loaded in FP16 because the Tesla T4 does not provide the BF16 execution path we want for a bit-for-bit precision-matched reference. This run proves no training/update occurred during the experiment; it is not a BF16-equivalent numeric execution claim.

## Minimal-pair behavioral baseline

### Depression positive vs denial
- Positive statement: correctly surfaced as an explicit fact.
- Denial: recognized semantically, but output schema changed from list fields to booleans.

### Psychosis positive vs denial
- Positive auditory-hallucination statement was captured.
- Denial statement was placed in `facts` while `negated` remained empty.
- This is a concrete baseline weakness to reproduce with paraphrases.

### Decreased need for sleep vs ordinary sleep loss
- 3 hours of sleep + high energy + no tiredness was distinguished from 3 hours + daytime exhaustion.
- No diagnosis was requested or generated.

### Medication dose certainty vs uncertainty
- Certain 50 mg dose was represented as certain.
- “50 mg, maybe 100 mg; I am not sure” preserved uncertainty.
- This pair produced the strongest internal representation separation in the initial six-pair set.

### Collateral conflict vs agreement
- Patient/mother conflict was recognized as a source conflict.
- Agreement condition was not marked as conflict.
- Schema typing again varied between lists and booleans.

### Assessed-negative vs not-assessed
- Explicit denial of panic attacks was differentiated from “panic attacks were not assessed.”
- This simple pair behaved as intended.

## Normalized hidden-state findings

To reduce the confound from late-layer norm growth, the first analysis used:
- angular separation: `1 - cosine_similarity`
- relative L2: `||a-b|| / mean(||a||, ||b||)`

### Aggregate highest mean angular separation across six pairs
1. Layer 18 — mean 1-cos ≈ 0.013505; mean relative L2 ≈ 0.15397
2. Layer 19 — ≈ 0.012998; ≈ 0.14759
3. Layer 20 — ≈ 0.010640; ≈ 0.13427
4. Layer 13 — ≈ 0.008596; ≈ 0.12553
5. Layer 16 — ≈ 0.008497; ≈ 0.12358
6. Layer 15 — ≈ 0.008494; ≈ 0.12393
7. Layer 17 — ≈ 0.008325; ≈ 0.12201
8. Layer 14 — ≈ 0.007932; ≈ 0.12037

### Pair-specific strongest angular regions
- Depression positive/negative: layers 16, 15, 13, 18, 14, 17
- Psychosis positive/negative: layers 18, 16, 17, 15, 14, 13
- Decreased need vs sleep loss: layers 18, 19, 13, 15, 14, 20
- Dose certain/uncertain: layers 19, 18, 20, 21, 17, 16
- Collateral conflict/agreement: layers 18, 19, 20, 21
- Assessed-negative/not-assessed: layers 19, 18, 20, 13, 7, 15

## Interpretation
The repeated concentration around layers ~13–20, especially 18–20, makes that region a strong candidate for deeper instrumentation.

It does **not** establish that these are “psychiatry layers.” The result may reflect a more general semantic/state-discrimination region. Causal tests, larger paraphrase sets, token-level traces, attention/MLP decomposition, ablation, and activation patching are required before any surgical target is accepted.

## Next experimental phase
1. Expand each concept into many paraphrased minimal pairs and lexical controls.
2. Capture decisive-token representations, not only the final prompt token.
3. Build frozen linear probes layer-by-layer.
4. Decompose candidate layers into attention vs MLP effects.
5. Test head/channel ablation and activation patching.
6. Only then test a small reversible LoRA/adapter intervention on experimentally supported modules/layers.
7. Compare against this frozen baseline before any merge.

## Artifacts
Kaggle output:
- `baseline_summary.json`
- `pair_results.json`
- `hidden_last_token_vectors.npz`
- `top_layer_contrasts.json`

Baseline implementation:
- `emw_qwen_research/qwen_original_baseline_v01.py`
- pinned GitHub commit: `0e8ca0c85603bbe9f8f1ed1fca74414c95cbb3ba`
