# EMW Qwen Nonclinical Semantic Control v0.2b

## Purpose
Test whether the Layer-18/19 effects observed in the clinical causal scan are psychiatric-specific or reflect general semantic mechanisms.

## Model/runtime
- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Revision: `c03e6d358207e414f1eca0bb1891e29f1db0e242`
- Same dual-T4 FP16 setup
- Frozen weights; no training

## Control task
Six non-clinical semantic axes mirrored the clinical logical structures:
- positive/present vs explicit denial
- checked-and-absent vs not assessed
- exact quantity vs uncertain quantity
- source conflict vs agreement
- reduced resource + preserved output vs reduced resource + poor output
- explicit correction vs confirmation

The correction axis again had only 0.50 baseline accuracy and remains unsuitable for strong causal conclusions.

## Main result
**Layer 18 attention is not psychiatry-specific.**

It also shows strong non-clinical representational separation and causal patching effects:
- L18 attention control patch: mean margin drop ≈ 2.704, flip rate ≈ 29.2%
- Clinical L18 attention patch: mean margin drop ≈ 3.928, flip rate ≈ 34.7%

L19 attention and L19/L20 MLP effects were also present in non-clinical controls.

## Direct matched comparison: L18 attention

| Semantic structure | Clinical separation | Control separation | Clinical/control ratio | Clinical patch drop | Control patch drop |
|---|---:|---:|---:|---:|---:|
| positive vs denial | 0.5714 | 0.2287 | 2.50× | 7.360 | 1.899 |
| denied vs not assessed | 0.6654 | 0.3505 | 1.90× | 4.178 | 2.609 |
| dose/quantity certain vs uncertain | 0.4604 | 0.4005 | 1.15× | 3.369 | 2.691 |
| source conflict vs agreement | 0.3616 | 0.5084 | 0.71× | 2.838 | 5.653 |
| decreased need vs reduced-resource analogue | 0.5207 | 0.1889 | 2.76× | 4.715 | 2.441 |
| self-correction | 0.2146 | 0.1832 | 1.17× | 1.105 | 0.931 |

## Interpretation
The data currently support a **general semantic role** for Layer-18 attention, with possible clinical amplification for some distinctions.

Strongest clinically amplified candidates:
- explicit symptom presence vs denial
- decreased need for sleep vs ordinary sleep-loss/fatigue
- denied vs not-assessed

Not clinically selective:
- quantity/dose uncertainty was similar to generic quantity uncertainty
- source conflict was actually stronger in the non-clinical control

Therefore it would be incorrect to call Layer 18 a psychiatric layer or to train only that full layer.

## Next step
Decompose Layer 18 (and Layer 19 as comparison) into their 28 query-head output slots before `o_proj`:
- head_dim = 128
- hidden_size = 3584
- 28 heads × 128 = 3584

Use:
1. final-token per-head ablation
2. opposite-pair per-head activation patching
3. clinical vs non-clinical differential effect

Only heads with reproducible clinical amplification should advance to deeper channel/projection analysis.
