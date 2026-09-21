# EMW Qwen Causal Component Scan v0.2

## Status
PASS — causal component scan completed on the frozen official Qwen checkpoint.

## Model
- `Qwen/Qwen2.5-Coder-7B-Instruct`
- Revision: `c03e6d358207e414f1eca0bb1891e29f1db0e242`
- Parameters: `7,615,616,512`
- Runtime: 2 × Tesla T4, FP16
- Candidate layers: 13, 16, 18, 19, 20
- Activation-patching layers: 18, 19, 20
- 6 semantic/clinical axes × 6 matched pairs = 72 statements

## Integrity controls
- training_performed: false
- optimizer_created: false
- backward_called: false
- adapter_loaded: false
- lora_loaded: false
- trainable_parameter_count: 0
- weights_modified_by_experiment: false

Interventions were **final-token component only**. This matters: results should not be interpreted as whole-layer necessity.

## Baseline forced-choice task

Accuracy:
- explicit positive vs denial: 1.00
- denied vs not assessed: 1.00
- dose certain vs uncertain: 1.00
- source conflict vs agreement: 1.00
- decreased need vs ordinary sleep deprivation: 1.00
- explicit self-correction vs no correction: 0.50
- overall: 0.9167

The self-correction axis is therefore not a valid high-confidence causal benchmark in this version and must be redesigned before strong conclusions are drawn from it.

## Representational component separation

Across the five well-performing axes, a recurrent pattern emerged:

- **Layer 18 attention** was usually the strongest or near-strongest matched-pair separator.
- **Layer 19 attention** and **Layer 19 MLP** were frequently next.
- **Layer 20 MLP** also carried substantial separation.
- Residual-stream separation was smaller than component-level attention/MLP separation in these tests.

Examples of mean matched-pair `1 - cosine` separation:

| Axis | Strongest observed component |
|---|---|
| explicit positive vs denial | L18 attention ≈ 0.5714 |
| denied vs not assessed | L18 attention ≈ 0.6654 |
| dose certain vs uncertain | L18 attention ≈ 0.4604 |
| source conflict vs agreement | L19 attention ≈ 0.4527 (L19 MLP ≈ 0.4389) |
| decreased need vs sleep deprivation | L18 attention ≈ 0.5207 |
| self-correction vs no correction | L18 attention ≈ 0.2146, but baseline task accuracy was only 0.50 |

These are correlations, not yet evidence of psychiatric specificity.

## Final-token ablation

Zeroing the final-token attention contribution at **Layer 18** caused the largest overall margin reduction:

- L18 attention: mean correct-margin change ≈ **-1.348**
- L19 attention: ≈ **-0.609**
- L20 MLP: ≈ **-0.205**
- L19 MLP: ≈ **-0.196**
- L18 MLP: ≈ **-0.088**

Accuracy did not drop in this ablation test, indicating redundancy / recoverability. The intervention weakened confidence but usually did not cross the decision boundary.

Largest L18-attention margin reductions by reliable axis:
- explicit positive vs denial: ≈ -2.738
- decreased need vs sleep deprivation: ≈ -1.481
- denied vs not assessed: ≈ -1.376
- dose certainty vs uncertainty: ≈ -1.168
- source conflict vs agreement: ≈ -1.112

## Matched activation patching

Replacing the target statement's final-token component output with the **opposite matched statement's** component produced much stronger causal effects.

Overall:
- **L18 attention**: mean target-margin drop ≈ **3.928**, prediction flip rate ≈ **34.7%**
- L19 attention: margin drop ≈ 2.383, flip rate ≈ 6.9%
- L19 MLP: margin drop ≈ 1.268, flip rate ≈ 2.8%
- L20 MLP: margin drop ≈ 1.245, flip rate ≈ 2.8%
- L18 MLP: margin drop ≈ 0.593, flip rate 0%
- L20 attention: essentially no overall effect

Strong L18-attention patch effects:
- explicit positive vs denial: margin drop ≈ **7.36**, flip rate ≈ **66.7%**
- denied vs not assessed: ≈ **4.18**, flip rate **50%**
- decreased need vs sleep deprivation: ≈ **4.71**, flip rate **33.3%**
- source conflict vs agreement: ≈ **2.84**, flip rate **50%**
- dose certain vs uncertain: ≈ **3.37**, flip rate ≈ **8.3%**

This is the strongest causal evidence obtained so far that the Layer-18 attention contribution at the final token participates materially in these distinctions.

## Interpretation

The evidence supports:
1. Layer 18 attention is not merely correlated with several tested distinctions; transplanting its final-token output from the opposite matched statement frequently moves or flips the model's classification.
2. Layer 19 attention carries a weaker but still reproducible causal signal.
3. MLP effects appear later/more distributed, especially Layer 19 and Layer 20.
4. Simple ablation shows redundancy: removing one component often reduces confidence without changing the final label.
5. Activation patching is more diagnostic than zero ablation because it injects a structured opposite-state representation rather than only deleting information.

The evidence does **not** yet support:
- “Layer 18 is a psychiatry layer.”
- editing or pruning Layer 18 directly.
- training a LoRA only on Layer 18.
- interpreting any individual head/neuron as a psychiatric concept.

The same logical distinctions (negation, uncertainty, disagreement, correction) exist outside psychiatry. A non-clinical semantic control is required before claiming domain specificity.

## Next required experiment
Run matched **non-clinical controls** with the same logical structures:
- present vs denied
- known vs uncertain
- disagreement vs agreement
- correction vs confirmation
- low-resource-but-energized vs low-resource-and-fatigued analogues where appropriate

Compare L18/L19 attention and L19/L20 MLP separation, ablation, and patching effects. Only effects selectively amplified for psychiatric/clinical content should become candidates for domain-specialist intervention.

## Artifacts
Kaggle:
- `phase2_summary.json`
- `baseline_rows.json`
- `component_separation.json`
- `candidate_component_vectors.npz`
- `ablation_summary.json`
- `ablation_rows.json`
- `patch_summary.json`
- `patch_rows.json`

Implementation:
- `emw_qwen_research/qwen_causal_component_scan_v02.py`
- pinned code commit: `289bae7ffb3596b788d47763aa68f533580fdc17`
