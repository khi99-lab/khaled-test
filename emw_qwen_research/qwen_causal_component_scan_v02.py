import gc
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
CANDIDATE_LAYERS = [13, 16, 18, 19, 20]
PATCH_LAYERS = [18, 19, 20]
OUT = Path("/kaggle/working/emw_qwen_causal_scan_v02")
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(17)

AXES = {
    "explicit_positive_vs_denial": {
        "a": "A = the symptom or experience is explicitly present/reported.",
        "b": "B = the symptom or experience is explicitly denied.",
        "pairs": [
            ("dep", "The patient reports feeling depressed most days.", "The patient denies feeling depressed."),
            ("voices", "The patient reports hearing voices when alone.", "The patient denies hearing voices when alone."),
            ("panic", "The patient reports having panic attacks twice this week.", "The patient denies having panic attacks."),
            ("nightmares", "The patient reports recurrent nightmares.", "The patient denies having nightmares."),
            ("checking", "The patient reports repeatedly checking the door lock.", "The patient denies repeatedly checking the door lock."),
            ("racing", "The patient reports racing thoughts at night.", "The patient denies racing thoughts at night."),
        ],
    },
    "denied_vs_not_assessed": {
        "a": "A = the item was explicitly assessed and denied.",
        "b": "B = the item was explicitly not assessed.",
        "pairs": [
            ("panic", "The patient explicitly denies panic attacks.", "Panic attacks were not assessed during this visit."),
            ("voices", "The patient explicitly denies hearing voices.", "Auditory hallucinations were not assessed during this visit."),
            ("appetite", "The patient explicitly denies appetite changes.", "Appetite was not assessed during this visit."),
            ("nightmares", "The patient explicitly denies nightmares.", "Nightmares were not assessed during this visit."),
            ("compulsions", "The patient explicitly denies compulsive behaviors.", "Compulsive behaviors were not assessed during this visit."),
            ("substance", "The patient explicitly denies alcohol use.", "Alcohol use was not assessed during this visit."),
        ],
    },
    "dose_certain_vs_uncertain": {
        "a": "A = one exact medication dose is explicitly stated as certain.",
        "b": "B = the medication dose is explicitly uncertain or ambiguous.",
        "pairs": [
            ("sertraline", "I take sertraline 50 mg every morning.", "I take sertraline 50 mg, maybe 100 mg; I am not sure."),
            ("quetiapine", "I take quetiapine 100 mg at bedtime.", "I think my quetiapine is 100 mg or 200 mg at bedtime; I am unsure."),
            ("fluoxetine", "My fluoxetine dose is definitely 20 mg daily.", "My fluoxetine might be 20 mg or 40 mg daily; I cannot remember."),
            ("lamotrigine", "I take lamotrigine 150 mg each evening.", "I believe I take lamotrigine 100 mg or 150 mg each evening; I am not certain."),
            ("aripiprazole", "I take aripiprazole 10 mg every day.", "I am not sure whether my aripiprazole is 5 mg or 10 mg every day."),
            ("methylphenidate", "My methylphenidate dose is 18 mg every morning.", "My methylphenidate dose may be 18 mg or 27 mg every morning; I am unsure."),
        ],
    },
    "source_conflict_vs_agreement": {
        "a": "A = the two speakers explicitly disagree about the same clinical issue.",
        "b": "B = the two speakers explicitly agree about the same clinical issue.",
        "pairs": [
            ("sleep", "The patient says she sleeps well. Her mother says she wakes repeatedly every night.", "The patient says she sleeps well. Her mother also says she sleeps well every night."),
            ("adherence", "The patient says he takes every dose. His father says he misses medication several times a week.", "The patient says he takes every dose. His father also says he takes every dose."),
            ("appetite", "The patient says her appetite is normal. Her spouse says she has barely been eating.", "The patient says her appetite is normal. Her spouse also says her appetite is normal."),
            ("school", "The child says school concentration is fine. The parent says the child cannot stay focused in class.", "The child says school concentration is fine. The parent also says concentration at school is fine."),
            ("mood", "The patient says his mood has been good. His partner says he has seemed persistently sad.", "The patient says his mood has been good. His partner also says his mood has been good."),
            ("panic", "The patient says she has had no panic episodes. Her sister says she witnessed two panic episodes this week.", "The patient says she has had no panic episodes. Her sister also says she has seen no panic episodes."),
        ],
    },
    "decreased_need_vs_sleep_deprivation": {
        "a": "A = markedly reduced sleep with preserved or increased energy and no tiredness.",
        "b": "B = reduced sleep accompanied by tiredness, fatigue, or exhaustion.",
        "pairs": [
            ("p1", "For four nights I slept only 3 hours but felt unusually energetic and did not feel tired.", "For four nights I slept only 3 hours and felt exhausted during the day."),
            ("p2", "I have slept about 2 hours nightly for three days and still feel full of energy.", "I have slept about 2 hours nightly for three days and feel drained and sleepy."),
            ("p3", "Despite sleeping only 4 hours, I wake up energized and do not need more sleep.", "After sleeping only 4 hours, I wake up fatigued and wish I could sleep longer."),
            ("p4", "I barely slept this week, yet I feel unusually active and never tired.", "I barely slept this week, and I feel tired and slowed down."),
            ("p5", "Three hours of sleep feels sufficient and I remain highly energetic.", "Three hours of sleep leaves me exhausted and unable to function well."),
            ("p6", "I am sleeping far less than usual but I do not feel sleepy at all.", "I am sleeping far less than usual and I feel sleepy throughout the day."),
        ],
    },
    "explicit_self_correction_vs_no_correction": {
        "a": "A = the speaker explicitly corrects or replaces an earlier statement.",
        "b": "B = the speaker confirms a statement without correcting it.",
        "pairs": [
            ("sleep", "Earlier I said I sleep 8 hours, but that was wrong; it is actually about 5 hours.", "Earlier I said I sleep 5 hours, and that is correct."),
            ("dose", "I said 100 mg before, but I need to correct that: my dose is 50 mg.", "I said 50 mg before, and yes, my dose is 50 mg."),
            ("days", "I first said the symptoms started two days ago, but actually they started two weeks ago.", "I first said the symptoms started two weeks ago, and that timing is correct."),
            ("frequency", "I said once a month earlier; correction, it happens about once a week.", "I said once a week earlier, and that frequency is correct."),
            ("adherence", "I told you I never miss doses, but I need to correct that; I miss about two doses a week.", "I told you I miss about two doses a week, and that is correct."),
            ("appetite", "I said my appetite was normal, but let me correct that: it has been much lower.", "I said my appetite has been much lower, and that is correct."),
        ],
    },
}

print("PHASE2_START", flush=True)
print("MODEL", MODEL_ID, REVISION, flush=True)
print("CUDA", torch.cuda.is_available(), torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    print("GPU", i, torch.cuda.get_device_name(i), flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=False,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

layers = model.model.layers
assert len(layers) == 28, len(layers)
input_device = model.get_input_embeddings().weight.device
param_count = sum(p.numel() for p in model.parameters())
trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("PARAM_COUNT", param_count, "TRAINABLE", trainable_count, flush=True)

def tensor_part(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    raise TypeError(type(out))

def replace_final(out, vec=None, zero=False):
    t = tensor_part(out)
    y = t.clone()
    if zero:
        y[:, -1, :] = 0
    else:
        y[:, -1, :] = vec.to(device=y.device, dtype=y.dtype)
    if torch.is_tensor(out):
        return y
    if isinstance(out, tuple):
        return (y,) + tuple(out[1:])
    return [y] + list(out[1:])

def pick_answer_tokens():
    for aa, bb in [(" A", " B"), ("A", "B"), (" 0", " 1"), ("0", "1")]:
        ia = tokenizer.encode(aa, add_special_tokens=False)
        ib = tokenizer.encode(bb, add_special_tokens=False)
        if len(ia) == 1 and len(ib) == 1 and ia[0] != ib[0]:
            return aa, bb, ia[0], ib[0]
    raise RuntimeError("No single-token answer pair found")

ans_a_text, ans_b_text, tok_a, tok_b = pick_answer_tokens()
print("ANSWER_TOKENS", repr(ans_a_text), tok_a, repr(ans_b_text), tok_b, flush=True)

def prompt_for(axis_name, text):
    cfg = AXES[axis_name]
    return (
        "Classify the statement using exactly one label.\n"
        + cfg["a"] + "\n"
        + cfg["b"] + "\n"
        + "Statement: " + text + "\n"
        + "Answer:"
    )

def encode_prompt(axis_name, text):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_for(axis_name, text)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    x = tokenizer(rendered, return_tensors="pt")
    return {k: v.to(input_device) for k, v in x.items()}

def correct_margin(logits, label):
    la = float(logits[tok_a].float().cpu())
    lb = float(logits[tok_b].float().cpu())
    pred = "A" if la >= lb else "B"
    margin = (la - lb) if label == "A" else (lb - la)
    return margin, pred, la, lb

def fingerprint(m):
    names = [
        "model.embed_tokens.weight",
        "model.layers.13.self_attn.q_proj.weight",
        "model.layers.18.mlp.up_proj.weight",
        "model.layers.20.self_attn.o_proj.weight",
        "lm_head.weight",
    ]
    ps = dict(m.named_parameters())
    out = {}
    for name in names:
        p = ps[name].detach()
        f = p.reshape(-1)
        ids = [0, f.numel() // 3, 2 * f.numel() // 3, f.numel() - 1]
        out[name] = [float(f[j].float().cpu()) for j in ids]
    return out

fingerprint_before = fingerprint(model)

samples = []
for axis_name, cfg in AXES.items():
    for pair_index, (pair_id, text_a, text_b) in enumerate(cfg["pairs"]):
        samples.append({"axis": axis_name, "pair_index": pair_index, "pair_id": pair_id, "side": "A", "label": "A", "text": text_a})
        samples.append({"axis": axis_name, "pair_index": pair_index, "pair_id": pair_id, "side": "B", "label": "B", "text": text_b})

cache = {}
baseline_rows = []

def capture_sample(sample):
    captures = {}
    handles = []
    def make_hook(layer_idx, comp):
        def hook(module, inp, out):
            captures[(layer_idx, comp)] = tensor_part(out)[0, -1, :].detach().float().cpu()
        return hook
    for li in CANDIDATE_LAYERS:
        handles.append(layers[li].self_attn.register_forward_hook(make_hook(li, "attn")))
        handles.append(layers[li].mlp.register_forward_hook(make_hook(li, "mlp")))
    x = encode_prompt(sample["axis"], sample["text"])
    with torch.inference_mode():
        out = model(**x, output_hidden_states=True, use_cache=False, return_dict=True)
    for h in handles:
        h.remove()
    for li in CANDIDATE_LAYERS:
        captures[(li, "resid")] = out.hidden_states[li + 1][0, -1, :].detach().float().cpu()
    margin, pred, la, lb = correct_margin(out.logits[0, -1, :], sample["label"])
    return captures, {"margin": margin, "pred": pred, "logit_a": la, "logit_b": lb}

print("BASELINE_CAPTURE_START", len(samples), flush=True)
for idx, s in enumerate(samples, 1):
    caps, met = capture_sample(s)
    key = (s["axis"], s["pair_index"], s["side"])
    cache[key] = caps
    row = dict(s)
    row.update(met)
    baseline_rows.append(row)
    if idx % 12 == 0:
        print("BASELINE_CAPTURE", idx, "/", len(samples), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

baseline_lookup = {(r["axis"], r["pair_index"], r["side"]): r for r in baseline_rows}

def summarize_rows(rows):
    out = {}
    axes = sorted(set(r["axis"] for r in rows))
    for axis in axes + ["__overall__"]:
        rr = rows if axis == "__overall__" else [r for r in rows if r["axis"] == axis]
        out[axis] = {
            "n": len(rr),
            "accuracy": sum(r["pred"] == r["label"] for r in rr) / len(rr),
            "mean_correct_margin": statistics.mean(r["margin"] for r in rr),
            "median_correct_margin": statistics.median(r["margin"] for r in rr),
        }
    return out

baseline_metrics = summarize_rows(baseline_rows)
print("BASELINE_METRICS", json.dumps(baseline_metrics), flush=True)

# Component/residual matched-pair separation.
component_sep = {}
for axis_name, cfg in AXES.items():
    component_sep[axis_name] = {}
    for li in CANDIDATE_LAYERS:
        component_sep[axis_name][str(li)] = {}
        for comp in ["attn", "mlp", "resid"]:
            vals = []
            for pi in range(len(cfg["pairs"])):
                a = cache[(axis_name, pi, "A")][(li, comp)]
                b = cache[(axis_name, pi, "B")][(li, comp)]
                cos = float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())
                rel = float(torch.linalg.vector_norm(a - b).item() / (((torch.linalg.vector_norm(a) + torch.linalg.vector_norm(b)) / 2).item() + 1e-12))
                vals.append({"pair_index": pi, "one_minus_cos": 1.0 - cos, "relative_l2": rel})
            component_sep[axis_name][str(li)][comp] = {
                "mean_one_minus_cos": statistics.mean(v["one_minus_cos"] for v in vals),
                "mean_relative_l2": statistics.mean(v["relative_l2"] for v in vals),
                "pairs": vals,
            }

# Final-token component ablation.
ablation_rows = []
for li in CANDIDATE_LAYERS:
    for comp in ["attn", "mlp"]:
        module = layers[li].self_attn if comp == "attn" else layers[li].mlp
        def ablate_hook(module, inp, out):
            return replace_final(out, zero=True)
        handle = module.register_forward_hook(ablate_hook)
        print("ABLATION_START", li, comp, flush=True)
        for s in samples:
            x = encode_prompt(s["axis"], s["text"])
            with torch.inference_mode():
                out = model(**x, use_cache=False, return_dict=True)
            margin, pred, la, lb = correct_margin(out.logits[0, -1, :], s["label"])
            base = baseline_lookup[(s["axis"], s["pair_index"], s["side"])]
            ablation_rows.append({
                **s,
                "layer": li,
                "component": comp,
                "margin": margin,
                "pred": pred,
                "baseline_margin": base["margin"],
                "baseline_pred": base["pred"],
                "margin_change": margin - base["margin"],
            })
        handle.remove()
        gc.collect()
        torch.cuda.empty_cache()
        print("ABLATION_DONE", li, comp, flush=True)

ablation_summary = {}
for li in CANDIDATE_LAYERS:
    ablation_summary[str(li)] = {}
    for comp in ["attn", "mlp"]:
        rows = [r for r in ablation_rows if r["layer"] == li and r["component"] == comp]
        metric = summarize_rows(rows)
        for axis, vals in metric.items():
            base = baseline_metrics[axis]
            vals["accuracy_delta_vs_baseline"] = vals["accuracy"] - base["accuracy"]
            vals["mean_margin_delta_vs_baseline"] = vals["mean_correct_margin"] - base["mean_correct_margin"]
        ablation_summary[str(li)][comp] = metric

# Matched activation patching: replace only final-token attention/MLP contribution
# with the corresponding vector from the opposite semantic member of the pair.
patch_rows = []
for li in PATCH_LAYERS:
    for comp in ["attn", "mlp"]:
        module = layers[li].self_attn if comp == "attn" else layers[li].mlp
        print("PATCH_START", li, comp, flush=True)
        for axis_name, cfg in AXES.items():
            for pi, _pair in enumerate(cfg["pairs"]):
                for target_side, source_side in [("A", "B"), ("B", "A")]:
                    target = next(s for s in samples if s["axis"] == axis_name and s["pair_index"] == pi and s["side"] == target_side)
                    source_vec = cache[(axis_name, pi, source_side)][(li, comp)]
                    def patch_hook(module, inp, out, v=source_vec):
                        return replace_final(out, vec=v, zero=False)
                    handle = module.register_forward_hook(patch_hook)
                    x = encode_prompt(target["axis"], target["text"])
                    with torch.inference_mode():
                        out = model(**x, use_cache=False, return_dict=True)
                    handle.remove()
                    margin, pred, la, lb = correct_margin(out.logits[0, -1, :], target["label"])
                    base = baseline_lookup[(axis_name, pi, target_side)]
                    patch_rows.append({
                        **target,
                        "source_side": source_side,
                        "layer": li,
                        "component": comp,
                        "patched_margin": margin,
                        "patched_pred": pred,
                        "baseline_margin": base["margin"],
                        "margin_drop": base["margin"] - margin,
                        "flipped_from_baseline": pred != base["pred"],
                    })
        gc.collect()
        torch.cuda.empty_cache()
        print("PATCH_DONE", li, comp, flush=True)

patch_summary = {}
for li in PATCH_LAYERS:
    patch_summary[str(li)] = {}
    for comp in ["attn", "mlp"]:
        patch_summary[str(li)][comp] = {}
        rows_lc = [r for r in patch_rows if r["layer"] == li and r["component"] == comp]
        for axis in list(AXES.keys()) + ["__overall__"]:
            rr = rows_lc if axis == "__overall__" else [r for r in rows_lc if r["axis"] == axis]
            patch_summary[str(li)][comp][axis] = {
                "n": len(rr),
                "mean_margin_drop": statistics.mean(r["margin_drop"] for r in rr),
                "median_margin_drop": statistics.median(r["margin_drop"] for r in rr),
                "flip_rate_from_baseline": sum(r["flipped_from_baseline"] for r in rr) / len(rr),
            }

# Save selected vectors for later frozen probes.
npz = {}
for key, caps in cache.items():
    axis, pi, side = key
    for (li, comp), vec in caps.items():
        npz[f"{axis}__p{pi}__{side}__l{li}__{comp}"] = vec.numpy().astype(np.float32)
np.savez_compressed(OUT / "candidate_component_vectors.npz", **npz)

fingerprint_after = fingerprint(model)
summary = {
    "experiment": "EMW_QWEN_CAUSAL_COMPONENT_SCAN_V0.2",
    "model_id": MODEL_ID,
    "revision": REVISION,
    "parameter_count": param_count,
    "candidate_layers": CANDIDATE_LAYERS,
    "patch_layers": PATCH_LAYERS,
    "axes": list(AXES.keys()),
    "pairs_per_axis": 6,
    "total_statements": len(samples),
    "intervention": "final-token component only",
    "training_performed": False,
    "optimizer_created": False,
    "backward_called": False,
    "adapter_loaded": False,
    "lora_loaded": False,
    "trainable_parameter_count": trainable_count,
    "weights_modified_by_experiment": fingerprint_before != fingerprint_after,
    "baseline_metrics": baseline_metrics,
}

(OUT / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
(OUT / "baseline_rows.json").write_text(json.dumps(baseline_rows, indent=2))
(OUT / "component_separation.json").write_text(json.dumps(component_sep, indent=2))
(OUT / "ablation_summary.json").write_text(json.dumps(ablation_summary, indent=2))
(OUT / "ablation_rows.json").write_text(json.dumps(ablation_rows, indent=2))
(OUT / "patch_summary.json").write_text(json.dumps(patch_summary, indent=2))
(OUT / "patch_rows.json").write_text(json.dumps(patch_rows, indent=2))

print("WEIGHTS_MODIFIED", fingerprint_before != fingerprint_after, flush=True)
print("PHASE2_SUMMARY_JSON", json.dumps(summary), flush=True)
print("PHASE2_COMPLETE", flush=True)
