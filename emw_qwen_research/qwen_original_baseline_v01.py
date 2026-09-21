import json, time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUT = Path("/kaggle/working/emw_qwen_original_baseline_v01")
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(17)

print("BASELINE_START", flush=True)
print("CUDA", torch.cuda.is_available(), torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    print("GPU", i, torch.cuda.get_device_name(i), flush=True)

revision = HfApi().model_info(MODEL_ID).sha
print("HF_REVISION", revision, flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=revision,
    trust_remote_code=False,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

input_device = model.get_input_embeddings().weight.device
param_count = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("PARAM_COUNT", param_count, flush=True)
print("TRAINABLE_COUNT", trainable, flush=True)
print("DEVICE_MAP", json.dumps({k: str(v) for k, v in model.hf_device_map.items()}, sort_keys=True), flush=True)

def fingerprint(m):
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.13.mlp.up_proj.weight",
        "model.layers.27.self_attn.o_proj.weight",
        "lm_head.weight",
    ]
    ps = dict(m.named_parameters())
    out = {}
    for name in names:
        p = ps[name].detach()
        f = p.reshape(-1)
        idx = [0, f.numel() // 5, f.numel() // 2, 4 * f.numel() // 5, f.numel() - 1]
        out[name] = {
            "shape": list(p.shape),
            "dtype": str(p.dtype),
            "samples": [float(f[j].float().cpu()) for j in idx],
        }
    return out

before = fingerprint(model)

pairs = [
    (
        "depression_positive_vs_negative",
        "The patient reports feeling depressed most days.",
        "The patient denies feeling depressed.",
    ),
    (
        "psychosis_positive_vs_negative",
        "The patient reports hearing voices when alone.",
        "The patient denies hearing voices when alone.",
    ),
    (
        "mania_decreased_need_vs_sleep_loss",
        "For four nights I slept only 3 hours but felt unusually energetic and did not feel tired.",
        "For four nights I slept only 3 hours and felt exhausted during the day.",
    ),
    (
        "dose_certain_vs_uncertain",
        "I take sertraline 50 mg every morning.",
        "I think I take sertraline 50 mg, maybe 100 mg; I am not sure.",
    ),
    (
        "collateral_conflict_vs_agreement",
        "The patient says she sleeps well. Her mother says she wakes repeatedly every night.",
        "The patient says she sleeps well. Her mother also says she sleeps well every night.",
    ),
    (
        "assessed_negative_vs_not_assessed",
        "The patient explicitly denies panic attacks.",
        "Panic attacks were not assessed during this visit.",
    ),
]

system = (
    "Return compact JSON with exactly these keys: facts, negated, uncertain, "
    "not_assessed, source_conflicts. Use only explicit information. "
    "Do not diagnose or recommend treatment."
)

def encode(text):
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    x = tokenizer(rendered, return_tensors="pt")
    return {k: v.to(input_device) for k, v in x.items()}

def hidden(text):
    x = encode(text)
    with torch.inference_mode():
        o = model(**x, output_hidden_states=True, use_cache=False, return_dict=True)
    return torch.stack([h[0, -1, :].detach().float().cpu() for h in o.hidden_states])

def generate(text):
    x = encode(text)
    with torch.inference_mode():
        y = model.generate(
            **x,
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(y[0, x["input_ids"].shape[1] :], skip_special_tokens=True)

records = []
vectors = {}
for n, (pid, a, b) in enumerate(pairs, 1):
    t0 = time.time()
    print(f"PAIR_START {n}/{len(pairs)} {pid}", flush=True)
    ha, hb = hidden(a), hidden(b)
    vectors[pid + "_a"] = ha.numpy().astype(np.float32)
    vectors[pid + "_b"] = hb.numpy().astype(np.float32)

    metrics = []
    for i, (va, vb) in enumerate(zip(ha, hb)):
        metrics.append(
            {
                "stage": "embedding" if i == 0 else f"layer_{i-1}",
                "cosine_similarity": float(
                    torch.nn.functional.cosine_similarity(va[None], vb[None]).item()
                ),
                "l2_delta": float(torch.linalg.vector_norm(va - vb).item()),
                "norm_a": float(torch.linalg.vector_norm(va).item()),
                "norm_b": float(torch.linalg.vector_norm(vb).item()),
            }
        )

    rec = {
        "pair_id": pid,
        "a": a,
        "b": b,
        "output_a": generate(a),
        "output_b": generate(b),
        "layer_metrics": metrics,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    records.append(rec)
    print("PAIR_DONE", pid, rec["elapsed_sec"], flush=True)

after = fingerprint(model)
weights_modified = before != after

np.savez_compressed(OUT / "hidden_last_token_vectors.npz", **vectors)
(OUT / "pair_results.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))

rank = {}
for rec in records:
    rank[rec["pair_id"]] = {
        "largest_l2_delta": sorted(
            rec["layer_metrics"], key=lambda z: z["l2_delta"], reverse=True
        )[:10],
        "lowest_cosine_similarity": sorted(
            rec["layer_metrics"], key=lambda z: z["cosine_similarity"]
        )[:10],
    }
(OUT / "top_layer_contrasts.json").write_text(json.dumps(rank, indent=2, ensure_ascii=False))

summary = {
    "experiment": "EMW_QWEN_ORIGINAL_BRAIN_BASELINE_V0.1",
    "model_id": MODEL_ID,
    "hf_revision": revision,
    "parameter_count": param_count,
    "runtime_dtype": "float16",
    "gpu_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "training_performed": False,
    "optimizer_created": False,
    "backward_called": False,
    "adapter_loaded": False,
    "lora_loaded": False,
    "trainable_parameter_count": trainable,
    "weights_modified_by_experiment": weights_modified,
    "pair_count": len(records),
    "hidden_state_stages_per_prompt": len(records[0]["layer_metrics"]),
    "fingerprint_before": before,
    "fingerprint_after": after,
}
(OUT / "baseline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print("WEIGHTS_MODIFIED", weights_modified, flush=True)
print("RESULT_DIR", str(OUT), flush=True)
print("BASELINE_SUMMARY_JSON", json.dumps(summary), flush=True)
print("BASELINE_COMPLETE", flush=True)
