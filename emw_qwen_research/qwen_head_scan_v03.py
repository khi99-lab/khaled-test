import gc
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
SCAN_LAYERS = [18, 19]
N_HEADS = 28
HEAD_DIM = 128
HIDDEN = N_HEADS * HEAD_DIM
BATCH_SIZE = 8
OUT = Path("/kaggle/working/emw_qwen_head_scan_v03")
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(17)

DEFS = {
    "present_vs_denied": (
        "A = the condition is explicitly present or reported.",
        "B = the condition is explicitly denied or absent.",
    ),
    "assessed_negative_vs_not_assessed": (
        "A = the item was explicitly assessed or checked and found absent.",
        "B = the item was explicitly not assessed or not checked.",
    ),
    "certain_vs_uncertain": (
        "A = one exact value is explicitly stated as certain.",
        "B = the value is explicitly uncertain or ambiguous.",
    ),
    "conflict_vs_agreement": (
        "A = the two sources explicitly disagree about the same issue.",
        "B = the two sources explicitly agree about the same issue.",
    ),
    "reduced_input_high_vs_low_output": (
        "A = a markedly reduced input/resource is paired with preserved or increased output/energy.",
        "B = a markedly reduced input/resource is paired with reduced output/energy or fatigue.",
    ),
}

CLINICAL = {
    "present_vs_denied": [
        ("dep", "The patient reports feeling depressed most days.", "The patient denies feeling depressed."),
        ("voices", "The patient reports hearing voices when alone.", "The patient denies hearing voices when alone."),
        ("panic", "The patient reports having panic attacks twice this week.", "The patient denies having panic attacks."),
        ("checking", "The patient reports repeatedly checking the door lock.", "The patient denies repeatedly checking the door lock."),
    ],
    "assessed_negative_vs_not_assessed": [
        ("panic", "The patient explicitly denies panic attacks.", "Panic attacks were not assessed during this visit."),
        ("voices", "The patient explicitly denies hearing voices.", "Auditory hallucinations were not assessed during this visit."),
        ("appetite", "The patient explicitly denies appetite changes.", "Appetite was not assessed during this visit."),
        ("compulsions", "The patient explicitly denies compulsive behaviors.", "Compulsive behaviors were not assessed during this visit."),
    ],
    "certain_vs_uncertain": [
        ("sertraline", "I take sertraline 50 mg every morning.", "I take sertraline 50 mg, maybe 100 mg; I am not sure."),
        ("quetiapine", "I take quetiapine 100 mg at bedtime.", "I think my quetiapine is 100 mg or 200 mg at bedtime; I am unsure."),
        ("lamotrigine", "I take lamotrigine 150 mg each evening.", "I believe I take lamotrigine 100 mg or 150 mg each evening; I am not certain."),
        ("aripiprazole", "I take aripiprazole 10 mg every day.", "I am not sure whether my aripiprazole is 5 mg or 10 mg every day."),
    ],
    "conflict_vs_agreement": [
        ("sleep", "The patient says she sleeps well. Her mother says she wakes repeatedly every night.", "The patient says she sleeps well. Her mother also says she sleeps well every night."),
        ("adherence", "The patient says he takes every dose. His father says he misses medication several times a week.", "The patient says he takes every dose. His father also says he takes every dose."),
        ("mood", "The patient says his mood has been good. His partner says he has seemed persistently sad.", "The patient says his mood has been good. His partner also says his mood has been good."),
        ("panic", "The patient says she has had no panic episodes. Her sister says she witnessed two panic episodes this week.", "The patient says she has had no panic episodes. Her sister also says she has seen no panic episodes."),
    ],
    "reduced_input_high_vs_low_output": [
        ("s1", "For four nights I slept only 3 hours but felt unusually energetic and did not feel tired.", "For four nights I slept only 3 hours and felt exhausted during the day."),
        ("s2", "I have slept about 2 hours nightly for three days and still feel full of energy.", "I have slept about 2 hours nightly for three days and feel drained and sleepy."),
        ("s3", "Despite sleeping only 4 hours, I wake up energized and do not need more sleep.", "After sleeping only 4 hours, I wake up fatigued and wish I could sleep longer."),
        ("s4", "I barely slept this week, yet I feel unusually active and never tired.", "I barely slept this week, and I feel tired and slowed down."),
    ],
}

CONTROL = {
    "present_vs_denied": [
        ("lamp", "The lamp is on.", "The lamp is not on."),
        ("package", "The package arrived this morning.", "The package did not arrive this morning."),
        ("alarm", "The alarm is active.", "The alarm is not active."),
        ("printer", "The printer is connected.", "The printer is not connected."),
    ],
    "assessed_negative_vs_not_assessed": [
        ("crack", "The inspector checked for cracks and found none.", "Cracks were not checked during the inspection."),
        ("leak", "The technician checked for leaks and found none.", "Leaks were not checked during the inspection."),
        ("virus", "The file was scanned for malware and none was found.", "The file was not scanned for malware."),
        ("damage", "The package was checked for damage and none was found.", "Package damage was not assessed."),
    ],
    "certain_vs_uncertain": [
        ("flour", "The recipe uses exactly 500 grams of flour.", "The recipe uses 500 or 600 grams of flour; I am not sure."),
        ("file", "The file size is exactly 20 megabytes.", "The file size is 20 or 25 megabytes; I am not sure."),
        ("train", "The train leaves at exactly 8:30 AM.", "The train leaves at 8:30 or 9:00 AM; I am not sure."),
        ("price", "The price is exactly 40 dollars.", "The price is 40 or 50 dollars; I am not certain."),
    ],
    "conflict_vs_agreement": [
        ("door", "Alex says the door is locked. Jordan says the door is unlocked.", "Alex says the door is locked. Jordan also says the door is locked."),
        ("delivery", "The driver says the package was delivered. The customer says it was not delivered.", "The driver says the package was delivered. The customer also says it was delivered."),
        ("machine", "The operator says the machine is working. The supervisor says it is broken.", "The operator says the machine is working. The supervisor also says it is working."),
        ("invoice", "The vendor says the invoice was paid. Accounting says it remains unpaid.", "The vendor says the invoice was paid. Accounting also says it was paid."),
    ],
    "reduced_input_high_vs_low_output": [
        ("battery", "The device ran on 10% battery but maintained full performance.", "The device ran on 10% battery and became very slow."),
        ("fuel", "The generator used very little fuel yet kept producing full power.", "The generator used very little fuel and produced much less power."),
        ("memory", "The program used half its usual memory but ran faster than normal.", "The program used half its usual memory and ran much slower."),
        ("power", "The motor received less power but kept running at full speed.", "The motor received less power and slowed substantially."),
    ],
}

print("HEAD_SCAN_START", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
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
input_device = model.get_input_embeddings().weight.device
assert model.config.hidden_size == HIDDEN
assert model.config.num_attention_heads == N_HEADS
print("ARCH", model.config.hidden_size, model.config.num_attention_heads, HEAD_DIM, flush=True)

def choose_answer_tokens():
    for aa, bb in [(" A", " B"), ("A", "B"), (" 0", " 1"), ("0", "1")]:
        ia = tokenizer.encode(aa, add_special_tokens=False)
        ib = tokenizer.encode(bb, add_special_tokens=False)
        if len(ia) == 1 and len(ib) == 1 and ia[0] != ib[0]:
            return ia[0], ib[0]
    raise RuntimeError("no answer tokens")
TA, TB = choose_answer_tokens()

def make_prompt(axis, text):
    adef, bdef = DEFS[axis]
    return (
        "Classify using exactly one label.\n"
        + adef + "\n" + bdef + "\n"
        + "Statement: " + text + "\nAnswer:"
    )

samples = []
for domain, dataset in [("clinical", CLINICAL), ("control", CONTROL)]:
    for axis, pairs in dataset.items():
        for pi, (pid, a, b) in enumerate(pairs):
            samples.append({"domain":domain,"axis":axis,"pair_index":pi,"pair_id":pid,"side":"A","label":"A","text":a})
            samples.append({"domain":domain,"axis":axis,"pair_index":pi,"pair_id":pid,"side":"B","label":"B","text":b})

def batches(items, n=BATCH_SIZE):
    for i in range(0, len(items), n):
        yield items[i:i+n]

def encode_batch(batch):
    rendered = [
        tokenizer.apply_chat_template(
            [{"role":"user","content":make_prompt(s["axis"],s["text"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for s in batch
    ]
    x = tokenizer(rendered, return_tensors="pt", padding=True)
    return {k:v.to(input_device) for k,v in x.items()}

def batch_metrics(logits, batch):
    rows=[]
    la=logits[:, TA].float().cpu()
    lb=logits[:, TB].float().cpu()
    for j,s in enumerate(batch):
        a=float(la[j]); b=float(lb[j])
        pred="A" if a>=b else "B"
        margin=(a-b) if s["label"]=="A" else (b-a)
        rows.append((margin,pred))
    return rows

def fingerprint():
    ps=dict(model.named_parameters())
    names=["model.layers.18.self_attn.o_proj.weight","model.layers.19.self_attn.o_proj.weight","lm_head.weight"]
    out={}
    for n in names:
        f=ps[n].detach().reshape(-1)
        idx=[0,f.numel()//3,2*f.numel()//3,f.numel()-1]
        out[n]=[float(f[i].float().cpu()) for i in idx]
    return out

fp_before=fingerprint()

# Baseline + capture concatenated head outputs immediately before o_proj.
baseline=[]
head_cache={}
for batch in batches(samples):
    captured={}
    handles=[]
    for li in SCAN_LAYERS:
        def mk(layer_idx):
            def hook(module, args):
                x=args[0]
                assert x.ndim==3 and x.shape[-1]==HIDDEN, x.shape
                captured[layer_idx]=x[:,-1,:].detach().float().cpu()
            return hook
        handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(mk(li)))
    x=encode_batch(batch)
    with torch.inference_mode():
        out=model(**x,use_cache=False,return_dict=True)
    for h in handles: h.remove()
    mets=batch_metrics(out.logits[:,-1,:],batch)
    for j,s in enumerate(batch):
        key=(s["domain"],s["axis"],s["pair_index"],s["side"])
        for li in SCAN_LAYERS:
            head_cache[(key,li)]=captured[li][j].clone()
        baseline.append({**s,"margin":mets[j][0],"pred":mets[j][1]})

lookup={(r["domain"],r["axis"],r["pair_index"],r["side"]):r for r in baseline}

def aggregate(rows, margin_field="margin", pred_field="pred"):
    out={}
    for domain in ["clinical","control"]:
        rr=[r for r in rows if r["domain"]==domain]
        out[domain]={
            "n":len(rr),
            "accuracy":sum(r[pred_field]==r["label"] for r in rr)/len(rr),
            "mean_margin":statistics.mean(r[margin_field] for r in rr),
        }
        for axis in DEFS:
            aa=[r for r in rr if r["axis"]==axis]
            out[domain+"::"+axis]={
                "n":len(aa),
                "accuracy":sum(r[pred_field]==r["label"] for r in aa)/len(aa),
                "mean_margin":statistics.mean(r[margin_field] for r in aa),
            }
    return out

baseline_summary=aggregate(baseline)
print("HEAD_BASELINE",json.dumps(baseline_summary),flush=True)

# Per-head matched-pair representational separation.
head_sep={}
for li in SCAN_LAYERS:
    head_sep[str(li)]={}
    for h in range(N_HEADS):
        s0=h*HEAD_DIM; s1=(h+1)*HEAD_DIM
        head_sep[str(li)][str(h)]={}
        for domain,dataset in [("clinical",CLINICAL),("control",CONTROL)]:
            vals=[]
            axis_vals={axis:[] for axis in DEFS}
            for axis,pairs in dataset.items():
                for pi in range(len(pairs)):
                    a=head_cache[((domain,axis,pi,"A"),li)][s0:s1]
                    b=head_cache[((domain,axis,pi,"B"),li)][s0:s1]
                    cos=float(torch.nn.functional.cosine_similarity(a[None],b[None]).item())
                    rel=float(torch.linalg.vector_norm(a-b).item()/((((torch.linalg.vector_norm(a)+torch.linalg.vector_norm(b))/2).item())+1e-12))
                    vals.append((1-cos,rel)); axis_vals[axis].append((1-cos,rel))
            d={"mean_one_minus_cos":statistics.mean(v[0] for v in vals),
               "mean_relative_l2":statistics.mean(v[1] for v in vals)}
            for axis,vv in axis_vals.items():
                d[axis]={"mean_one_minus_cos":statistics.mean(v[0] for v in vv),
                         "mean_relative_l2":statistics.mean(v[1] for v in vv)}
            head_sep[str(li)][str(h)][domain]=d

# Head ablation before o_proj: zero one 128-d head slot at final token only.
abl_rows=[]
for li in SCAN_LAYERS:
    op=layers[li].self_attn.o_proj
    for hidx in range(N_HEADS):
        s0=hidx*HEAD_DIM; s1=(hidx+1)*HEAD_DIM
        def hook(module,args,a=s0,b=s1):
            x=args[0].clone()
            x[:,-1,a:b]=0
            return (x,)+tuple(args[1:])
        handle=op.register_forward_pre_hook(hook)
        for batch in batches(samples):
            x=encode_batch(batch)
            with torch.inference_mode():
                out=model(**x,use_cache=False,return_dict=True)
            mets=batch_metrics(out.logits[:,-1,:],batch)
            for j,s in enumerate(batch):
                base=lookup[(s["domain"],s["axis"],s["pair_index"],s["side"])]
                abl_rows.append({**s,"layer":li,"head":hidx,"margin":mets[j][0],"pred":mets[j][1],
                                 "margin_change":mets[j][0]-base["margin"]})
        handle.remove()
        if hidx%7==6: print("HEAD_ABL",li,hidx,flush=True)
    gc.collect(); torch.cuda.empty_cache()

head_abl={}
rank_records=[]
for li in SCAN_LAYERS:
    head_abl[str(li)]={}
    for h in range(N_HEADS):
        rr=[r for r in abl_rows if r["layer"]==li and r["head"]==h]
        d={}
        for domain in ["clinical","control"]:
            dd=[r for r in rr if r["domain"]==domain]
            loss=-statistics.mean(r["margin_change"] for r in dd)
            acc=sum(r["pred"]==r["label"] for r in dd)/len(dd)
            d[domain]={"mean_support_loss":loss,"accuracy":acc,
                       "accuracy_delta":acc-baseline_summary[domain]["accuracy"]}
            for axis in DEFS:
                aa=[r for r in dd if r["axis"]==axis]
                d[domain][axis]={"mean_support_loss":-statistics.mean(r["margin_change"] for r in aa),
                                 "accuracy":sum(r["pred"]==r["label"] for r in aa)/len(aa)}
        diff=d["clinical"]["mean_support_loss"]-d["control"]["mean_support_loss"]
        sep_diff=head_sep[str(li)][str(h)]["clinical"]["mean_one_minus_cos"]-head_sep[str(li)][str(h)]["control"]["mean_one_minus_cos"]
        d["clinical_minus_control_support_loss"]=diff
        d["clinical_minus_control_separation"]=sep_diff
        head_abl[str(li)][str(h)]=d
        rank_records.append({"layer":li,"head":h,"clinical_loss":d["clinical"]["mean_support_loss"],
                             "control_loss":d["control"]["mean_support_loss"],"diff_loss":diff,
                             "clinical_sep":head_sep[str(li)][str(h)]["clinical"]["mean_one_minus_cos"],
                             "control_sep":head_sep[str(li)][str(h)]["control"]["mean_one_minus_cos"],
                             "diff_sep":sep_diff})

# Candidate heads = union of top differential support, clinical support, differential separation.
candidates=set()
for key in ["diff_loss","clinical_loss","diff_sep"]:
    for r in sorted(rank_records,key=lambda z:z[key],reverse=True)[:5]:
        candidates.add((r["layer"],r["head"]))
candidates=sorted(candidates)
print("HEAD_CANDIDATES",candidates,flush=True)

# Opposite-pair per-head activation patching for candidates.
patch_rows=[]
for li,hidx in candidates:
    op=layers[li].self_attn.o_proj
    s0=hidx*HEAD_DIM; s1=(hidx+1)*HEAD_DIM
    patch_targets=[]
    for s in samples:
        other="B" if s["side"]=="A" else "A"
        source=head_cache[((s["domain"],s["axis"],s["pair_index"],other),li)][s0:s1]
        patch_targets.append((s,source))
    for i in range(0,len(patch_targets),BATCH_SIZE):
        chunk=patch_targets[i:i+BATCH_SIZE]
        batch=[x[0] for x in chunk]
        vec=torch.stack([x[1] for x in chunk],dim=0)
        def ph(module,args,v=vec,a=s0,b=s1):
            x=args[0].clone()
            x[:,-1,a:b]=v.to(device=x.device,dtype=x.dtype)
            return (x,)+tuple(args[1:])
        handle=op.register_forward_pre_hook(ph)
        xb=encode_batch(batch)
        with torch.inference_mode():
            out=model(**xb,use_cache=False,return_dict=True)
        handle.remove()
        mets=batch_metrics(out.logits[:,-1,:],batch)
        for j,s in enumerate(batch):
            base=lookup[(s["domain"],s["axis"],s["pair_index"],s["side"])]
            patch_rows.append({**s,"layer":li,"head":hidx,"patched_margin":mets[j][0],"patched_pred":mets[j][1],
                               "margin_drop":base["margin"]-mets[j][0],
                               "flipped":mets[j][1]!=base["pred"]})
    print("HEAD_PATCH",li,hidx,flush=True)

head_patch={}
for li,h in candidates:
    rr=[r for r in patch_rows if r["layer"]==li and r["head"]==h]
    d={}
    for domain in ["clinical","control"]:
        dd=[r for r in rr if r["domain"]==domain]
        d[domain]={"mean_margin_drop":statistics.mean(r["margin_drop"] for r in dd),
                   "flip_rate":sum(r["flipped"] for r in dd)/len(dd)}
        for axis in DEFS:
            aa=[r for r in dd if r["axis"]==axis]
            d[domain][axis]={"mean_margin_drop":statistics.mean(r["margin_drop"] for r in aa),
                             "flip_rate":sum(r["flipped"] for r in aa)/len(aa)}
    d["clinical_minus_control_margin_drop"]=d["clinical"]["mean_margin_drop"]-d["control"]["mean_margin_drop"]
    head_patch[f"{li}:{h}"]=d

# Save raw head vectors.
npz={}
for (key,li),vec in head_cache.items():
    domain,axis,pi,side=key
    npz[f"{domain}__{axis}__p{pi}__{side}__l{li}"]=vec.numpy().astype(np.float32)
np.savez_compressed(OUT/"o_proj_head_vectors.npz",**npz)

fp_after=fingerprint()
summary={
    "experiment":"EMW_QWEN_HEAD_SCAN_V0.3",
    "model_id":MODEL_ID,"revision":REVISION,
    "layers":SCAN_LAYERS,"n_heads":N_HEADS,"head_dim":HEAD_DIM,
    "clinical_statements":sum(s["domain"]=="clinical" for s in samples),
    "control_statements":sum(s["domain"]=="control" for s in samples),
    "baseline":baseline_summary,
    "candidate_heads":[{"layer":a,"head":b} for a,b in candidates],
    "training_performed":False,"optimizer_created":False,"backward_called":False,
    "trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),
    "weights_modified_by_experiment":fp_before!=fp_after,
}
(OUT/"head_scan_summary.json").write_text(json.dumps(summary,indent=2))
(OUT/"head_separation.json").write_text(json.dumps(head_sep,indent=2))
(OUT/"head_ablation_summary.json").write_text(json.dumps(head_abl,indent=2))
(OUT/"head_ablation_rows.json").write_text(json.dumps(abl_rows,indent=2))
(OUT/"head_patch_summary.json").write_text(json.dumps(head_patch,indent=2))
(OUT/"head_patch_rows.json").write_text(json.dumps(patch_rows,indent=2))
(OUT/"head_rank_records.json").write_text(json.dumps(rank_records,indent=2))
print("HEAD_SCAN_SUMMARY_JSON",json.dumps(summary),flush=True)
print("HEAD_SCAN_COMPLETE",flush=True)
