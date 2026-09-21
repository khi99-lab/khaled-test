import gc
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
LAYERS = [18, 19, 20]
OUT = Path("/kaggle/working/emw_qwen_nonclinical_control_v02b")
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(17)

AXES = {
    "explicit_positive_vs_denial": {
        "a": "A = the stated condition is explicitly present or true.",
        "b": "B = the stated condition is explicitly denied or false.",
        "pairs": [
            ("lamp", "The lamp is on.", "The lamp is not on."),
            ("door", "The front door is locked.", "The front door is not locked."),
            ("package", "The package arrived this morning.", "The package did not arrive this morning."),
            ("server", "The server is currently online.", "The server is currently offline."),
            ("alarm", "The alarm is active.", "The alarm is not active."),
            ("printer", "The printer is connected.", "The printer is not connected."),
        ],
    },
    "denied_vs_not_assessed": {
        "a": "A = the property was explicitly checked and found absent.",
        "b": "B = the property was explicitly not checked or assessed.",
        "pairs": [
            ("crack", "The inspector checked for cracks and found none.", "Cracks were not checked during the inspection."),
            ("leak", "The technician checked for leaks and found none.", "Leaks were not checked during the inspection."),
            ("virus", "The file was scanned for malware and none was found.", "The file was not scanned for malware."),
            ("damage", "The package was checked for damage and none was found.", "Package damage was not assessed."),
            ("error", "The log was checked for errors and none were found.", "The log was not checked for errors."),
            ("rust", "The mechanic checked for rust and found none.", "Rust was not assessed during the inspection."),
        ],
    },
    "quantity_certain_vs_uncertain": {
        "a": "A = one exact quantity is explicitly stated as certain.",
        "b": "B = the quantity is explicitly uncertain or ambiguous.",
        "pairs": [
            ("flour", "The recipe uses exactly 500 grams of flour.", "The recipe uses 500 or 600 grams of flour; I am not sure."),
            ("file", "The file size is exactly 20 megabytes.", "The file size is 20 or 25 megabytes; I am not sure."),
            ("train", "The train leaves at exactly 8:30 AM.", "The train leaves at 8:30 or 9:00 AM; I am not sure."),
            ("box", "The box weighs exactly 10 kilograms.", "The box weighs 10 or 12 kilograms; I am unsure."),
            ("cable", "The cable is exactly 2 meters long.", "The cable is 2 or 3 meters long; I cannot remember."),
            ("price", "The price is exactly 40 dollars.", "The price is 40 or 50 dollars; I am not certain."),
        ],
    },
    "source_conflict_vs_agreement": {
        "a": "A = the two sources explicitly disagree about the same issue.",
        "b": "B = the two sources explicitly agree about the same issue.",
        "pairs": [
            ("door", "Alex says the door is locked. Jordan says the door is unlocked.", "Alex says the door is locked. Jordan also says the door is locked."),
            ("delivery", "The driver says the package was delivered. The customer says it was not delivered.", "The driver says the package was delivered. The customer also says it was delivered."),
            ("machine", "The operator says the machine is working. The supervisor says it is broken.", "The operator says the machine is working. The supervisor also says it is working."),
            ("weather", "One sensor says it is raining. The second sensor says it is not raining.", "One sensor says it is raining. The second sensor also says it is raining."),
            ("invoice", "The vendor says the invoice was paid. Accounting says it remains unpaid.", "The vendor says the invoice was paid. Accounting also says it was paid."),
            ("window", "The tenant says the window is closed. The inspector says the window is open.", "The tenant says the window is closed. The inspector also says the window is closed."),
        ],
    },
    "reduced_resource_high_vs_low_output": {
        "a": "A = markedly reduced resource use with preserved or increased output/performance.",
        "b": "B = reduced resource use accompanied by reduced output/performance.",
        "pairs": [
            ("battery", "The device ran on 10% battery but maintained full performance.", "The device ran on 10% battery and became very slow."),
            ("fuel", "The generator used very little fuel yet kept producing full power.", "The generator used very little fuel and produced much less power."),
            ("memory", "The program used half its usual memory but ran faster than normal.", "The program used half its usual memory and ran much slower."),
            ("workers", "The team had half the usual staff but completed more work than normal.", "The team had half the usual staff and completed much less work."),
            ("bandwidth", "The service had very low bandwidth but remained unusually responsive.", "The service had very low bandwidth and became sluggish."),
            ("power", "The motor received less power but kept running at full speed.", "The motor received less power and slowed substantially."),
        ],
    },
    "explicit_self_correction_vs_no_correction": {
        "a": "A = the speaker explicitly corrects or replaces an earlier statement.",
        "b": "B = the speaker confirms an earlier statement without correcting it.",
        "pairs": [
            ("time", "Earlier I said 8 PM, but that was wrong; the meeting is at 5 PM.", "Earlier I said 5 PM, and that is correct."),
            ("price", "I said 100 dollars before, but I need to correct that: it is 50 dollars.", "I said 50 dollars before, and that is correct."),
            ("date", "I first said Tuesday, but actually the delivery is Wednesday.", "I first said Wednesday, and that is correct."),
            ("count", "I said there were ten boxes; correction, there are six.", "I said there are six boxes, and that is correct."),
            ("distance", "I told you it was 20 miles, but I need to correct that; it is 12 miles.", "I told you it is 12 miles, and that is correct."),
            ("version", "I said version 4 earlier; correction, the installed version is 3.", "I said version 3 earlier, and that is correct."),
        ],
    },
}

print("CONTROL_START", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=REVISION, trust_remote_code=False,
    torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
layers = model.model.layers
input_device = model.get_input_embeddings().weight.device

def tpart(out):
    if torch.is_tensor(out): return out
    if isinstance(out, (tuple,list)) and out and torch.is_tensor(out[0]): return out[0]
    raise TypeError(type(out))

def repl(out, vec=None, zero=False):
    t=tpart(out); y=t.clone()
    if zero: y[:,-1,:]=0
    else: y[:,-1,:]=vec.to(y.device,y.dtype)
    if torch.is_tensor(out): return y
    if isinstance(out,tuple): return (y,)+tuple(out[1:])
    return [y]+list(out[1:])

def answer_tokens():
    for a,b in [(" A"," B"),("A","B"),(" 0"," 1"),("0","1")]:
        ia=tokenizer.encode(a,add_special_tokens=False); ib=tokenizer.encode(b,add_special_tokens=False)
        if len(ia)==len(ib)==1 and ia[0]!=ib[0]: return ia[0],ib[0]
    raise RuntimeError
TA,TB=answer_tokens()

def prompt(axis,text):
    c=AXES[axis]
    return "Classify using exactly one label.\n"+c["a"]+"\n"+c["b"]+"\nStatement: "+text+"\nAnswer:"

def enc(axis,text):
    s=tokenizer.apply_chat_template([{"role":"user","content":prompt(axis,text)}],tokenize=False,add_generation_prompt=True)
    x=tokenizer(s,return_tensors="pt")
    return {k:v.to(input_device) for k,v in x.items()}

def metric(logits,label):
    a=float(logits[TA].float().cpu()); b=float(logits[TB].float().cpu())
    pred="A" if a>=b else "B"
    margin=(a-b) if label=="A" else (b-a)
    return margin,pred

samples=[]
for axis,c in AXES.items():
    for pi,(pid,a,b) in enumerate(c["pairs"]):
        samples += [
            {"axis":axis,"pair_index":pi,"pair_id":pid,"side":"A","label":"A","text":a},
            {"axis":axis,"pair_index":pi,"pair_id":pid,"side":"B","label":"B","text":b},
        ]

cache={}; baseline=[]
for idx,s in enumerate(samples,1):
    caps={}; hs=[]
    def mk(li,comp):
        def hook(m,inp,out):
            caps[(li,comp)]=tpart(out)[0,-1,:].detach().float().cpu()
        return hook
    for li in LAYERS:
        hs += [layers[li].self_attn.register_forward_hook(mk(li,"attn")),
               layers[li].mlp.register_forward_hook(mk(li,"mlp"))]
    x=enc(s["axis"],s["text"])
    with torch.inference_mode(): o=model(**x,use_cache=False,return_dict=True)
    for h in hs: h.remove()
    m,p=metric(o.logits[0,-1,:],s["label"])
    cache[(s["axis"],s["pair_index"],s["side"])]=caps
    baseline.append({**s,"margin":m,"pred":p})
    if idx%18==0: print("CONTROL_BASE",idx,"/",len(samples),flush=True)

lookup={(r["axis"],r["pair_index"],r["side"]):r for r in baseline}
def summary(rows):
    out={}
    for axis in list(AXES)+["__overall__"]:
        rr=rows if axis=="__overall__" else [r for r in rows if r["axis"]==axis]
        out[axis]={"n":len(rr),"accuracy":sum(r["pred"]==r["label"] for r in rr)/len(rr),
                   "mean_margin":statistics.mean(r["margin"] for r in rr)}
    return out
base_summary=summary(baseline)

sep={}
for axis,c in AXES.items():
    sep[axis]={}
    for li in LAYERS:
        sep[axis][str(li)]={}
        for comp in ["attn","mlp"]:
            vals=[]
            for pi in range(6):
                a=cache[(axis,pi,"A")][(li,comp)]; b=cache[(axis,pi,"B")][(li,comp)]
                cos=float(torch.nn.functional.cosine_similarity(a[None],b[None]).item())
                rel=float(torch.linalg.vector_norm(a-b).item()/((((torch.linalg.vector_norm(a)+torch.linalg.vector_norm(b))/2).item())+1e-12))
                vals.append((1-cos,rel))
            sep[axis][str(li)][comp]={"mean_one_minus_cos":statistics.mean(v[0] for v in vals),
                                      "mean_relative_l2":statistics.mean(v[1] for v in vals)}

abl=[]
for li in LAYERS:
    for comp in ["attn","mlp"]:
        mod=layers[li].self_attn if comp=="attn" else layers[li].mlp
        h=mod.register_forward_hook(lambda m,i,o: repl(o,zero=True))
        for s in samples:
            x=enc(s["axis"],s["text"])
            with torch.inference_mode(): o=model(**x,use_cache=False,return_dict=True)
            m,p=metric(o.logits[0,-1,:],s["label"]); base=lookup[(s["axis"],s["pair_index"],s["side"])]
            abl.append({**s,"layer":li,"component":comp,"margin":m,"pred":p,"margin_change":m-base["margin"]})
        h.remove(); gc.collect(); torch.cuda.empty_cache()
        print("CONTROL_ABL",li,comp,flush=True)

abl_summary={}
for li in LAYERS:
    abl_summary[str(li)]={}
    for comp in ["attn","mlp"]:
        rows=[r for r in abl if r["layer"]==li and r["component"]==comp]
        abl_summary[str(li)][comp]={}
        for axis in list(AXES)+["__overall__"]:
            rr=rows if axis=="__overall__" else [r for r in rows if r["axis"]==axis]
            bs=base_summary[axis]
            acc=sum(r["pred"]==r["label"] for r in rr)/len(rr)
            mm=statistics.mean(r["margin"] for r in rr)
            abl_summary[str(li)][comp][axis]={"accuracy":acc,"accuracy_delta_vs_baseline":acc-bs["accuracy"],
                                              "mean_margin":mm,"mean_margin_delta_vs_baseline":mm-bs["mean_margin"]}

patch=[]
for li in LAYERS:
    for comp in ["attn","mlp"]:
        mod=layers[li].self_attn if comp=="attn" else layers[li].mlp
        for axis in AXES:
            for pi in range(6):
                for target,source in [("A","B"),("B","A")]:
                    s=next(z for z in samples if z["axis"]==axis and z["pair_index"]==pi and z["side"]==target)
                    vec=cache[(axis,pi,source)][(li,comp)]
                    def ph(m,i,o,v=vec): return repl(o,vec=v)
                    h=mod.register_forward_hook(ph)
                    x=enc(s["axis"],s["text"])
                    with torch.inference_mode(): o=model(**x,use_cache=False,return_dict=True)
                    h.remove()
                    m,p=metric(o.logits[0,-1,:],s["label"]); base=lookup[(axis,pi,target)]
                    patch.append({**s,"source_side":source,"layer":li,"component":comp,
                                  "margin_drop":base["margin"]-m,"flipped":p!=base["pred"]})
        gc.collect(); torch.cuda.empty_cache()
        print("CONTROL_PATCH",li,comp,flush=True)

patch_summary={}
for li in LAYERS:
    patch_summary[str(li)]={}
    for comp in ["attn","mlp"]:
        patch_summary[str(li)][comp]={}
        rows=[r for r in patch if r["layer"]==li and r["component"]==comp]
        for axis in list(AXES)+["__overall__"]:
            rr=rows if axis=="__overall__" else [r for r in rows if r["axis"]==axis]
            patch_summary[str(li)][comp][axis]={"mean_margin_drop":statistics.mean(r["margin_drop"] for r in rr),
                                                "flip_rate":sum(r["flipped"] for r in rr)/len(rr)}

out_summary={"experiment":"EMW_QWEN_NONCLINICAL_CONTROL_V0.2B","model_id":MODEL_ID,"revision":REVISION,
             "layers":LAYERS,"total_statements":len(samples),"training_performed":False,
             "trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),
             "baseline_metrics":base_summary}
(OUT/"control_summary.json").write_text(json.dumps(out_summary,indent=2))
(OUT/"component_separation.json").write_text(json.dumps(sep,indent=2))
(OUT/"ablation_summary.json").write_text(json.dumps(abl_summary,indent=2))
(OUT/"patch_summary.json").write_text(json.dumps(patch_summary,indent=2))
print("CONTROL_SUMMARY_JSON",json.dumps(out_summary),flush=True)
print("CONTROL_COMPLETE",flush=True)
