"""EMW v0.6: English distributed-LoRA robustness confirmation.
Synthetic research only. Repeats the distributed v0.5 architecture across three seeds
and evaluates on new harder, context-rich English holdouts. No patient data, diagnosis,
prescribing, checkpoint merge, or production writes.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, re, subprocess, sys, time, traceback
from pathlib import Path
import qwen_lora_pilot_v05 as v5

VERSION = "EMW_QWEN_ENGLISH_DISTRIBUTED_LORA_CONFIRMATION_V0.6"
MODEL_ID = v5.MODEL_ID
REVISION = v5.REVISION
SEEDS = {"replicate_1706":1706, "replicate_2718":2718, "replicate_3141":3141}
ARMS = {k:{"layers":list(range(28)),"rank":4} for k in SEEDS}
EXPECTED_TRAINABLE = v5.EXPECTED_TRAINABLE
TARGETS = v5.TARGETS
STATES = v5.STATES
STEPS, ACCUM, LR, MAX_LENGTH = 100, 4, 1e-4, 512

DEV_TERMS = {
 "clinical":["fear of driving","difficulty concentrating in meetings","intrusive memories after an argument",
             "checking the front door repeatedly","avoiding phone calls","waking before the alarm"],
 "control":["rattling fan blades","loose cabinet latches","intermittent screen flicker",
            "sticking drawer rails","faded warning labels","uneven table legs"]
}
TEST_TERMS = {
 "clinical":["recurrent mental replay of conversations","fear of entering crowded stores",
             "difficulty enjoying hobbies","repeated reassurance seeking","nighttime awakening",
             "unwanted images of accidents","avoiding reminders of a stressful event",
             "difficulty starting routine tasks","feeling detached from surroundings",
             "persistent self-critical thoughts","restlessness while sitting","loss of interest in social plans"],
 "control":["intermittent valve clicking","misaligned door sensors","scratched protective coating",
            "loose mounting brackets","faint motor vibration","stiff control knobs",
            "missing cable clips","warped plastic covers","unreadable inventory stickers",
            "uneven shelf supports","dented access panels","faded calibration marks"]
}

INSTRUCTION = (
 "Extract only the explicit evidence about the named target. Return one JSON object with exactly "
 "status, source, timeframe, evidence. status must be present, absent, not_assessed, or uncertain. "
 "Use absent only for an explicit denial, never for a topic that was not evaluated. Use uncertain "
 "only when the focal speaker explicitly cannot establish presence or absence. source must be the "
 "author of the focal statement. timeframe must be current or historical as specified for the focal "
 "target. evidence must copy the entire focal statement exactly, without surrounding quotation marks. "
 "Ignore distractors, other targets, and other speakers. Do not diagnose or recommend treatment."
)

FAMILIES = [
 (
  "After discussing an unrelated topic, the record turns to the target.",
  ("The focal speaker clearly states that {t} is occurring.",
   "The focal speaker directly states that {t} is not occurring.",
   "The target {t} was scheduled for discussion, but it was never evaluated and no answer was obtained.",
   "The focal speaker says they cannot determine whether {t} is occurring.")
 ),
 (
  "A second speaker comments about a different issue; that comment is not evidence for this target.",
  ("After considering the target carefully, the focal speaker confirms {t}.",
   "After considering the target carefully, the focal speaker explicitly denies {t}.",
   "No question about {t} was asked, and the record explicitly leaves it unassessed.",
   "The focal speaker gives only an uncertain report about {t} and cannot confirm or deny it.")
 ),
 (
  "The note contains both historical background and a separate current distractor.",
  ("For the specified timeframe, the focal statement affirms {t}.",
   "For the specified timeframe, the focal statement rejects {t}.",
   "For the specified timeframe, {t} was not assessed at all.",
   "For the specified timeframe, the focal statement says the status of {t} remains uncertain.")
 ),
 (
  "The surrounding note includes a preliminary thought that must not override the final focal statement.",
  ("The focal speaker says: I was unsure at first, but to be clear, {t} is present.",
   "The focal speaker says: I wondered about it at first, but to be clear, {t} is absent.",
   "The focal speaker says: We intended to cover {t}, but we never assessed it and obtained no answer.",
   "The focal speaker says: I still cannot tell whether {t} is present or absent.")
 )
]

def write_json(path, value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
 tmp.replace(path)

def sha(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def patch_v5(seed):
 v5.VERSION=VERSION
 v5.SEED=seed
 v5.ARMS=ARMS
 v5.EXPECTED_TRAINABLE=EXPECTED_TRAINABLE
 v5.STEPS=STEPS
 v5.ACCUM=ACCUM
 v5.LR=LR
 v5.MAX_LENGTH=MAX_LENGTH
 v5.score=score

def hard_case(split,domain,idx,target,state):
 family=idx%len(FAMILIES)
 lead,variants=FAMILIES[family]
 source=(["patient","collateral","clinician","patient"][idx%4] if domain=="clinical" else "inspector")
 timeframe="current" if idx%2==0 else "historical"
 evidence=variants[state].format(t=target)
 other_source="collateral" if domain=="clinical" and source!="collateral" else ("clinician" if domain=="clinical" else "supervisor")
 distractor_target=("sleep schedule" if domain=="clinical" else "shipping carton")
 text=(
   f"Named target: {target}\n"
   f"Required timeframe for this target: {timeframe}.\n"
   f"Author of the focal statement: {source}.\n"
   f"{lead}\n"
   f"Distractor from {other_source}: The separate issue of {distractor_target} was mentioned, but it is not the named target.\n"
   f"Focal statement: \"{evidence}\"\n"
   "End of target-specific record."
 )
 expected={"status":STATES[state],"source":source,"timeframe":timeframe,"evidence":evidence}
 return {"id":f"{split}_{domain}_{idx:02d}_h{family}_s{state}","split":split,"domain":domain,
         "family":f"{split}_hard_{family}","group_id":f"{split}_{domain}_{idx:02d}_h{family}",
         "target":target,"instruction":INSTRUCTION,"input":text,"expected":expected,
         "assistant":json.dumps(expected,ensure_ascii=False,separators=(",",":"))}

def make_hard(split,terms):
 rows=[]
 for domain,targets in terms.items():
  for idx,target in enumerate(targets):
   for state in range(4):
    rows.append(hard_case(split,domain,idx,target,state))
 return rows

def build_data():
 base=v5.build_data()
 data={"train":base["train"],"replay":base["replay"],"regression":base["regression"],
       "dev":make_hard("dev",DEV_TERMS),"test":make_hard("test",TEST_TERMS)}
 patch_v5(1706)
 v5.validate_data(data)
 assert len(data["dev"])==48 and len(data["test"])==96
 train_inputs={r["input"] for r in data["train"]}
 assert train_inputs.isdisjoint(r["input"] for r in data["dev"])
 assert train_inputs.isdisjoint(r["input"] for r in data["test"])
 assert {r["input"] for r in data["dev"]}.isdisjoint(r["input"] for r in data["test"])
 return data

def strip_outer_fence(text):
 s=text.strip()
 fence=chr(96)*3
 if s.startswith(fence) and s.endswith(fence):
  inner=s[len(fence):-len(fence)].strip()
  if inner.lower().startswith("json"):
   inner=inner[4:].lstrip()
  return inner, True
 return s, False

def score(row,text):
 if row["domain"]=="general":
  return {"valid":True,"exact":text.strip()==row["expected"]}
 expected=row["expected"]
 cleaned,had_fence=strip_outer_fence(text)
 obj=None
 try: obj=json.loads(cleaned)
 except (ValueError,TypeError): pass
 semantic_valid=isinstance(obj,dict) and set(obj)==set(expected) and all(isinstance(x,str) for x in obj.values())
 strict_json_no_fence=semantic_valid and not had_fence
 if not isinstance(obj,dict): obj={}
 return {
   "valid":semantic_valid,
   "strict_json_no_fence":strict_json_no_fence,
   "had_outer_code_fence":had_fence,
   "status_correct":obj.get("status")==expected["status"],
   "source_correct":obj.get("source")==expected["source"],
   "timeframe_correct":obj.get("timeframe")==expected["timeframe"],
   "evidence_exact":obj.get("evidence")==expected["evidence"],
   "evidence_is_source_substring":bool(obj.get("evidence")) and isinstance(obj.get("evidence"),str) and obj["evidence"] in row["input"],
   "all_fields_correct":semantic_valid and obj==expected,
   "unassessed_to_absent":expected["status"]=="not_assessed" and obj.get("status")=="absent",
   "uncertain_to_certain":expected["status"]=="uncertain" and obj.get("status") in ("present","absent")
 }

def self_test():
 data=build_data(); checked=0
 for key in ["train","dev","test","regression"]:
  for r in data[key]:
   s=score(r,r["assistant"])
   assert s.get("all_fields_correct",s.get("exact")); checked+=1
 sample=next(r for r in data["test"] if r["expected"]["status"]=="not_assessed")
 fence=chr(96)*3
 fenced=fence+"json\n"+sample["assistant"]+"\n"+fence
 fs=score(sample,fenced)
 assert fs["valid"] and fs["all_fields_correct"] and fs["had_outer_code_fence"] and not fs["strict_json_no_fence"]
 altered=dict(sample["expected"]); altered["status"]="absent"
 assert score(sample,json.dumps(altered))["unassessed_to_absent"]
 assert len({r["id"] for r in data["test"]})==96
 print(json.dumps({"self_test":"PASS","checked":checked,"counts":{k:len(v) for k,v in data.items()},
                   "test_sha256":sha(data["test"]),"seeds":SEEDS},ensure_ascii=False))

def worker(arm,root,model_path):
 seed=SEEDS[arm]
 patch_v5(seed)
 w=v5.Worker(arm,root,model_path)
 w.env["confirmation_version"]=VERSION
 w.env["replicate_seed"]=seed
 write_json(Path(root)/arm/"environment.json",w.env)
 result=w.run()
 result["replicate_seed"]=seed
 result["harder_holdout"]=True
 result["scoring_note"]="Optional outer JSON code fences are semantically parsed but tracked separately."
 write_json(Path(root)/arm/"result.json",result)
 return result

def summarize_arm(root,arm):
 result=json.loads((root/arm/"result.json").read_text())
 b=result["metrics"]["baseline"]["test"]
 a=result["metrics"]["after"]["test"]
 cb=b["clinical"]; ca=a["clinical"]; ab=b["task_all"]; aa=a["task_all"]
 regb=result["metrics"]["baseline"]["regression"]["general"]; rega=result["metrics"]["after"]["regression"]["general"]
 checks={
   "clinical_status_at_least_46_of_48":ca["status_correct"]>=46,
   "overall_status_at_least_92_of_96":aa["status_correct"]>=92,
   "no_unassessed_to_absent_on_test":aa["unassessed_to_absent"]==0,
   "no_uncertain_to_certain_on_test":aa["uncertain_to_certain"]==0,
   "evidence_exact_at_least_94_of_96":aa["evidence_exact"]>=94,
   "source_correct_at_least_94_of_96":aa["source_correct"]>=94,
   "timeframe_correct_at_least_94_of_96":aa["timeframe_correct"]>=94,
   "regression_not_worse_by_more_than_one":rega["exact"]>=regb["exact"]-1,
   "frozen_base_unchanged":bool(result["frozen_base_unchanged"]),
   "adapter_disable_restores_reference":bool(result["disabled_adapter_restores_reference"])
 }
 return {"seed":SEEDS[arm],"baseline_clinical_status":cb["status_correct"],"after_clinical_status":ca["status_correct"],
         "baseline_all_status":ab["status_correct"],"after_all_status":aa["status_correct"],
         "after_all_fields_correct":aa["all_fields_correct"],"after_unassessed_to_absent":aa["unassessed_to_absent"],
         "after_uncertain_to_certain":aa["uncertain_to_certain"],"after_evidence_exact":aa["evidence_exact"],
         "regression_before":regb["exact"],"regression_after":rega["exact"],
         "checks":checks,"pass":all(checks.values())}

def run_group(root,model_path,arms,gpus):
 procs=[]
 for arm,gpu in zip(arms,gpus):
  env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); env["TOKENIZERS_PARALLELISM"]="false"
  log=(root/f"{arm}_worker.log").open("w")
  cmd=[sys.executable,str(Path(__file__).resolve()),"--worker",arm,"--out",str(root),"--model-path",model_path]
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
  procs.append((arm,p,log))
  print("V06_WORKER_STARTED",arm,"GPU",gpu,flush=True)
 while any(p.poll() is None for _,p,_ in procs):
  stat={}
  for arm,p,_ in procs:
   f=root/arm/"progress.json"
   stat[arm]=json.loads(f.read_text()) if f.exists() else {"phase":"starting","exit_code":p.poll()}
  write_json(root/"progress.json",stat)
  print("V06_WORKERS",json.dumps(stat),flush=True)
  time.sleep(20)
 for _,_,log in procs: log.close()
 for arm,p,_ in procs:
  if p.returncode!=0:
   raise RuntimeError(f"WORKER_FAILED {arm}: "+(root/f"{arm}_worker.log").read_text()[-5000:])

def launch(root):
 import torch
 from huggingface_hub import snapshot_download
 patch_v5(1706)
 root=Path(root); root.mkdir(parents=True,exist_ok=False)
 data=build_data(); write_json(root/"data.json",data)
 prereg={
  "version":VERSION,"model_id":MODEL_ID,"revision":REVISION,
  "architecture":{"layers":list(range(28)),"rank":4,"target_modules":TARGETS,
                  "trainable_parameters":EXPECTED_TRAINABLE},
  "replicate_seeds":SEEDS,"steps_per_replicate":STEPS,"accumulation":ACCUM,"learning_rate":LR,
  "data_counts":{k:len(v) for k,v in data.items()},"data_sha256":{k:sha(v) for k,v in data.items()},
  "holdout":"new English context-rich synthetic dev/test; no v0.5 test rows reused",
  "scoring":"semantic JSON allows one outer markdown code fence; strict no-fence compliance tracked separately",
  "confirmation_gate":{
   "per_seed_clinical_status_min":"46/48",
   "per_seed_overall_status_min":"92/96",
   "per_seed_unassessed_to_absent":"0",
   "per_seed_uncertain_to_certain":"0",
   "per_seed_evidence_exact_min":"94/96",
   "per_seed_source_timeframe_min":"94/96 each",
   "regression_drop_max":1,
   "all_three_replicates_must_pass":True
  },
  "selection_rule":"Fixed 100-step final adapters; no test-driven retuning, checkpoint selection, or retries.",
  "original_weights_frozen":True,"adapter_merge":False,"uses_real_patients":False,
  "purpose":"Confirm robustness of v0.5 distributed adapter before freezing it as a reusable information-state adapter."
 }
 write_json(root/"preregistration.json",prereg)
 v5.preflight_tiny(root)
 if torch.cuda.device_count()<1: raise RuntimeError("GPU_REQUIRED")
 cache=Path("/kaggle/temp/emw_v06_hf_cache"); cache.mkdir(parents=True,exist_ok=True)
 print("V06_DOWNLOAD_REFERENCE",flush=True)
 model_path=snapshot_download(MODEL_ID,revision=REVISION,cache_dir=str(cache),
   allow_patterns=["*.json","*.safetensors","*.txt","*.model"],token=False)
 devices=torch.cuda.device_count()
 if devices>=2:
  run_group(root,model_path,["replicate_1706","replicate_2718"],[0,1])
  run_group(root,model_path,["replicate_3141"],[0])
 else:
  for arm in SEEDS: run_group(root,model_path,[arm],[0])
 summaries={arm:summarize_arm(root,arm) for arm in SEEDS}
 baseline_files=[json.loads((root/a/"baseline_test.json").read_text()) for a in SEEDS]
 baseline_agreement=True
 ref=baseline_files[0]
 for other in baseline_files[1:]:
  baseline_agreement &= all(x["output"]==y["output"] for x,y in zip(ref,other))
 base_hashes=[json.loads((root/a/"frozen_base_before.json").read_text()) for a in SEEDS]
 base_hash_match=all(h==base_hashes[0] for h in base_hashes[1:])
 ready=all(s["pass"] for s in summaries.values()) and baseline_agreement and base_hash_match
 result={"status":"COMPLETE_RESEARCH_ONLY","replicates":summaries,
         "between_seed_baseline_output_agreement":baseline_agreement,
         "between_seed_base_hash_match":base_hash_match,
         "confirmation_gate_pass":ready,
         "freeze_recommended":ready,
         "automatic_merge":False,"clinical_deployment_approved":False,
         "interpretation":"Confirms only the narrow English information-state adapter task if gate passes; not diagnostic or medication competence."}
 write_json(root/"comparison.json",result)
 write_json(root/"artifact_sha256.json",{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in root.rglob("*") if p.is_file() and p.name!="artifact_sha256.json"})
 print("V06_FINISHED",json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser(); p.add_argument("--run",action="store_true"); p.add_argument("--self-test",action="store_true")
 p.add_argument("--worker",choices=list(SEEDS)); p.add_argument("--out",default="/kaggle/working/emw_qwen_lora_confirmation_v06")
 p.add_argument("--model-path"); a=p.parse_args()
 if a.self_test: self_test()
 elif a.worker:
  try: worker(a.worker,a.out,a.model_path)
  except Exception as e:
   d=Path(a.out)/a.worker; d.mkdir(parents=True,exist_ok=True)
   write_json(d/"failure.json",{"error":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()})
   raise
 elif a.run: launch(a.out)
 else: p.print_help()

if __name__=="__main__": main()
