"""EMW v0.5: English-only, equal-parameter LoRA placement pilot.
Synthetic research only. No patient data, diagnosis, prescribing, production writes,
checkpoint merging, or control of other Kaggle notebooks. Requires --run explicitly.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, re, subprocess, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path

MODEL_ID='Qwen/Qwen2.5-Coder-7B-Instruct'
REVISION='c03e6d358207e414f1eca0bb1891e29f1db0e242'
VERSION='EMW_QWEN_ENGLISH_LORA_PLACEMENT_PILOT_V0.5'
SEED=1705
TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
ARMS={'distributed':{'layers':list(range(28)),'rank':4},
      'selective':{'layers':list(range(13,21)),'rank':14}}
EXPECTED_TRAINABLE=10_092_544
STATES=('present','absent','not_assessed','uncertain')
ACCUM=4; STEPS=100; LR=1e-4; MAX_LENGTH=512
INSTRUCTION=(
 'Extract only the explicit evidence about the target. Return one JSON object with exactly '
 'status, source, timeframe, evidence. status is present (explicitly reported), absent '
 '(explicitly denied), not_assessed (explicitly not evaluated), or uncertain (an uncertain report). '
 'source is the named author of the focal statement: patient, collateral, clinician, or inspector. '
 'timeframe is current or historical, as explicitly specified for the target. '
 'evidence must copy the entire focal statement exactly, without the surrounding quotation marks. '
 'Do not turn lack of assessment into a denial. Do not diagnose or recommend treatment.')
TERMS={
 'train':{'clinical':['persistent sadness','excessive worry','loss of interest','rapid speech',
  'restlessness','social withdrawal','irritability','loss of appetite','increased appetite',
  'low energy','avoidance of crowds','frequent checking','unpleasant dreams','feeling detached',
  'distractibility','feelings of guilt'],
  'control':['roof leaks','loose hinges','broken tiles','missing labels','cracked handles',
  'corroded contacts','damaged cables','blocked vents']},
 'dev':{'clinical':['sensitivity to noise','fear of elevators','indecisiveness','early waking',
  'reassurance seeking','agitation'],
  'control':['sticking keys','uneven wheels','faded markings','frayed straps','bent pins','dust deposits']},
 'test':{'clinical':['repeated hand washing','fear of public speaking','reduced enjoyment of music',
  'difficulty sustaining attention','avoidance of reminders','sudden crying spells',
  'feeling watched','unwanted repetitive images','reduced desire to socialize',
  'preoccupation with mistakes','difficulty making routine decisions','feeling emotionally numb'],
  'control':['misaligned panels','intermittent buzzing','stiff buttons','discolored seals',
  'leaking connectors','worn bearings','unreadable serial numbers','dented lids',
  'scraped paint','missing washers','warped shelves','unstable brackets']}}
# Split by wording families AND target terms. No v0.1-v0.4 result files are used.
TEMPLATES={
 'train':[
  ('The focal speaker reports {t}.','The focal speaker explicitly denies {t}.',
   'The focal speaker says {t} was not assessed.','The focal speaker is unsure whether {t} is present.'),
  ('When asked about {t}, the answer was yes.','When asked about {t}, the answer was no.',
   'No one asked about {t}, so its status was left unassessed.','Asked about {t}, the speaker could not give a definite answer.'),
  ('The account confirms the presence of {t}.','The account confirms the absence of {t}.',
   'The account records that {t} has not been evaluated.','The account describes {t} as possible but unconfirmed.'),
  ('Regarding {t}, the speaker states it occurs.','Regarding {t}, the speaker states it does not occur.',
   'Regarding {t}, the speaker states no assessment took place.','Regarding {t}, the speaker cannot confirm or exclude it.')],
 'dev':[
  ('The answer establishes that {t} is happening.','The answer establishes that {t} is not happening.',
   'The encounter did not include any enquiry into {t}.','The speaker remains undecided about whether {t} is happening.'),
  ('A direct enquiry about {t} received a positive reply.','A direct enquiry about {t} received a negative reply.',
   'The enquiry omitted {t} entirely; there was no evaluation of it.','The reply about {t} was explicitly inconclusive.')],
 'test':[
  ('The speaker affirms experiencing {t}.','The speaker rejects experiencing {t} when directly questioned.',
   'Evaluation of {t} was postponed, with no answer obtained.','The speaker thinks {t} might be occurring but cannot be sure.'),
  ('A direct question established that {t} had occurred.','A direct question established that {t} had not occurred.',
   'The session ended before {t} could be explored; it remains unevaluated.','The recollection of {t} is too uncertain to establish its presence.'),
  ('The focal account says {t} is definitely part of the experience.','The focal account says {t} is definitely not part of the experience.',
   'Although {t} was on the agenda, it was never discussed or evaluated.','The focal account says {t} may or may not be part of the experience.'),
  ('For the target, the speaker gives an unequivocal report of {t}.','For the target, the speaker gives an unequivocal denial of {t}.',
   'No evaluation or response concerning {t} is available because the topic was skipped.',
   'For the target, the speaker provides only a tentative report of {t}.')]}


def write_json(path, value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8'); tmp.replace(path)

def sha(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def make_case(split,domain,idx,target,family,state):
 sources=['patient','collateral','clinician','patient'] if domain=='clinical' else ['inspector']*4
 source=sources[(idx+family)%4]; timeframe='current' if (idx+family)%2==0 else 'historical'
 evidence=TEMPLATES[split][family][state].format(t=target)
 record=(f'Target: {target}\nTimeframe explicitly specified for this target: {timeframe}.\n'
         f'Author of the focal statement: {source}.\nFocal statement: "{evidence}"\n')
 if idx%2==0:
  record+='Unrelated background: a different issue was mentioned earlier. Do not use that issue for this target.'
 answer={'status':STATES[state],'source':source,'timeframe':timeframe,'evidence':evidence}
 return {'id':f'{split}_{domain}_{idx:02d}_f{family}_s{state}','split':split,'domain':domain,
         'family':f'{split}_f{family}','group_id':f'{split}_{domain}_{idx:02d}_f{family}',
         'target':target,'instruction':INSTRUCTION,'input':record,'expected':answer,
         'assistant':json.dumps(answer,ensure_ascii=False,separators=(',',':'))}


def general_cases(split,n):
 """Small deterministic instruction checks, not a general-intelligence benchmark."""
 rows=[]
 for i in range(n):
  k=i%4; a=(i+13 if split=='test' else i+2); b=7 if split=='test' else 3
  if k==0: q=f'Return only the integer sum of {a} and {b}.'; ans=str(a+b)
  elif k==1:
   word=(['planet','window','garden','velvet','silver','cloud','river','copper'][i//4] if split=='test'
         else ['apple','stone','paper','glass'][i//4])
   q=f'Return only this word in uppercase: {word}'; ans=word.upper()
  elif k==2: q=f'Return only the smallest integer in this list: {a+9}, {a}, {a+4}.'; ans=str(a)
  else:
   q=f'Return only the code between square brackets in: item [R{a:03d}] is ready.'; ans=f'R{a:03d}'
  rows.append({'id':f'{split}_general_{i:02d}','split':split,'domain':'general','family':f'general_{k}',
               'group_id':f'{split}_general_{i:02d}','instruction':'Follow the user instruction exactly.',
               'input':q,'assistant':ans,'expected':ans})
 return rows


def build_data():
 out={}
 for split in ('train','dev','test'):
  rows=[]
  for domain,terms in TERMS[split].items():
   for i,target in enumerate(terms):
    families=range(4) if split=='train' else [i%len(TEMPLATES[split])]
    for f in families:
     for s in range(4): rows.append(make_case(split,domain,i,target,f,s))
  out[split]=rows
 out['replay']=general_cases('train',16); out['regression']=general_cases('test',32)
 out['train']=out['train']+out['replay']
 random.Random(SEED).shuffle(out['train'])
 validate_data(out)
 return out


def validate_data(data):
 assert [len(data[k]) for k in ['train','dev','test','regression']]==[400,48,96,32]
 assert set(t for d in TERMS['train'].values() for t in d).isdisjoint(t for d in TERMS['test'].values() for t in d)
 for a,b in [('train','dev'),('train','test'),('dev','test'),('replay','regression')]:
  assert {r['input'] for r in data[a]}.isdisjoint(r['input'] for r in data[b])
 for split in ('train','dev','test'):
  rows=data[split]; assert len({r['id'] for r in rows})==len(rows)
  for r in rows:
   if r['domain']!='general':
    e=r['expected']; assert e['evidence'] in r['input']; assert e['status'] in STATES
    assert e['timeframe'] in ['current','historical']
 for arm,a in ARMS.items(): assert len(a['layers'])*a['rank']*90_112==EXPECTED_TRAINABLE
 assert STEPS*ACCUM==len(data['train'])


def score(row,text):
 if row['domain']=='general':
  return {'valid':True,'exact':text.strip()==row['expected']}
 expected=row['expected']; obj=None
 try: obj=json.loads(text.strip())
 except (ValueError,TypeError): pass
 valid=isinstance(obj,dict) and set(obj)==set(expected) and all(isinstance(v,str) for v in obj.values())
 if not isinstance(obj,dict): obj={}
 return {'valid':valid,'status_correct':obj.get('status')==expected['status'],
         'source_correct':obj.get('source')==expected['source'],
         'timeframe_correct':obj.get('timeframe')==expected['timeframe'],
         'evidence_exact':obj.get('evidence')==expected['evidence'],
         'evidence_is_source_substring':bool(obj.get('evidence')) and isinstance(obj.get('evidence'),str) and obj['evidence'] in row['input'],
         'all_fields_correct':valid and obj==expected,
         'unassessed_to_absent':expected['status']=='not_assessed' and obj.get('status')=='absent',
         'uncertain_to_certain':expected['status']=='uncertain' and obj.get('status') in ('present','absent')}


def self_test():
 data=build_data(); n=0
 for name in ['train','dev','test','regression']:
  for r in data[name]:
   m=score(r,r['assistant']); assert m.get('all_fields_correct',m.get('exact')); n+=1
 sample=data['test'][0]
 assert not score(sample,'not json')['valid']; n+=1
 assert not score(sample,'```json\n'+sample['assistant']+'\n```')['valid']; n+=1
 testneg=next(r for r in data['test'] if r['expected']['status']=='not_assessed')
 altered=dict(testneg['expected']); altered['status']='absent'
 assert score(testneg,json.dumps(altered))['unassessed_to_absent']; n+=1
 prefix=[1,2,3]; completion=[4,5]; labels=[-100]*len(prefix)+completion
 assert labels[:3]==[-100]*3 and labels[3:]==completion; n+=1
 print(json.dumps({'self_test':'PASS','checked_rows':n,'counts':{k:len(v) for k,v in data.items()},
                   'trainable_per_arm':EXPECTED_TRAINABLE,'test_hash':sha(data['test'])}))


def aggregate(rows):
 groups=defaultdict(list)
 for r in rows:
  groups[r['domain']].append(r)
  if r['domain']!='general': groups['task_all'].append(r)
 out={}
 for group,rs in groups.items():
  fields=list(rs[0]['scores'])
  out[group]={'n':len(rs),**{k:sum(bool(r['scores'][k]) for r in rs) for k in fields}}
 return out


class Worker:
 def __init__(self,arm,root,model_path):
  import torch, transformers, peft, bitsandbytes, accelerate
  from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
  from peft import prepare_model_for_kbit_training,get_peft_model,LoraConfig
  self.t=torch; self.arm=arm; self.root=Path(root); self.out=self.root/arm
  self.out.mkdir(exist_ok=False); self.start=time.monotonic(); self.log('loading_model')
  if not torch.cuda.is_available(): raise RuntimeError('CUDA_REQUIRED')
  torch.set_num_threads(2); torch.manual_seed(SEED); random.seed(SEED)
  torch.cuda.manual_seed_all(SEED); torch.backends.cuda.matmul.allow_tf32=False
  self.data=json.loads((self.root/'data.json').read_text()); validate_data(self.data)
  self.tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
  self.tok.padding_side='left'
  if self.tok.pad_token_id is None:self.tok.pad_token=self.tok.eos_token
  bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16)
  base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
       torch_dtype=torch.float16,quantization_config=bnb,device_map={'':0},attn_implementation='sdpa')
  assert base.config.num_hidden_layers==28 and base.config.hidden_size==3584
  base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
                                      gradient_checkpointing_kwargs={'use_reentrant':False})
  base.config.use_cache=False
  cfg=ARMS[arm]
  torch.manual_seed(SEED)
  lc=LoraConfig(r=cfg['rank'],lora_alpha=2*cfg['rank'],target_modules=TARGETS,
       layers_to_transform=cfg['layers'],layers_pattern='layers',lora_dropout=0.0,
       bias='none',task_type='CAUSAL_LM',init_lora_weights=True)
  self.model=get_peft_model(base,lc); self.model.eval()
  self.trainables=[p for p in self.model.parameters() if p.requires_grad]
  names=[n for n,p in self.model.named_parameters() if p.requires_grad]
  assert all('lora_' in n for n in names),names
  ntrain=sum(p.numel() for p in self.trainables)
  assert ntrain==EXPECTED_TRAINABLE,(ntrain,EXPECTED_TRAINABLE)
  adapted=sorted({int(m.group(1)) for n in names if (m:=re.search(r'layers\.(\d+)\.',n))})
  assert adapted==cfg['layers'],adapted
  assert all(p.dtype==torch.float32 for p in self.trainables),'ADAPTERS_MUST_BE_FP32'
  self.env={'arm':arm,'model_id':MODEL_ID,'revision':REVISION,'training_mode':'QLoRA NF4 double quantized',
   'compute_dtype':'float16','torch':torch.__version__,'transformers':transformers.__version__,
   'peft':peft.__version__,'bitsandbytes':bitsandbytes.__version__,'accelerate':accelerate.__version__,
   'gpu':torch.cuda.get_device_name(0),'rank':cfg['rank'],'layers':adapted,
   'trainable_parameters':ntrain,'seed':SEED,'max_length':MAX_LENGTH,'optimizer_steps':STEPS,
   'gradient_accumulation':ACCUM,'learning_rate':LR,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  write_json(self.out/'environment.json',self.env)
  write_json(self.out/'trainable_names.json',names)
  self.base_before=self.base_hash(); write_json(self.out/'frozen_base_before.json',self.base_before)
  self.initial_adapter_hash=self.adapter_hash()
  self.tokens=[self.training_tokens(r) for r in self.data['train']]
  write_json(self.out/'supervision_audit.json',{'n':len(self.tokens),'max_sequence':max(len(x['input_ids']) for x in self.tokens),
   'assistant_tokens':sum(sum(x!=-100 for x in z['labels']) for z in self.tokens),
   'prompt_tokens_masked':True,'no_truncation':True,'training_order_sha256':sha([r['id'] for r in self.data['train']])})

 def log(self,phase,**extra):
  rec={'arm':self.arm,'phase':phase,'elapsed_seconds':round(time.monotonic()-self.start,2),**extra}
  write_json(self.out/'progress.json',rec); print('V05_PROGRESS',json.dumps(rec),flush=True)

 def deadline(self):
  if time.monotonic()-self.start>6500:raise TimeoutError('THIS_WORKER_BUDGET_ONLY')

 def digest_tensor(self,p):
  h=hashlib.sha256(); h.update(str((list(p.shape),str(p.dtype))).encode())
  flat=p.detach().reshape(-1)
  for start in range(0,flat.numel(),2_000_000):
   x=flat[start:start+2_000_000].contiguous().cpu().view(self.t.uint8).numpy()
   h.update(x.tobytes())
  return h.hexdigest()

 def base_hash(self):
  result={}
  for n,p in self.model.named_parameters():
   if 'lora_' in n:continue
   canonical=n.removeprefix('base_model.model.').replace('.base_layer.','.')
   result[canonical]=self.digest_tensor(p)
   qs=getattr(p,'quant_state',None)
   if qs is not None:
    for k,v in qs.as_dict(packed=True).items():
     if self.t.is_tensor(v):result[canonical+'::quant::'+k]=self.digest_tensor(v)
     else:result[canonical+'::quant::'+k]=sha(v)
  return result

 def adapter_hash(self):
  return {n:self.digest_tensor(p) for n,p in self.model.named_parameters() if 'lora_' in n}

 def prompt(self,r):
  return self.tok.apply_chat_template([{'role':'system','content':r['instruction']},
      {'role':'user','content':r['input']}],tokenize=False,add_generation_prompt=True)

 def training_tokens(self,r):
  prefix=self.tok.encode(self.prompt(r),add_special_tokens=False)
  full=self.tok.encode(self.prompt(r)+r['assistant'],add_special_tokens=False)
  assert full[:len(prefix)]==prefix,'UNSTABLE_ASSISTANT_BOUNDARY'
  full=full+[self.tok.eos_token_id]; labels=[-100]*len(prefix)+full[len(prefix):]
  assert len(full)<=MAX_LENGTH,('TRAIN_TOO_LONG',r['id'],len(full))
  assert any(x!=-100 for x in labels)
  return {'input_ids':full,'labels':labels}

 def generate(self,records,tag,batch_size=4):
  t=self.t; rows=[]; self.model.eval(); self.model.config.use_cache=True
  for i in range(0,len(records),batch_size):
   self.deadline(); batch=records[i:i+batch_size]
   x=self.tok([self.prompt(r) for r in batch],return_tensors='pt',padding=True,add_special_tokens=False)
   if x.input_ids.shape[1]>MAX_LENGTH:raise RuntimeError('EVAL_TOO_LONG')
   x={k:v.to('cuda:0') for k,v in x.items()}; width=x['input_ids'].shape[1]
   with t.inference_mode():
    y=self.model.generate(**x,do_sample=False,max_new_tokens=128,
      pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id,
      repetition_penalty=1.0,use_cache=True)
   for j,r in enumerate(batch):
    ids=y[j,width:]; text=self.tok.decode(ids,skip_special_tokens=True)
    rows.append({'id':r['id'],'domain':r['domain'],'family':r['family'],'group_id':r['group_id'],
      'target':r.get('target'),'expected':r['expected'],'output':text,'scores':score(r,text),
      'hit_generation_cap':len(ids)>=128 and self.tok.eos_token_id not in ids.tolist()})
   if i%32==0:
    write_json(self.out/(tag+'.json'),rows); self.log(tag,done=len(rows),total=len(records))
  write_json(self.out/(tag+'.json'),rows);write_json(self.out/(tag+'_summary.json'),aggregate(rows))
  return rows

 def baseline(self):
  refs={}
  for name in ['dev','test','regression']:
   refs[name]=self.generate(self.data[name],'baseline_'+name)
  sentinel=self.data['test'][:4]
  with self.model.disable_adapter():disabled=self.generate(sentinel,'zero_adapter_disabled_check')
  reference={r['id']:r for r in refs['test']}
  matches=all(r['output']==reference[r['id']]['output'] for r in disabled)
  write_json(self.out/'zero_adapter_gate.json',{'identical_generated_output':matches})
  if not matches:raise RuntimeError('INITIAL_ZERO_ADAPTER_GATE_FAILED')
  return refs

 def train(self):
  t=self.t; self.model.train(); self.model.config.use_cache=False
  opt=t.optim.AdamW(self.trainables,lr=LR,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0)
  scaler=t.amp.GradScaler('cuda',init_scale=256.0,growth_interval=2000)
  opt.zero_grad(set_to_none=True); losses=[]; start=time.monotonic()
  for step in range(STEPS):
   self.deadline(); factor=min((step+1)/8,1.0)*min((STEPS-step)/max(STEPS-8,1),1.0)
   for g in opt.param_groups:g['lr']=LR*factor
   current=[]
   for j in range(ACCUM):
    row=self.tokens[step*ACCUM+j]
    x={k:t.tensor([v],dtype=t.long,device='cuda:0') for k,v in row.items()}
    x['attention_mask']=t.ones_like(x['input_ids'])
    with t.autocast('cuda',dtype=t.float16):
     result=self.model(**x,use_cache=False); raw_loss=result.loss
    if not t.isfinite(raw_loss):raise RuntimeError('NONFINITE_TRAIN_LOSS')
    current.append(float(raw_loss.detach()));scaler.scale(raw_loss/ACCUM).backward()
   scaler.unscale_(opt)
   grad=t.nn.utils.clip_grad_norm_(self.trainables,1.0,error_if_nonfinite=True)
   if not any(p.grad is not None and bool(t.count_nonzero(p.grad)) for p in self.trainables):
    raise RuntimeError('NO_ADAPTER_GRADIENTS')
   old_scale=scaler.get_scale();scaler.step(opt);scaler.update()
   if scaler.get_scale()<old_scale:raise RuntimeError('SCALER_SKIPPED_UPDATE')
   opt.zero_grad(set_to_none=True)
   rec={'step':step+1,'mean_loss':sum(current)/len(current),'lr':LR*factor,'grad_norm':float(grad),
        'elapsed_seconds':round(time.monotonic()-start,2)};losses.append(rec)
   with (self.out/'training_log.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   if step==0 or (step+1)%10==0:self.log('training',**rec)
   if (step+1)%50==0:
    self.model.save_pretrained(self.out/f'checkpoint_step_{step+1:03d}',safe_serialization=True,save_embedding_layers=False)
  del opt,scaler
  self.model.eval(); self.model.config.use_cache=True;t.cuda.empty_cache()
  self.model.save_pretrained(self.out/'adapter',safe_serialization=True,save_embedding_layers=False)
  return losses

 def run(self):
  refs=self.baseline();self.log('baseline_complete')
  losses=self.train();self.log('training_complete',optimizer_steps=len(losses))
  after_hash=self.base_hash();write_json(self.out/'frozen_base_after.json',after_hash)
  unchanged=after_hash==self.base_before
  changed=sum(v!=self.initial_adapter_hash.get(k) for k,v in self.adapter_hash().items())
  write_json(self.out/'integrity.json',{'frozen_base_and_quant_state_unchanged':unchanged,
    'base_hash_entries':len(after_hash),'changed_adapter_tensors':changed,
    'original_checkpoint_overwritten':False,'merge_performed':False})
  if not unchanged or not changed:raise RuntimeError('WEIGHT_INTEGRITY_OR_NO_UPDATE_FAILURE')
  after={name:self.generate(self.data[name],'after_'+name) for name in ['dev','test','regression']}
  with self.model.disable_adapter():back=self.generate(self.data['test'][:4],'trained_adapter_disabled_check')
  lookup={r['id']:r for r in refs['test']};restore=all(r['output']==lookup[r['id']]['output'] for r in back)
  if not restore:raise RuntimeError('DISABLED_ADAPTER_REFERENCE_CHANGED')
  metrics={'baseline':{k:aggregate(v) for k,v in refs.items()},'after':{k:aggregate(v) for k,v in after.items()}}
  b=metrics['baseline']['test']['task_all'];a=metrics['after']['test']['task_all']
  regb=metrics['baseline']['regression']['general'];rega=metrics['after']['regression']['general']
  promising=(a['status_correct']>=b['status_correct']+5 and a['unassessed_to_absent']<=b['unassessed_to_absent']
       and a['uncertain_to_certain']<=b['uncertain_to_certain'] and a['evidence_exact']>=b['evidence_exact']-1
       and rega['exact']>=regb['exact']-1)
  result={'status':'COMPLETE_RESEARCH_ONLY','arm':self.arm,'metrics':metrics,'optimizer_steps':len(losses),
   'frozen_base_unchanged':unchanged,'disabled_adapter_restores_reference':restore,
   'changed_adapter_tensors':changed,'pilot_promising_predeclared_gate':promising,
   'clinical_deployment_approved':False,'elapsed_seconds':round(time.monotonic()-self.start,2),
   'limitations':['single seed','synthetic template-family-disjoint test, not clinically validated',
     'source/timeframe explicitly provided','QLoRA baseline, not earlier FP16 comparison',
     'location and per-layer rank differ jointly; no isolated causal attribution to location',
     'general regression panel is narrow','no evidence of broad psychiatric expertise from this pilot']}
  write_json(self.out/'result.json',result);self.log('COMPLETE_RESEARCH_ONLY')
  return result


def preflight_tiny(root):
 """CPU smoke test actual PEFT target selection and assistant-mask math on tiny Qwen2."""
 import torch
 from transformers import Qwen2Config,Qwen2ForCausalLM
 from peft import LoraConfig,get_peft_model
 config=Qwen2Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=28,
                   num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
 counts=[]
 for arm,a in ARMS.items():
  torch.manual_seed(SEED);base=Qwen2ForCausalLM(config);base.requires_grad_(False)
  model=get_peft_model(base,LoraConfig(r=a['rank'],lora_alpha=2*a['rank'],target_modules=TARGETS,
       layers_to_transform=a['layers'],layers_pattern='layers',bias='none',task_type='CAUSAL_LM'))
  names=[n for n,p in model.named_parameters() if p.requires_grad]
  assert all('lora_' in n for n in names)
  counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
  model.train();inp=torch.tensor([[3,4,5,6,7]]);labels=torch.tensor([[-100,-100,-100,6,7]])
  loss=model(input_ids=inp,labels=labels).loss;loss.backward()
  assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
  assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
 assert counts[0]==counts[1],counts
 write_json(Path(root)/'peft_cpu_preflight.json',{'pass':True,'tiny_trainable_counts':counts,
   'real_model_trainable_expected':EXPECTED_TRAINABLE,'real_gpu_test_not_implied':True})


def launch(root):
 import torch
 from huggingface_hub import snapshot_download
 root=Path(root);root.mkdir(parents=True,exist_ok=False)
 data=build_data();write_json(root/'data.json',data)
 prereg={'version':VERSION,'model_id':MODEL_ID,'revision':REVISION,'arms':ARMS,
   'target_modules':TARGETS,'equal_trainable_parameters':EXPECTED_TRAINABLE,'data_counts':{k:len(v) for k,v in data.items()},
   'data_sha256':{k:sha(v) for k,v in data.items()},'steps_per_arm':STEPS,'accumulation':ACCUM,
   'learning_rate':LR,'dropout':0,'seed':SEED,'max_length':MAX_LENGTH,
   'selection_rule':'Final fixed-step adapter; no test-driven stopping, checkpoint selection, retries or retuning.',
   'primary_endpoint':'heldout English status correct / 96, compared to same prepared quantized base',
   'pilot_promising_gate':{'minimum_status_gain_count':5,'unassessed_to_absent_must_not_increase':True,
    'uncertain_to_certain_must_not_increase':True,'max_evidence_exact_drop':1,'max_regression_count_drop':1},
   'limitations':'Synthetic development pilot; exposed hereafter; no clinical approval or psychiatric-specialization claim.',
   'original_weights_frozen':True,'adapter_merge':False,'uses_real_patients':False,
   'training_quantization':'NF4 double quantization; FP16 compute; FP32 adapters and preparation norms/embeddings',
   'not_matched_to_v04':'different quantization, prompt, data and dependencies; baseline rerun in this experiment',
   'rank_location_confound':'Ranks 4 vs 14 equalize total parameters; cannot identify location as sole cause.',
   'readme_sources':['https://huggingface.co/docs/peft/developer_guides/quantization',
     'https://huggingface.co/docs/peft/package_reference/lora']}
 write_json(root/'preregistration.json',prereg)
 preflight_tiny(root)
 cache=Path('/kaggle/temp/emw_v05_hf_cache');cache.mkdir(parents=True,exist_ok=True)
 print('V05_DOWNLOAD_REFERENCE',flush=True)
 model_path=snapshot_download(MODEL_ID,revision=REVISION,cache_dir=str(cache),
   allow_patterns=['*.json','*.safetensors','*.txt','*.model'],token=False)
 write_json(root/'reference_files.json',[{'name':p.name,'size':p.stat().st_size} for p in Path(model_path).iterdir() if p.is_file()])
 if torch.cuda.device_count()<1:raise RuntimeError('GPU_REQUIRED')
 workers=[]
 for i,arm in enumerate(ARMS):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i%torch.cuda.device_count());env['TOKENIZERS_PARALLELISM']='false'
  command=[sys.executable,str(Path(__file__).resolve()),'--worker',arm,'--out',str(root),'--model-path',model_path]
  log=(root/f'{arm}_worker.log').open('w')
  proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
  workers.append((arm,proc,log));print('V05_WORKER_STARTED',arm,flush=True)
  if torch.cuda.device_count()==1:proc.wait();log.close()
 while any(p.poll() is None for _,p,_ in workers):
  statuses={}
  for arm,p,_ in workers:
   f=root/arm/'progress.json'
   statuses[arm]=json.loads(f.read_text()) if f.exists() else {'phase':'starting','exit_code':p.poll()}
  write_json(root/'progress.json',statuses);print('V05_WORKERS',json.dumps(statuses),flush=True)
  time.sleep(20)
 for _,_,log in workers:
  if not log.closed:log.close()
 results={}
 for arm,p,_ in workers:
  rp=root/arm/'result.json'
  if p.returncode!=0 or not rp.exists():
   results[arm]={'status':'FAILED','returncode':p.returncode,'log_tail':(root/f'{arm}_worker.log').read_text()[-6000:]}
  else:results[arm]=json.loads(rp.read_text())
 ok=all(r.get('status')=='COMPLETE_RESEARCH_ONLY' for r in results.values())
 out={'status':'COMPLETE_RESEARCH_ONLY' if ok else 'PARTIAL_OR_FAILED','arms':results,
      'automatic_merge':False,'clinical_deployment_approved':False}
 if ok:
  b0=json.loads((root/'distributed'/'baseline_test.json').read_text())
  b1=json.loads((root/'selective'/'baseline_test.json').read_text())
  out['between_arm_baseline_output_agreement']=sum(x['output']==y['output'] for x,y in zip(b0,b1))/len(b0)
  out['between_arm_base_hash_match']=json.loads((root/'distributed'/'frozen_base_before.json').read_text())==json.loads((root/'selective'/'frozen_base_before.json').read_text())
  if not out['between_arm_base_hash_match']:out['comparison_interpretation']='BASE_MISMATCH_NO_WINNER'
  else:out['comparison_interpretation']='EXPLORATORY_SINGLE_SEED_NO_AUTOMATIC_WINNER'
 write_json(root/'comparison.json',out)
 write_json(root/'artifact_sha256.json',{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in root.rglob('*') if p.is_file() and p.name!='artifact_sha256.json'})
 print('V05_FINISHED',out['status'],flush=True)
 if not ok:raise RuntimeError('ONE_OR_MORE_PILOT_ARMS_FAILED_SEE_LOGS')


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--self-test',action='store_true')
 p.add_argument('--worker',choices=list(ARMS));p.add_argument('--out',default='/kaggle/working/emw_qwen_lora_pilot_v05')
 p.add_argument('--model-path');a=p.parse_args()
 if a.self_test:self_test();return
 if a.worker:
  try:Worker(a.worker,a.out,a.model_path).run()
  except Exception as e:
   write_json(Path(a.out)/a.worker/'failure.json',{'error':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
   raise
 elif a.run:launch(a.out)
 else:p.print_help()

if __name__=='__main__':main()
