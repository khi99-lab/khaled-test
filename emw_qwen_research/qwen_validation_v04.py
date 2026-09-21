"""EMW Qwen v0.4: new synthetic cases, measurement audit, preregistered head tests.
Research only. No clinical deployment, training, optimizer, or weight writes.
Run: python qwen_validation_v04.py --run; local checks: --self-test
"""
from __future__ import annotations
import argparse, contextlib, hashlib, inspect, json, math, os, re, signal
import statistics, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path

MODEL_ID = 'Qwen/Qwen2.5-Coder-7B-Instruct'
REVISION = 'c03e6d358207e414f1eca0bb1891e29f1db0e242'
VERSION = 'EMW_QWEN_MEASUREMENT_AND_HEAD_VALIDATION_V0.4'
HEADS = {'h21': [21], 'h24': [24], 'both': [21,24], 'control': [8]}
LAYER, HEAD_DIM = 18, 128
BATCH = 4
SITES = ('prompt_end', 'evidence_end', 'neutral')
# Freeze these before observing the model outputs. No reranking on this sample.
CONDITIONS = [('zero','h21'),('zero','h24'),('zero','both'),('zero','control'),
              ('opposite','both'),('same_state','both'),('opposite','control')]
POLICY = {'noop_logit_atol': 0.001, 'batch_margin_atol': 0.15,
          'eligible_label_accuracy': 0.8, 'eligible_swap_consistency': 0.9,
          'eligible_generated_label_validity': 0.9,
          'eligible_generation_score_agreement': 0.9,
          'paired_bootstrap_seed': 1704, 'paired_bootstrap_resamples': 1000}
LABELS = {'assessment': ('denied','unassessed'),
          'certainty': ('certain','uncertain'), 'sources': ('conflict','agreement')}
DESCRIPTIONS = {
'en': {'assessment': ('the target was explicitly assessed and denied or found absent',
                      'the target was explicitly not assessed; its presence is unknown'),
       'certainty': ('the speaker is certain of one exact target value',
                     'the speaker explicitly expresses uncertainty about the target value'),
       'sources': ('the two sources disagree about the same target in the same period',
                   'the two sources agree about the same target in the same period')},
'ar': {'assessment': ('تم تقييم الأمر المستهدف صراحة ونفيه أو التأكد من عدم وجوده',
                      'لم يتم تقييم الأمر المستهدف صراحة ولا نعرف وجوده أو غيابه'),
       'certainty': ('المتحدث متأكد من قيمة واحدة محددة للأمر المستهدف',
                     'المتحدث يصرح بعدم التأكد من قيمة الأمر المستهدف'),
       'sources': ('المصدران مختلفان بشأن الأمر المستهدف في الفترة نفسها',
                   'المصدران متفقان بشأن الأمر المستهدف في الفترة نفسها')}}


def make_cases():
    """60 new matched pairs / 120 statements; no source uses a real patient."""
    cases=[]
    def add(domain, axis, i, target, before, evidence0, evidence1, after=''):
        language = 'en' if i < 6 else 'ar'
        pair=f'{domain}_{axis}_{i:02d}'
        for state,evidence in enumerate((evidence0,evidence1)):
            neutral='Record begins. ' if language=='en' else 'بداية السجل. '
            # A shared neutral prefix occurs BEFORE all state-specific evidence.
            prefix=neutral+before
            text=prefix+evidence+after
            cases.append({'id':f'{pair}_{state}', 'pair_id':pair, 'domain':domain,
                          'axis':axis,'index':i,'language':language,'state':state,
                          'target':target,'text':text,'evidence':evidence,
                          'evidence_start':len(prefix),'neutral':neutral.rstrip()})
    clinical_targets=['episodes of derealization','fear of leaving home','unwanted aggressive thoughts',
                      'episodes of binge eating','flashbacks after the accident','difficulty initiating sleep',
                      'نوبات تبدد الشخصية','الخوف من الأماكن المزدحمة','الأفكار الاقتحامية','الكوابيس بعد الحادث']
    control_targets=['water stains on the ceiling','missing pages in the shipment','loose screws in the frame',
                     'scratches on the display','corrosion inside the valve','faults in the sensor',
                     'بقع الماء على السقف','الصفحات المفقودة','البراغي المرتخية','الخدوش على الشاشة']
    for domain,targets in [('clinical',clinical_targets),('control',control_targets)]:
        for i,t in enumerate(targets):
            if i<6:
                if domain=='clinical':
                    variations=[
                      (f'The clinician asked about {t}. The patient answered: ', '"No, that has not happened to me."', '"We did not discuss that subject at this visit."'),
                      (f'Review item: {t}. The note states: ', '"Asked directly; the patient said this was absent."', '"This item was left unasked; no answer was recorded."'),
                      (f'Today the record addresses {t}: ', '"The patient was asked and explicitly reported none."', '"This topic was never raised with the patient."')]
                else:
                    variations=[
                      (f'The inspector was asked about {t}. The report says: ', '"I checked directly and found none."', '"I did not examine that issue during this inspection."'),
                      (f'Review item: {t}. The report states: ', '"Checked directly; this defect was absent."', '"This item was left unchecked; no result was recorded."'),
                      (f'Today the record addresses {t}: ', '"The technician examined it and explicitly reported none."', '"This issue was never examined by the technician."')]
                a,b,c=variations[i%3]
                # Not-assessed is a narrator/report statement, not an implausible patient answer.
                if i%3==0 and domain=='clinical':
                    a=f'The visit record concerns {t}. It states: '
                    b='"The clinician asked directly and the patient denied it."'
                    c='"The clinician did not ask about it and no assessment was made."'
                add(domain,'assessment',i,t,a,b,c,' End of record.')
            else:
                a=f'الأمر المستهدف هو {t}. ورد في السجل: '
                if domain=='clinical':
                    b='«سأل الطبيب عنه مباشرة فأجاب المريض بأنه غير موجود»'
                    c='«لم يسأل الطبيب عنه ولم يتم تقييمه في هذه الزيارة»'
                else:
                    b='«فحصه الفني مباشرة وأكد عدم وجوده»'
                    c='«لم يفحصه الفني ولم يتم تقييمه في هذه المراجعة»'
                add(domain,'assessment',i,t,a,b,c,' انتهى السجل.')
    meds=['escitalopram','venlafaxine','mirtazapine','lithium','buspirone','duloxetine',
          'الإسيتالوبرام','الفينلافاكسين','الميرتازابين','البوسبيرون']
    values=['10 mg','75 mg','15 mg','300 mg','5 mg','30 mg','10 ملغ','75 ملغ','15 ملغ','5 ملغ']
    objs=['length of the shelf','number of chairs','delivery time','volume of the container',
          'weight of the parcel','temperature setting','طول الرف','عدد الكراسي','وزن الطرد','سعة الوعاء']
    cvalues=['90 cm','14','14:20','750 mL','3 kg','22 degrees','90 سم','14','3 كغ','750 مل']
    for domain,targets,vals in [('clinical',meds,values),('control',objs,cvalues)]:
        for i,(t,v) in enumerate(zip(targets,vals)):
            if i<6:
                target=f'reported dose of {t}' if domain=='clinical' else t
                prefix=f'The speaker is describing {target}. They say: '
                # Identical candidate value on both sides isolates explicit uncertainty.
                e0=f'"The value is {v}; I checked the label and I am certain."'
                e1=f'"The value may be {v}; I have not checked the label and I am uncertain."'
                if i%2:
                    e0=f'"I am sure that {v} is the correct value, not a guess."'
                    e1=f'"I can only guess {v}; I cannot confirm that value."'
                add(domain,'certainty',i,target,prefix,e0,e1,' Nothing else was reported.')
            else:
                target=f'الجرعة المبلّغ عنها من {t}' if domain=='clinical' else t
                add(domain,'certainty',i,target,f'المتحدث يصف {target}. قال: ',
                    f'«القيمة هي {v} وقد تحققت منها وأنا متأكد»',
                    f'«قد تكون القيمة {v} لكني لم أتحقق ولست متأكدا»',' لا توجد معلومات إضافية.')
    ct=[('patient','roommate','ate breakfast every morning'),('patient','caregiver','attended every therapy session'),
        ('adolescent','parent','finished homework each evening'),('patient','partner','left the house each day'),
        ('patient','sibling','kept the planned bedtime'),('patient','friend','went to the support group'),
        ('المريض','مرافقه','تناول الإفطار كل صباح'),('المريض','زوجته','حضر كل جلسات العلاج'),
        ('الطالب','والدته','أكمل الواجب كل مساء'),('المريض','أخوه','خرج من المنزل كل يوم')]
    nt=[('operator','supervisor','backed up the files every evening'),('driver','dispatcher','delivered each parcel'),
        ('clerk','auditor','counted the receipts every morning'),('host','caretaker','closed the gate each night'),
        ('technician','manager','tested the pump each day'),('chef','assistant','checked the oven temperature'),
        ('الموظف','مديره','حفظ الملفات كل مساء'),('السائق','المشرف','سلم جميع الطرود'),
        ('الفني','المدير','فحص المضخة كل يوم'),('الطاهي','مساعده','فحص حرارة الفرن')]
    for domain,triples in [('clinical',ct),('control',nt)]:
        for i,(a,b,event) in enumerate(triples):
            if i<6:
                pref=f'Both speakers describe the same person and the same past seven days. The {a} says: "I {event}." The {b} says: '
                e0=f'"That is not true: they did not do that during those seven days."'
                e1=f'"That is true: they did exactly that during those seven days."'
                add(domain,'sources',i,event,pref,e0,e1,' These are two reports, not verified findings.')
            else:
                pref=f'المصدران يصفان الشخص نفسه والأيام السبعة نفسها. يقول {a}: «{event}». ويقول {b}: '
                e0='«هذه المعلومة غير صحيحة فهو لم يفعل ذلك في تلك الأيام»'
                e1='«هذه المعلومة صحيحة فهو فعل ذلك فعلا في تلك الأيام»'
                add(domain,'sources',i,event,pref,e0,e1,' هذه روايات وليست حقائق مستقلة التحقق.')
    validate_cases(cases)
    return cases


def validate_cases(cases):
    assert len(cases)==120 and len({c['id'] for c in cases})==120
    assert len({c['text'] for c in cases})==120
    assert Counter(c['domain'] for c in cases)=={'clinical':60,'control':60}
    assert Counter(c['language'] for c in cases)=={'en':72,'ar':48}
    groups=defaultdict(list)
    for c in cases:
        assert c['text'][c['evidence_start']:c['evidence_start']+len(c['evidence'])]==c['evidence']
        assert c['text'].startswith(c['neutral'])
        groups[c['pair_id']].append(c)
    assert all({c['state'] for c in g}=={0,1} for g in groups.values())


def build_prompt(c, mode='letters', swap=False):
    language=c['language']; descriptions=DESCRIPTIONS[language][c['axis']]
    order=[1,0] if swap else [0,1]
    if language=='en':
        body=f'Target: {c["target"]}\nStatement:\n{c["text"]}\n'
        if mode=='letters':
            instruction='Classify the target from the explicit evidence. Reply with exactly A or B, without explanation.\n'
        else:
            instruction='Classify the target from the explicit evidence. Start with the exact category, then |, then a short exact quote from the statement. Do not use A/B.\n'
    else:
        body=f'الأمر المستهدف: {c["target"]}\nالنص:\n{c["text"]}\n'
        instruction=('صنف الأمر المستهدف بناء على الدليل الصريح. أجب بحرف A أو B فقط دون شرح.\n' if mode=='letters'
                     else 'صنف الأمر المستهدف. ابدأ باسم الفئة الإنجليزي المحدد ثم | ثم اقتباس قصير حرفي من النص. لا تستخدم A أو B.\n')
    if mode=='letters':
        options='\n'.join(f'{"AB"[j]} = {descriptions[s]}' for j,s in enumerate(order))
    else:
        options='\n'.join(f'{LABELS[c["axis"]][s]} = {descriptions[s]}' for s in order)
    return instruction+options+'\n'+body


def parse_generated(text, axis, mode, swap=False):
    s=text.strip()
    if mode=='letters':
        m=re.fullmatch(r'([AB])\s*[.!。]?',s)
        if not m: return None
        return [1,0][ord(m[1])-65] if swap else ord(m[1])-65
    m=re.match(r'^([a-z]+)\s*\|\s*(.+)',s,flags=re.S)
    if m and m[1] in LABELS[axis]: return LABELS[axis].index(m[1])
    return None


def batch_position_ids(mask):
    p=mask.long().cumsum(-1)-1
    return p.masked_fill(mask==0,0)


def intervention_tensor(x, positions, heads, source=None, zero=False):
    y=x.clone()
    assert len(positions)==x.shape[0]
    for b,pos in enumerate(positions):
        assert 0<=pos<x.shape[1]
        for h in heads:
            left,right=h*HEAD_DIM,(h+1)*HEAD_DIM
            if zero: y[b,pos,left:right]=0
            else: y[b,pos,left:right]=source[b,left:right].to(y.device,y.dtype)
    return y


def self_test():
    import torch
    cases=make_cases(); assertions=4
    for c in cases:
        for swap in (False,True):
            word='AB'[1-c['state'] if swap else c['state']]
            assert parse_generated(word,c['axis'],'letters',swap)==c['state']; assertions+=1
            assert c['text'] in build_prompt(c,'letters',swap); assertions+=1
        assert parse_generated(LABELS[c['axis']][c['state']]+' | quote',c['axis'],'semantic')==c['state']; assertions+=1
    for bad in ['A or B','Probably A','', 'AB','Answer: A']:
        assert parse_generated(bad,'assessment','letters') is None; assertions+=1
    assert parse_generated('unassessed','assessment','semantic') is None; assertions+=1
    mask=torch.tensor([[0,0,1,1],[1,1,1,1]])
    assert batch_position_ids(mask).tolist()==[[0,0,0,1],[0,1,2,3]]; assertions+=1
    x=torch.randn(2,5,3584); src=torch.randn(2,3584); original=x.clone()
    y=intervention_tensor(x,[3,4],[21,24],source=src)
    assert torch.equal(x,original); assertions+=1
    for b,p in enumerate([3,4]):
        assert torch.equal(y[b,p,2688:2816],src[b,2688:2816]); assertions+=1
        assert torch.equal(y[b,p,3072:3200],src[b,3072:3200]); assertions+=1
    z=intervention_tensor(x,[3,4],[21,24],source=torch.stack([x[0,3],x[1,4]]))
    assert torch.equal(x,z); assertions+=1
    other=torch.ones_like(x,dtype=torch.bool)
    for b,p in enumerate([3,4]):
        for h in [21,24]: other[b,p,h*128:(h+1)*128]=False
    assert torch.equal(x[other],y[other]); assertions+=1
    print(json.dumps({'status':'PASS','assertions':assertions,'cases':120,'pairs':60,
                      'language_counts':Counter(c['language'] for c in cases)},ensure_ascii=False))


if __name__=='__main__' and '--self-test' in sys.argv:
    self_test()

class Experiment:
    def __init__(self, out):
        import torch, transformers
        import numpy as np
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch, self.np = torch, np
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.started=time.monotonic(); self.cache={}; self.rows=[]
        self.status('initializing')
        self.cases=make_cases(); self.lookup={c['id']:c for c in self.cases}
        self.save('cases.json',self.cases)
        self.save('preregistration.json',{'version':VERSION,'model_id':MODEL_ID,'revision':REVISION,
            'layer':LAYER,'head_dim':HEAD_DIM,'heads':HEADS,'conditions':CONDITIONS,'sites':SITES,
            'policy':POLICY,'scope':'synthetic research cases; not clinical competence validation',
            'selected_before_results':True,'clinical_cases':60,'control_cases':60,
            'primary_analysis_unit':'matched pair, clustered across A/B mappings',
            'no_training':True,'no_checkpoint_writes':True})
        if not torch.cuda.is_available(): raise RuntimeError('GPU_NOT_AVAILABLE')
        torch.manual_seed(17); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,trust_remote_code=False)
        self.tokenizer.padding_side='left'
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=REVISION,
            trust_remote_code=False,torch_dtype=torch.float16,device_map='balanced',
            max_memory={i:'11GiB' for i in range(torch.cuda.device_count())},
            low_cpu_mem_usage=True,attn_implementation='sdpa')
        self.model.eval(); self.model.requires_grad_(False)
        self.device=self.model.get_input_embeddings().weight.device
        self.module=self.model.model.layers[LAYER].self_attn.o_proj
        assert len(self.model.model.layers)==28
        assert self.model.config.hidden_size==3584 and self.model.config.num_attention_heads==28
        assert not getattr(self.model,'is_quantized',False)
        assert not any(p.requires_grad for p in self.model.parameters())
        self.environment={'torch':torch.__version__,'transformers':transformers.__version__,
            'numpy':np.__version__,'python':sys.version,'dtype':'float16','backend':'sdpa',
            'model_revision':REVISION,'model_id':MODEL_ID,
            'device_map':{k:str(v) for k,v in self.model.hf_device_map.items()},
            'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'parameter_count':sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.save('environment.json',self.environment)
        self.save('model_config.json',self.model.config.to_dict())
        self.save('generation_config.json',self.model.generation_config.to_dict())
        self.module_source=inspect.getsource(type(self.model.model.layers[LAYER].self_attn))
        (self.out/'attention_implementation.txt').write_text(self.module_source,encoding='utf8')
        self.before=self.full_weight_digest(); self.save('weight_hashes_before.json',self.before)
        self.label_ids={}
        for letter in ['A','B']:
            choices={tuple(self.tokenizer.encode(v,add_special_tokens=False)) for v in [letter,' '+letter]}
            assert all(len(x)==1 for x in choices),'LABEL_VARIANT_NOT_SINGLE_TOKEN'
            self.label_ids[letter]=sorted(x[0] for x in choices)
        self.save('answer_token_audit.json',{'ids':self.label_ids,
            'decoded':{k:[self.tokenizer.decode([x]) for x in v] for k,v in self.label_ids.items()}})

    def save(self,name,data):
        p=self.out/name; tmp=p.with_name(p.name+'.tmp')
        tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        tmp.replace(p)

    def status(self,phase,**kwargs):
        self.save('progress.json',{'phase':phase,'elapsed_seconds':round(time.monotonic()-self.started,2),**kwargs})
        print('V04_PROGRESS',phase,json.dumps(kwargs,ensure_ascii=False),flush=True)

    def deadline(self):
        if time.monotonic()-self.started>6400: raise TimeoutError('LOCAL_EXPERIMENT_BUDGET_REACHED')

    def full_weight_digest(self):
        """Hash EVERY byte of the loaded FP16 parameters, not a few sample weights."""
        t=self.torch; hashes={}
        for name,p in self.model.named_parameters():
            flat=p.detach().reshape(-1); h=hashlib.sha256()
            h.update(json.dumps({'name':name,'shape':list(p.shape),'dtype':str(p.dtype)}).encode())
            for start in range(0,flat.numel(),4_000_000):
                part=flat[start:start+4_000_000].contiguous().cpu().view(t.uint8).numpy()
                h.update(part.tobytes())
            hashes[name]=h.hexdigest()
        return hashes

    def item(self,c,mode='letters',swap=False,completion=None):
        rendered=self.tokenizer.apply_chat_template([{'role':'user','content':build_prompt(c,mode,swap)}],
            tokenize=False,add_generation_prompt=True)
        enc=self.tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
        ids=enc['input_ids']; offsets=enc['offset_mapping']; start=rendered.index(c['text'])
        def last_overlap(a,b):
            found=[i for i,(s,e) in enumerate(offsets) if e>a and s<b and e>s]
            if not found: raise ValueError('SPAN_NOT_TOKENIZED')
            return found[-1]
        sites={'prompt_end':len(ids)-1,
               'evidence_end':last_overlap(start+c['evidence_start'],start+c['evidence_start']+len(c['evidence'])),
               'neutral':last_overlap(start,start+len(c['neutral']))}
        assert sites['neutral']<sites['evidence_end']<sites['prompt_end']
        extra=[]
        if completion is not None:
            whole=self.tokenizer.encode(rendered+completion,add_special_tokens=False)
            if whole[:len(ids)]!=ids: raise ValueError('UNSTABLE_COMPLETION_BOUNDARY')
            extra=whole[len(ids):]
            assert extra
        return {'case':c,'mode':mode,'swap':swap,'prompt_len':len(ids),'seq':ids+extra,
                'completion':extra,'sites':sites}

    def inputs(self,items):
        t=self.torch; width=max(len(i['seq']) for i in items)
        if width>1024: raise RuntimeError('UNEXPECTED_PROMPT_LENGTH')
        x=t.full((len(items),width),self.tokenizer.pad_token_id,dtype=t.long,device=self.device)
        mask=t.zeros_like(x)
        for b,i in enumerate(items):
            x[b,width-len(i['seq']):]=t.tensor(i['seq'],device=self.device)
            mask[b,width-len(i['seq']):]=1
        return {'input_ids':x,'attention_mask':mask,'position_ids':batch_position_ids(mask)}

    def cache_key(self,i,site):
        return (i['case']['id'],i['mode'],i['swap'],site)

    def donor(self,i,action):
        c=i['case']
        if action=='self': return c
        if action=='opposite': return self.lookup[c['pair_id']+'_'+str(1-c['state'])]
        if action=='same_state':
            group=sorted([s for s in self.cases if (s['domain'],s['axis'],s['language'],s['state'])==
                          (c['domain'],c['axis'],c['language'],c['state'])],key=lambda s:s['index'])
            n=next(j for j,s in enumerate(group) if s['id']==c['id'])
            return group[(n+1)%len(group)]
        raise ValueError(action)

    @contextlib.contextmanager
    def hook(self,items,width,capture=False,intervention=None):
        calls=[0]
        def callback(module,args):
            x=args[0]
            # Generation: intervene once at prefill; never patch later generated tokens.
            if calls[0]>0: return None
            calls[0]+=1
            assert x.ndim==3 and x.shape[-1]==3584 and x.shape[1]==width
            if capture:
                for b,i in enumerate(items):
                    for site,pos in i['sites'].items():
                        absolute=width-len(i['seq'])+pos
                        self.cache[self.cache_key(i,site)]=x[b,absolute].detach().float().cpu().clone()
            if intervention is None: return None
            action,group,site=intervention
            positions=[width-len(i['seq'])+i['sites'][site] for i in items]
            source=None
            if action!='zero':
                source=self.torch.stack([self.cache[(self.donor(i,action)['id'],i['mode'],i['swap'],site)] for i in items])
            y=intervention_tensor(x,positions,HEADS[group],source,zero=action=='zero')
            return (y,)+tuple(args[1:])
        handle=self.module.register_forward_pre_hook(callback)
        try: yield
        finally: handle.remove()

    def forward(self,items,keep=1,capture=False,intervention=None):
        self.deadline(); inputs=self.inputs(items)
        with self.hook(items,inputs['input_ids'].shape[1],capture,intervention),self.torch.inference_mode():
            o=self.model(**inputs,use_cache=False,logits_to_keep=keep,return_dict=True)
        return o.logits.detach().float().cpu(), inputs['input_ids'].shape[1]

    def letter_results(self,items,capture=False,intervention=None):
        logits,_=self.forward(items,capture=capture,intervention=intervention)
        z=logits[:, -1]; lp=z.log_softmax(-1); out=[]
        for b,i in enumerate(items):
            a=float(self.torch.logsumexp(lp[b,self.label_ids['A']],dim=0))
            v=float(self.torch.logsumexp(lp[b,self.label_ids['B']],dim=0))
            state=(0 if a>=v else 1) ^ int(i['swap'])
            # Both accepted surface forms are scored; bare and space-prefixed audits remain available.
            out.append({'id':i['case']['id'],'domain':i['case']['domain'],'axis':i['case']['axis'],
                'language':i['case']['language'],'pair_id':i['case']['pair_id'],'truth':i['case']['state'],
                'swap':i['swap'],'pred':state,'margin':(a-v)*(1 if i['case']['state']==int(i['swap']) else -1),
                'probability_mass_AB':math.exp(a)+math.exp(v),
                'raw_top_token':self.tokenizer.decode([int(z[b].argmax())]),
                'bare_margin':float(z[b,self.tokenizer.encode('A',add_special_tokens=False)[0]]-z[b,self.tokenizer.encode('B',add_special_tokens=False)[0]]),
                'space_margin':float(z[b,self.tokenizer.encode(' A',add_special_tokens=False)[0]]-z[b,self.tokenizer.encode(' B',add_special_tokens=False)[0]])})
        return out

    def generate(self,items,intervention=None):
        self.deadline(); inputs=self.inputs(items); inputs.pop('position_ids')
        width=inputs['input_ids'].shape[1]
        with self.hook(items,width,False,intervention),self.torch.inference_mode():
            seq=self.model.generate(**inputs,max_new_tokens=12 if items[0]['mode']=='letters' else 64,
                do_sample=False,use_cache=True,pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.0)
        out=[]
        for j,i in enumerate(items):
            text=self.tokenizer.decode(seq[j,width:],skip_special_tokens=True)
            pred=parse_generated(text,i['case']['axis'],i['mode'],i['swap'])
            out.append({'id':i['case']['id'],'domain':i['case']['domain'],'axis':i['case']['axis'],
                'language':i['case']['language'],'pair_id':i['case']['pair_id'],'truth':i['case']['state'],
                'swap':i['swap'],'mode':i['mode'],'pred':pred,'valid':pred is not None,'text':text})
        return out

    def semantic_scores(self,cases,intervention=None):
        requests=[]
        for c in cases:
            for state,word in enumerate(LABELS[c['axis']]):
                for prefix in ('',' '):
                    i=self.item(c,'semantic',False,prefix+word+' |')
                    i['candidate_state']=state; requests.append(i)
        totals=defaultdict(lambda:defaultdict(list))
        for b in chunks(requests,BATCH):
            keep=max(len(i['completion']) for i in b)+1
            logits,width=self.forward(b,keep=keep,intervention=intervention)
            lp=logits.log_softmax(-1)
            for row,i in enumerate(b):
                start=width-len(i['seq'])+i['prompt_len']-1-(width-keep)
                value=sum(float(lp[row,start+k,token]) for k,token in enumerate(i['completion']))
                totals[i['case']['id']][i['candidate_state']].append(value)
        rows=[]
        def lse(v):
            m=max(v); return m+math.log(sum(math.exp(x-m) for x in v))
        for c in cases:
            a,b=[lse(totals[c['id']][s]) for s in (0,1)]
            rows.append({'id':c['id'],'domain':c['domain'],'axis':c['axis'],'language':c['language'],
                'pair_id':c['pair_id'],'truth':c['state'],'pred':0 if a>=b else 1,
                'margin':(a-b)*(1 if c['state']==0 else -1),
                'label_loglikelihoods':[a,b], 'scoring':'sum conditional log-probability through category delimiter; no length normalization'})
        return rows

    def run(self):
        allitems=[self.item(c,'letters',sw) for sw in (False,True) for c in self.cases]
        semitems=[self.item(c,'semantic') for c in self.cases]
        self.save('prompt_token_audit.json',[{'id':i['case']['id'],'mode':i['mode'],'swap':i['swap'],
            'input_ids':i['seq'],'sites':i['sites'],
            'site_tokens':{k:self.tokenizer.decode([i['seq'][v]]) for k,v in i['sites'].items()}}
            for i in allitems+semitems])
        baseline=[]; generated=[]
        self.status('measurement_baseline',statements=120)
        for n,b in enumerate(chunks(allitems,BATCH)):
            baseline+=self.letter_results(b,capture=True)
            generated+=self.generate(b)
            if n%10==9:
                self.save('baseline_letters.json',baseline); self.save('baseline_generated_letters.json',generated)
                self.status('measurement_baseline',letter_prompts_done=len(baseline),letter_prompts_total=240)
        # Capture semantic prompt representations BEFORE any teacher-forced category tokens.
        for b in chunks(semitems,BATCH): self.forward(b,capture=True)
        self.save('baseline_letters.json',baseline); self.save('baseline_generated_letters.json',generated)
        semantic=self.semantic_scores(self.cases); semantic_gen=[]
        for n,b in enumerate(chunks(semitems,BATCH)):
            semantic_gen+=self.generate(b)
            if n%5==4: self.status('semantic_baseline',done=len(semantic_gen),total=120)
        self.save('baseline_semantic_scores.json',semantic)
        self.save('baseline_semantic_generation.json',semantic_gen)
        # Padding/position check: a mixed-length, preregistered panel with both label mappings.
        audititems=[i for i in allitems if i['case']['index'] in (0,6)]
        individual=[]; grouped=[]; noop=[]; baseby={(r['id'],r['swap']):r for r in baseline}
        for i in audititems: individual+=self.letter_results([i])
        for b in chunks(list(reversed(audititems)),BATCH): grouped+=self.letter_results(b)
        for b in chunks(allitems,BATCH): noop+=self.letter_results(b,intervention=('self','both','prompt_end'))
        individualby={(r['id'],r['swap']):r for r in individual}
        deltas=[abs(r['margin']-individualby[(r['id'],r['swap'])]['margin']) for r in grouped]
        batch_flips=sum(r['pred']!=individualby[(r['id'],r['swap'])]['pred'] for r in grouped)
        no_deltas=[abs(r['margin']-baseby[(r['id'],r['swap'])]['margin']) for r in noop]
        no_flips=sum(r['pred']!=baseby[(r['id'],r['swap'])]['pred'] for r in noop)
        technical={'noop_max_margin_delta':max(no_deltas),'noop_prediction_changes':no_flips,
                   'batch_max_margin_delta':max(deltas),'batch_prediction_changes':batch_flips,
                   'pass':max(no_deltas)<=POLICY['noop_logit_atol'] and no_flips==0 and
                          max(deltas)<=POLICY['batch_margin_atol'] and batch_flips==0}
        audit=measurement_audit(self.cases,baseline,generated,semantic,semantic_gen)
        audit['technical']=technical
        self.save('measurement_audit.json',audit)
        self.save('padding_audit_rows.json',{'individual':individual,'regrouped':grouped,'noop':noop})
        print('V04_MEASUREMENT_AUDIT',json.dumps(audit,ensure_ascii=False),flush=True)
        if not technical['pass']:
            return self.finish('AUDIT_FAILED_NO_CAUSAL_SCAN',audit,[])
        self.status('head_confirmation',selected_heads=HEADS)
        effects=[]
        for site in SITES:
            for action,group in CONDITIONS:
                intervention=(action,group,site); current=[]
                for b in chunks(allitems,BATCH): current+=self.letter_results(b,intervention=intervention)
                for r in current:
                    ref=baseby[(r['id'],r['swap'])]
                    r.update({'site':site,'action':action,'group':group,
                        'baseline_pred':ref['pred'],'margin_drop':ref['margin']-r['margin'],
                        'baseline_correct':ref['pred']==ref['truth'],
                        'correct_to_wrong':ref['pred']==ref['truth'] and r['pred']!=r['truth'],
                        'wrong_to_correct':ref['pred']!=ref['truth'] and r['pred']==r['truth']})
                effects+=current
                self.save('letter_intervention_rows.json',effects)
                self.status('head_confirmation',site=site,action=action,group=group)
        # Descriptive category likelihoods for all 120 cases; direct generation on fixed sentinel panel.
        semref={r['id']:r for r in semantic}; semeffects=[]; semgenerated=[]
        sentinel=[c for c in self.cases if c['index'] in (0,6)]  # 24 statements, chosen before outputs
        for site in ('prompt_end','evidence_end'):
            for action,group in [('zero','both'),('opposite','both'),('same_state','both'),('opposite','control')]:
                iv=(action,group,site); rows=self.semantic_scores(self.cases,iv)
                for r in rows:
                    ref=semref[r['id']]
                    r.update({'site':site,'action':action,'group':group,'baseline_pred':ref['pred'],
                              'margin_drop':ref['margin']-r['margin']})
                semeffects+=rows
                if group=='both' and action in ('zero','opposite'):
                    for b in chunks([self.item(c,'semantic') for c in sentinel],BATCH):
                        gr=self.generate(b,iv)
                        for r in gr: r.update({'site':site,'action':action,'group':group})
                        semgenerated+=gr
                self.save('semantic_intervention_rows.json',semeffects)
                self.save('semantic_intervention_generation.json',semgenerated)
                self.status('semantic_confirmation',site=site,action=action,group=group)
        cache_arrays={'__'.join(map(str,k)):v.numpy() for k,v in self.cache.items()}
        self.np.savez_compressed(self.out/'selected_activation_vectors.npz',**cache_arrays)
        summaries=summarize_effects(effects,self.np)
        self.save('paired_effect_summary.json',summaries)
        return self.finish('COMPLETE_RESEARCH_ONLY',audit,summaries)

    def finish(self,status,audit,summaries):
        after=self.full_weight_digest(); self.save('weight_hashes_after.json',after)
        changed=[n for n in self.before if self.before[n]!=after[n]]
        result={'status':status,'version':VERSION,'environment':self.environment,
                'cases':120,'pairs':60,'measurement_audit':audit,
                'weights_hashed_tensor_count':len(after),'changed_weight_tensors':changed,
                'full_runtime_parameter_bytes_unchanged':not changed,
                'training_performed':False,'optimizer_created':False,'backward_called':False,
                'adapter_loaded':False,'safety_systems_modified':False,
                'limitations':['FP16 execution of BF16-source checkpoint, not bit-identical BF16 inference',
                 'synthetic paired cases, not a validated clinical benchmark',
                 'head effects on task outputs do not establish psychiatric specificity',
                 'generated explanation correctness requires independent review',
                 'generation interventions affect prompt prefill only; subsequent tokens unpatched',
                 'semantic generation intervention panel is 24 preselected statements',
                 'same-state donor differs in topic within the same domain, language, and axis'],
                'elapsed_seconds':round(time.monotonic()-self.started,2)}
        if changed: result['status']='INTEGRITY_FAILED'
        self.save('run_summary.json',result)
        self.status(result['status'])
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.out.iterdir() if p.is_file() and p.name!='artifact_sha256.json'}
        self.save('artifact_sha256.json',manifest)
        print('V04_SUMMARY',json.dumps(result,ensure_ascii=False),flush=True)
        return result


def chunks(items,size):
    for k in range(0,len(items),size): yield items[k:k+size]


def measurement_audit(cases,letters,generated,semantic,semgen):
    letterby={(r['id'],r['swap']):r for r in letters}
    genby={(r['id'],r['swap']):r for r in generated}
    semby={r['id']:r for r in semantic}; semgenby={r['id']:r for r in semgen}
    out={}
    strata=sorted({(c['domain'],c['axis'],c['language']) for c in cases})
    for d,a,l in strata:
        cs=[c for c in cases if (c['domain'],c['axis'],c['language'])==(d,a,l)]
        ll=[letterby[(c['id'],sw)] for c in cs for sw in (False,True)]
        gg=[genby[(c['id'],sw)] for c in cs for sw in (False,True)]
        validity=sum(r['valid'] for r in gg)/len(gg)
        accuracy=sum(r['pred']==r['truth'] for r in ll)/len(ll)
        swap=sum(letterby[(c['id'],False)]['pred']==letterby[(c['id'],True)]['pred'] for c in cs)/len(cs)
        agree=sum(r['valid'] and r['pred']==letterby[(r['id'],r['swap'])]['pred'] for r in gg)/len(gg)
        sg=[semgenby[c['id']] for c in cs]
        out['::'.join((d,a,l))]={'n_statements':len(cs),'letter_score_accuracy':accuracy,
            'generated_letter_accuracy_invalid_counted_wrong':sum(r['pred']==r['truth'] for r in gg)/len(gg),
            'generated_letter_validity':validity,'letter_swap_consistency':swap,
            'score_vs_generated_agreement_invalid_counted_disagree':agree,
            'mean_AB_probability_mass':statistics.mean(r['probability_mass_AB'] for r in ll),
            'semantic_score_accuracy':sum(semby[c['id']]['pred']==c['state'] for c in cs)/len(cs),
            'semantic_generated_accuracy_invalid_counted_wrong':sum(r['pred']==r['truth'] for r in sg)/len(sg),
            'semantic_generated_validity':sum(r['valid'] for r in sg)/len(sg),
            'eligible_for_label_causal_interpretation':accuracy>=POLICY['eligible_label_accuracy'] and
                validity>=POLICY['eligible_generated_label_validity'] and swap>=POLICY['eligible_swap_consistency'] and
                agree>=POLICY['eligible_generation_score_agreement']}
    return {'strata':out,'note':'Eligibility is preregistered; all rows are saved, including invalid/failed strata. No clinical deployment claims.'}


def summarize_effects(rows,np):
    groups=defaultdict(list)
    for r in rows: groups[(r['domain'],r['axis'],r['language'],r['site'],r['action'],r['group'])].append(r)
    rng=np.random.default_rng(POLICY['paired_bootstrap_seed']); out=[]
    for key,rr in sorted(groups.items()):
        pairs=defaultdict(list)
        for r in rr: pairs[r['pair_id']].append(r['margin_drop'])
        values=np.array([statistics.mean(v) for _,v in sorted(pairs.items())])
        boot=rng.choice(values,size=(POLICY['paired_bootstrap_resamples'],len(values)),replace=True).mean(1)
        correct=[r for r in rr if r['baseline_correct']]
        out.append({'domain':key[0],'axis':key[1],'language':key[2],'site':key[3],
            'action':key[4],'group':key[5],'n_independent_pairs':len(values),
            'mean_margin_drop':float(values.mean()),'pair_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'correct_to_wrong_count':sum(r['correct_to_wrong'] for r in rr),
            'wrong_to_correct_count':sum(r['wrong_to_correct'] for r in rr),
            'n_baseline_correct_label_prompts':len(correct),
            'correct_to_wrong_rate':sum(r['correct_to_wrong'] for r in correct)/len(correct) if correct else None,
            'interpretation':'exploratory interval; no multiple-comparison-adjusted significance claim'})
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true'); parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--out',default='/kaggle/working/emw_qwen_validation_v04')
    args=parser.parse_args()
    if not args.run: return
    # This local process timeout cannot terminate any other notebook or session.
    def timeout_handler(signum,frame): raise TimeoutError('V04_PROCESS_TIME_LIMIT')
    signal.signal(signal.SIGALRM,timeout_handler); signal.alarm(6900)
    try:
        experiment=Experiment(args.out)
        experiment.run()
    except Exception as e:
        out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
        (out/'failure.json').write_text(json.dumps({'error':type(e).__name__,'message':str(e),
            'traceback':traceback.format_exc(),'training_requested':False},indent=2),encoding='utf8')
        print('V04_FAILED',type(e).__name__,str(e),flush=True)
        raise
    finally: signal.alarm(0)

if __name__=='__main__': main()
