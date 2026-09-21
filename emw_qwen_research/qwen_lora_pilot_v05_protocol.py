"""Frozen v0.5 entry point, including pre-run evaluation metadata balancing.
This module imports the pinned engine next to it. It changes dataset construction
BEFORE baseline or training, not model outputs, gold status labels or checkpoints.
The completed experiment must be cited with both engine and protocol revisions.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from collections import Counter

ENGINE_SHA256='16980b9b8ac0d4d918440f3187e33ce34e211c4860deb44754b7d101e8fcd9f1'
engine=Path(__file__).with_name('qwen_lora_pilot_v05.py')
assert hashlib.sha256(engine.read_bytes()).hexdigest()==ENGINE_SHA256
spec=importlib.util.spec_from_file_location('emw_v05_engine',engine)
core=importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
original_build=core.build_data


def balanced_data():
    data=original_build()
    for split in ('dev','test'):
        for row in data[split]:
            # In the original generic generator, family=idx%4 accidentally made
            # all test timeframes current. Independently vary metadata here.
            index=int(row['group_id'].split('_')[2])
            frame=index%4
            source=(['patient','collateral','clinician','patient'][frame]
                    if row['domain']=='clinical' else 'inspector')
            timeframe='current' if frame%2==0 else 'historical'
            previous=row['expected']
            old=f"Timeframe explicitly specified for this target: {previous['timeframe']}."
            new=f'Timeframe explicitly specified for this target: {timeframe}.'
            assert row['input'].count(old)==1
            row['input']=row['input'].replace(old,new)
            old=f"Author of the focal statement: {previous['source']}."
            new=f'Author of the focal statement: {source}.'
            assert row['input'].count(old)==1
            row['input']=row['input'].replace(old,new)
            row['expected']={**previous,'source':source,'timeframe':timeframe}
            row['assistant']=json.dumps(row['expected'],ensure_ascii=False,separators=(',',':'))
    core.validate_data(data)
    for domain in ('clinical','control'):
        rows=[r for r in data['test'] if r['domain']==domain]
        assert Counter(r['expected']['timeframe'] for r in rows)=={'current':24,'historical':24}
        assert Counter(r['expected']['status'] for r in rows)==dict.fromkeys(core.STATES,12)
    assert {r['expected']['source'] for r in data['test'] if r['domain']=='clinical'}=={'patient','collateral','clinician'}
    return data


core.build_data=balanced_data
if __name__=='__main__':
    if '--self-test' in sys.argv:
        core.self_test()
        data=balanced_data()
        print(json.dumps({'protocol':'PASS','hashes':{k:core.sha(v) for k,v in data.items()},
          'test_timeframes':dict(Counter(r['expected']['timeframe'] for r in data['test']))}))
    elif '--run' in sys.argv:
        import torch
        torch.set_num_threads(2)
        core.self_test()
        manifest={'engine_sha256':ENGINE_SHA256,
                  'protocol_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'all_data_constructed_before_baseline_or_training':True,
                  'no_previous_kaggle_outputs_loaded':True,
                  'english_only':True,
                  'run_type':'single-seed fixed-step development pilot; no clinical deployment'}
        core.write_json(Path('/kaggle/working/v05_launch_manifest.json'),manifest)
        core.launch('/kaggle/working/emw_qwen_lora_pilot_v05')
    else:
        print('Use --self-test or --run explicitly.')
