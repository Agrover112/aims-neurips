"""Replay evaluated decisions through the packaged streaming hook."""
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
stage=ROOT/'dist'/'anp-sa1-w64-bgem3-benchmarkholdout-sam-step700-expected-brier'
sys.path.insert(0,str(stage)); sys.path.insert(1,str(ROOT))
os.environ['PREDICTIVE_EVAL_LOCAL_SMOKE_TEST']='1'
os.environ['OMP_NUM_THREADS']='2'
runtime=importlib.import_module('aims_predictor.predictor')
labeling=importlib.import_module('labeling')
from organized.training.anp_acquisition import trajectory

predictor=runtime.NeuralPredictor(); runtime._PREDICTOR=predictor
rng=np.random.default_rng(42)
vectors=rng.normal(size=(100,1024)).astype(np.float32)
vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
labels=rng.integers(0,2,100)
bank=np.load(stage/'assets'/'acquisition-bank.npz')
vectors=np.tile(np.concatenate([bank['reference'],bank['probes']]),(2,1))[:100].copy()
features=torch.from_numpy(vectors)
ref=torch.from_numpy(bank['reference']); probes=torch.from_numpy(bank['probes'])
subject={'normalized_name':'replay-subject'}
items=[{'benchmark_id':'replay-benchmark','item_content':str(i)} for i in range(100)]
def encode(texts):
    return np.stack([vectors[int(t)] for t in texts])
predictor.embedder.encode_many=encode
summary=[]
for seed in (17,43):
    expected=trajectory(predictor.model,features,np.arange(100),np.arange(50),np.arange(50,100),
        lambda i:labels[i],labels[50:],'expected_brier',ref,probes,seed)
    uniforms=np.random.default_rng(seed).random(50)
    labeling._uniform=lambda pair,context: uniforms[50-context['items_remaining']]
    evidence=[]; positions=[]
    labeling._CALIBRATION.clear()
    for position in range(50):
        context={'labels_remaining':31-len(positions),'items_remaining':50-position,
                 'labels_acquired':len(positions),'max_labels':31,'subject_id':'replay','benchmark_id':'replay-benchmark'}
        # Unrelated task evidence must not enter the posterior or cache key.
        unrelated=[[[{'normalized_name':'other'},items[99]],1]]
        if position==10:
            labeling._CALIBRATION.clear()  # loss of worker-local cache cannot alter policy
        decision=labeling.acquisition_function([subject,items[position]],labeled=evidence+unrelated,context=context)
        assert type(decision) is bool
        if decision:
            positions.append(position+1)
            evidence.append([[subject,items[position]],int(labels[position])])
        if len(positions)==31:
            break
    assert positions==expected['purchase_positions'],(positions,expected['purchase_positions'])
    assert labeling.acquisition_function([subject,items[0]],context={'labels_remaining':0,'items_remaining':5}) is False
    assert labeling.acquisition_function([subject,items[0]],context={'labels_remaining':3,'items_remaining':3}) is True
    summary.append({'seed':seed,'matched_purchases':len(positions),'forced_purchases':expected['forced_purchases']})
report={'passed':True,'replays':summary,'checks':['offline decision parity','unrelated evidence isolation','cache reset invariance','budget limits']}
(ROOT/'dist'/'expected-brier-replay-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
