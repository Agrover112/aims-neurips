"""Causal expected Brier-change acquisition with training-only probe pacing."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from aims_predictor import predictor as runtime
from aims_predictor.acquisition_core import score, accept

_BANK=None
_CALIBRATION=OrderedDict()

def _uniform(input, context):
    payload=json.dumps([input,context],sort_keys=True,separators=(',',':'),default=str).encode()
    return int.from_bytes(hashlib.blake2b(payload,digest_size=8).digest(),'big')/2**64

def acquisition_function(input, prediction=None, labeled=None, context=None):
    global _BANK
    if context is None:
        raise ValueError('Expected Brier acquisition requires streaming context')
    b,r=int(context['labels_remaining']),int(context['items_remaining'])
    if b<=0:
        return False
    if r<=b:
        return True
    if runtime._PREDICTOR is None:
        runtime._PREDICTOR=runtime.NeuralPredictor()
    predictor=runtime._PREDICTOR
    subject,item=input
    subject_key=runtime._subject_key(subject)
    benchmark=str(item.get('benchmark_id','') or '')
    evidence=[(i,y) for s,i,y in runtime._evidence(labeled)
              if runtime._subject_key(s)==subject_key and str(i.get('benchmark_id','') or '')==benchmark][-runtime.MAX_CONTEXT:]
    vectors=predictor.embedder.encode_many([runtime._item_text(item)]+[runtime._item_text(i) for i,y in evidence])
    xc=torch.from_numpy(vectors[1:]); yc=torch.tensor([y for i,y in evidence],dtype=torch.float32)
    if _BANK is None:
        with np.load(Path(__file__).parent/'assets'/'acquisition-bank.npz',allow_pickle=False) as bank:
            _BANK=(torch.from_numpy(bank['reference'].copy()),torch.from_numpy(bank['probes'].copy()))
    reference,probes=_BANK
    key=hashlib.sha256(xc.numpy().tobytes()+yc.numpy().tobytes()).digest()
    if key not in _CALIBRATION:
        _CALIBRATION[key]=score(predictor.model,xc,yc,probes,'expected_brier',reference,None).numpy()
        if len(_CALIBRATION)>16:
            _CALIBRATION.popitem(last=False)
    else:
        _CALIBRATION.move_to_end(key)
    value=float(score(predictor.model,xc,yc,torch.from_numpy(vectors[:1]),'expected_brier',reference,None)[0])
    calibration=_CALIBRATION[key]
    if not np.isfinite(value) or not np.isfinite(calibration).all():
        raise RuntimeError('Nonfinite expected Brier score')
    return bool(accept(value,calibration,b,r,_uniform(input,context)))
