"""Expected prediction-change score for any binary context predictor.

Only numpy is required. The callback must reconstruct each supplied context,
including hypothetical labels, without retaining fit state between calls.
"""
import hashlib
import json
import numpy as np

def _probabilities(predict, xc, yc, xt):
    p=np.asarray(predict(xc,yc,xt),dtype=float)
    if p.shape!=(len(xt),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('predict must return one finite success probability per query')
    return p

def expected_brier_scores(predict, xc, yc, candidates, reference):
    """Return approximate expected Brier reduction per candidate (higher wins).

    Inputs are already transformed feature matrices; yc contains acquired
    binary outcomes only. References are permitted unlabeled training items.
    This is prediction change, not a guaranteed Bayesian risk reduction.
    """
    xc,yc,candidates,reference=map(np.asarray,(xc,yc,candidates,reference))
    if xc.ndim!=2 or candidates.ndim!=2 or reference.ndim!=2 or not len(reference):
        raise ValueError('Expected 2D feature matrices and a nonempty reference bank')
    if len(yc)!=len(xc) or yc.ndim!=1 or not np.isin(yc,[0,1]).all():
        raise ValueError('Context labels must be a matching 1D binary array')
    if xc.shape[1]!=candidates.shape[1] or xc.shape[1]!=reference.shape[1]:
        raise ValueError('Use the same embedding/preprocessing contract for every input')
    if not len(candidates):
        return np.empty(0)
    probabilities=_probabilities(predict,xc,yc,candidates)
    baseline=_probabilities(predict,xc,yc,reference)
    scores=[]
    for x,p in zip(candidates,probabilities):
        augmented=np.concatenate([xc,x[None,:]])
        changes=[]
        for label in (0,1):
            updated=_probabilities(predict,augmented,np.r_[yc,label],reference)
            changes.append(np.mean((updated-baseline)**2))
        scores.append((1-p)*changes[0]+p*changes[1])
    return np.asarray(scores)

def streaming_accept(value, calibration, labels_remaining, items_remaining, uniform):
    """Same randomized probe-tail pacing as the evaluated ANP policy."""
    b,r=int(labels_remaining),int(items_remaining)
    if b<=0:
        return False
    if r<=0:
        raise ValueError('items_remaining must include the current candidate')
    if b>=r:
        return True
    a=np.asarray(calibration,dtype=float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or not np.isfinite(value):
        raise ValueError('Finite calibration scores are required')
    if not 0<=uniform<1:
        raise ValueError('uniform must be in [0,1)')
    tail=(np.count_nonzero(a>value)+uniform*(1+np.count_nonzero(a==value)))/(len(a)+1)
    return bool(tail<=b/r)

def stable_uniform(input,context):
    payload=json.dumps([input,context],sort_keys=True,separators=(',',':'),default=str).encode()
    return int.from_bytes(hashlib.blake2b(payload,digest_size=8).digest(),'big')/2**64
