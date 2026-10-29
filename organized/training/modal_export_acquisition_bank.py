"""Export exactly the training-only reference/probe bank used in evaluation."""
import io
import json
from pathlib import Path
import modal

app=modal.App('aims-export-acquisition-bank')
volume=modal.Volume.from_name('aims-neurips-data')
image=modal.Image.debian_slim(python_version='3.11').pip_install('numpy>=2,<3')

@app.function(image=image,cpu=1,memory=4096,timeout=300,volumes={'/data':volume})
def export():
    import hashlib
    import pickle
    import numpy as np
    prepared=Path('/data/datasets/full/tabpfn-lora-benchmark-pca64-s20261018-v1')
    raw=Path('/data/datasets/full/anp-raw-bgem3-benchmark-s20261018-v1')
    with (prepared/'tasks.pkl').open('rb') as f:
        tasks=pickle.load(f)
    ids=np.unique(np.concatenate([t['items'] for t in tasks if t['role']=='train']))
    heldout=np.unique(np.concatenate([t['items'] for t in tasks if t['role']!='train']))
    assert not np.intersect1d(ids,heldout).size
    rng=np.random.default_rng(20261018)
    ref=rng.choice(ids,32,replace=False); probes=rng.choice(ids,32,replace=False)
    features=np.load(raw/'features.npy',mmap_mode='r')
    stream=io.BytesIO()
    np.savez(stream,reference=np.asarray(features[ref]),probes=np.asarray(features[probes]),reference_ids=ref,probe_ids=probes)
    data=stream.getvalue()
    return data,{'seed':20261018,'reference_count':32,'probe_count':32,'training_only':True,
        'bank_sha256':hashlib.sha256(data).hexdigest(),'raw_features_sha256':json.loads((raw/'ready.json').read_text())['features_sha256']}

@app.local_entrypoint()
def main():
    data,metadata=export.remote()
    target=Path(__file__).resolve().parents[2]/'organized'/'submission'/'assets'
    (target/'sam-acquisition-bank.npz').write_bytes(data)
    (target/'sam-acquisition-bank.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
