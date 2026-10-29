"""Real BGE-M3 cold/warm inference check of the upload archive."""
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents)>2 else Path('/root')
NAME='anp-sa1-w64-bgem3-benchmarkholdout-sam-step700-expected-brier'
app=modal.App('aims-sam-package-check')
volume=modal.Volume.from_name('aims-neurips-data')
image=modal.Image.debian_slim(python_version='3.11').pip_install(
    'torch==2.5.1',index_url='https://download.pytorch.org/whl/cpu'
).pip_install('numpy>=2,<3','transformers==4.48.3','sentencepiece').env(
    {'HF_HOME':'/data/huggingface-cache','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
)
if Path(__file__).resolve().parent != Path('/root'):
    image=image.add_local_file(ROOT/'dist'/(NAME+'.zip'),'/opt/submission.zip',copy=True)

@app.function(image=image,cpu=2,memory=8192,timeout=1200,volumes={'/data':volume})
def validate():
    import importlib
    import os
    import sys
    import time
    import zipfile
    import numpy as np
    from huggingface_hub import HfApi,snapshot_download
    api=HfApi(); revision=api.model_info('BAAI/bge-m3').sha
    files=api.list_repo_files('BAAI/bge-m3',revision=revision)
    weight='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
    snapshot=snapshot_download('BAAI/bge-m3',revision='main',allow_patterns=[weight,'config.json','tokenizer*','special_tokens_map.json','sentencepiece.bpe.model'])
    assert Path(snapshot).name==revision
    volume.commit()
    root=Path('/tmp/submission'); root.mkdir()
    with zipfile.ZipFile('/opt/submission.zip') as z:
        z.extractall(root)
    sys.path.insert(0,str(root))
    os.environ.pop('PREDICTIVE_EVAL_LOCAL_SMOKE_TEST',None)
    os.environ['AIMS_EMB_CACHE_DIR']='/tmp/check-embedding-cache'
    p=importlib.import_module('aims_predictor.predictor')
    predictor=p.NeuralPredictor()
    subject={'normalized_name':'submission-check'}
    target={'benchmark_id':'check','item_content':'What is the capital of France?'}
    evidence=[[subject,{'benchmark_id':'check','item_content':'What is two plus two?'}],1]
    start=time.monotonic()
    cold=predictor.predict([subject,target],[evidence])
    cold_time=time.monotonic()-start
    reference=predictor.embedder.encode_many([target['item_content']])[0]
    assert reference.shape==(1024,) and np.isclose(np.linalg.norm(reference),1.,atol=1e-5)
    second=p.NeuralPredictor()
    def fail():
        raise AssertionError('Warm disk-cache path tried to load BGE')
    second.embedder._load=fail
    start=time.monotonic()
    warm=second.predict([subject,target],[evidence])
    warm_time=time.monotonic()-start
    assert abs(cold-warm)<1e-7 and second.embedder.model is None
    assert 0<=cold<=1 and p._FALLBACK_CALLS==0 and not p._PREDICTOR_FAILED
    labeling=importlib.import_module('labeling')
    p._PREDICTOR=predictor
    labels=[]; positions=[]; timings=[]
    for position in range(50):
        item={'benchmark_id':'check','item_content':f'Compute {position+1} plus {position+2}. Give the integer answer.'}
        ctx={'subject_id':'check','benchmark_id':'check','labels_remaining':31-len(labels),
             'labels_acquired':len(labels),'max_labels':31,'items_remaining':50-position}
        probability=predictor.predict([subject,item],labels)
        start=time.monotonic()
        decision=labeling.acquisition_function([subject,item],prediction=probability,labeled=labels,context=ctx)
        timings.append(time.monotonic()-start)
        assert type(decision) is bool
        if decision:
            labels.append([[subject,item],position%2]); positions.append(position+1)
        if len(labels)==31:
            break
    assert len(labels)==31
    # Replaying a nonforced decision with only disk embeddings must also work
    # when both predictor and calibration caches are reconstructed.
    p._PREDICTOR=second; labeling._CALIBRATION.clear()
    c={'labels_remaining':31,'items_remaining':50,'labels_acquired':0,'max_labels':31,'subject_id':'check','benchmark_id':'check'}
    first_item={'benchmark_id':'check','item_content':'Compute 1 plus 2. Give the integer answer.'}
    replay=labeling.acquisition_function([subject,first_item],labeled=[],context=c)
    assert replay == (1 in positions) and second.embedder.model is None
    return {'real_encoder':'BAAI/bge-m3','encoder_revision':revision,'probability':cold,
            'cold_seconds':cold_time,'warm_seconds':warm_time,'warm_worker_loaded_encoder':False,'fallback_calls':0,
            'acquisition_candidates':len(timings),'labels_acquired':len(labels),'purchase_positions':positions,
            'acquisition_total_seconds':sum(timings),'acquisition_max_call_seconds':max(timings),
            'acquisition_mean_call_seconds':sum(timings)/len(timings),'restart_decision_matched':True}

@app.local_entrypoint()
def main():
    import json
    result=validate.remote()
    path=ROOT/'dist'/(NAME+'-real-check.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
