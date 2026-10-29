"""Train-episode empirical Bayes, frozen means, validation-only evaluation."""
import hashlib
import json
import pickle
from pathlib import Path
import modal

app=modal.App('aims-bayesian-adapters')
volume=modal.Volume.from_name('aims-neurips-data')
HERE=Path(__file__).resolve().parent
PREPARED=Path('/data/datasets/full/tabpfn-lora-benchmark-pca64-s20261018-v1')
RAW=Path('/data/datasets/full/anp-raw-bgem3-benchmark-s20261018-v1')
ANP=Path('/data/results/anp-ood/anp-ood-bge1024-20261029-v1/sam/best.pt')
RASCH=Path('/data/results/statistical-baselines/statistical-benchmark-20261029-v1/feature_rasch/best.pkl')
image=modal.Image.debian_slim(python_version='3.11').pip_install(
    'torch==2.5.1',index_url='https://download.pytorch.org/whl/cpu'
).pip_install('numpy>=2,<3','scipy>=1.11,<2','scikit-learn>=1.5,<2','wandb>=0.25,<1').env(
    {'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
for name in ('bayesian_adapters.py','anp_crossattn_blocks.py','cnp_blocks.py','tabpfn_lora_core.py'):
    image=image.add_local_file(HERE/name,'/root/'+name)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@app.function(image=image,cpu=2,memory=8192,timeout=6*3600,volumes={'/data':volume},
              secrets=[modal.Secret.from_name('wandb',required_keys=['WANDB_API_KEY'])])
def run(kind,group,smoke=False):
    import numpy as np
    import torch
    import wandb
    import bayesian_adapters as core
    from tabpfn_lora_core import make_bank
    from anp_crossattn_blocks import AttentiveNeuralProcess
    torch.set_num_threads(2)
    assert kind in ('anp_correction','gp_classifier')
    with (PREPARED/'tasks.pkl').open('rb') as f:
        tasks=pickle.load(f)
    train_bank=make_bank(tasks,'train',20267018,limit=11 if smoke else 66)
    # Hyperparameter fitting sees only disjoint support/query from train roles.
    for ep in train_bank:
        ep['targets']=ep['targets'][:64]
    bank=make_bank(tasks,'validation',20264018,limit=3 if smoke else 0)
    output=Path('/data/results/bayesian-adapters')/group/kind
    output.mkdir(parents=True,exist_ok=True)
    if kind=='anp_correction':
        source=ANP; source_hash=sha(source)
        state=torch.load(source,map_location='cpu',weights_only=False)
        assert state['best_step']==700
        model=AttentiveNeuralProcess(1024,width=64,self_attention_blocks=1,cross_attention_heads=4,latent_dim=32)
        model.load_state_dict(state['state_dict']); model.eval().requires_grad_(False)
        cache=output/'zero-logits.npz'
        if cache.exists():
            cached=np.load(cache); assert str(cached['source_hash'])==source_hash
            mean=cached['logits']
        else:
            features=torch.from_numpy(np.load(RAW/'features.npy'))
            values=[]
            with torch.inference_mode():
                for start in range(0,len(features),512):
                    values.append(model.logits(features[:0],torch.empty(0),features[start:start+512]).numpy())
            mean=np.concatenate(values).astype(float)
            np.savez(cache,logits=mean,source_hash=source_hash); volume.commit()
        configs=[{'offset_sd':u,'slope_sd':v} for u in (.5,1.,2.) for v in (.25,.5)]
        def predict(params,ci,y,ti):
            return core.correction(mean[ci],y,mean[ti],**params)
        features=None
    else:
        source=RASCH; source_hash=sha(source)
        with source.open('rb') as f:
            saved=pickle.load(f)
        # Fixed text-dependent mean from the existing training-fitted Rasch.
        # Its training fraction was previously selected on validation; recorded.
        mean=saved['state']['item_easiness']
        raw=np.load(PREPARED/'features.npy')
        features=(raw-saved['transform']['mean'])/saved['transform']['scale']
        configs=[{'amplitude':a,'lengthscale':ell,'offset_sd':1.} for a in (.25,1.,2.) for ell in (.5,1.,2.)]
        def predict(params,ci,y,ti):
            return core.gp_predict(features[ci],y,mean[ci],features[ti],mean[ti],**params)
    if smoke:
        configs=configs[:1]
    config={'kind':kind,'smoke':smoke,'source':str(source),'source_sha256':source_hash,
            'tasks_sha256':sha(PREPARED/'tasks.pkl'),'core_sha256':sha(core.__file__),
            'prepared':json.loads((PREPARED/'prepared.json').read_text()),
            'hyperparameter_selection':'train support/query Brier ALC only',
            'train_episodes':len(train_bank),'validation_episodes':len(bank),'configs':configs,
            'test_evaluated':False,'mean_frozen':True,
            'gp_mean_note':'reuse validation-selected feature Rasch; new kernel settings fit only on train episodes' if kind=='gp_classifier' else None}
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    done=output/'completed.json'
    if done.exists():
        result=json.loads(done.read_text()); assert result['identity']==identity
        return {k:v for k,v in result.items() if k!='tasks'}
    wb=wandb.init(project='aims-predictive-evaluation',group=group,name=kind,
        id='bayes-'+hashlib.sha256((group+'/'+kind).encode()).hexdigest()[:18],resume='allow',config=config,
        tags=['bayesian','benchmark-heldout','validation',kind])
    progress=output/'last.json'
    history=[]
    if progress.exists():
        old=json.loads(progress.read_text()); assert old['identity']==identity; history=old['history']
    for index in range(len(history),len(configs)):
        params=configs[index]
        score=core.evaluate(lambda ci,y,ti:predict(params,ci,y,ti),tasks,train_bank)
        history.append({'parameters':params,'training_query_alc':score['brier_alc']})
        progress.write_text(json.dumps({'identity':identity,'config':config,'history':history})); volume.commit()
        wb.log({'fit/trial':index+1,'fit/training_query_alc':score['brier_alc']})
        print(f'{kind}: fit {index+1}/{len(configs)} ALC={score["brier_alc"]:.6f}',flush=True)
    selected=min(history,key=lambda h:h['training_query_alc'])['parameters']
    (output/'best.json').write_text(json.dumps({'identity':identity,'parameters':selected,'config':config}))
    # Complete inference artifact: fixed item logits; GP additionally needs PCA features.
    np.savez(output/'inference-state.npz',mean_logits=mean,features=np.empty((0,0)) if features is None else features)
    result=core.evaluate(lambda ci,y,ti:predict(selected,ci,y,ti),tasks,bank)
    # Preserve familiar diagnostic keys; final diagnostics, not training curves.
    from sklearn.metrics import roc_auc_score
    diagnostic={}
    for role,population in (('train','training_subjects'),('validation','heldout_subjects')):
        db=make_bank(tasks,role,20266018,limit=11 if smoke and role=='train' else (3 if smoke else 32))
        for axis in ('same_subject_split_half','same_subject_response_holdout'):
            rows=[]
            for ep in db:
                task=tasks[ep['task']]; order=np.r_[ep['acquisition'],ep['targets']]
                cut=len(ep['acquisition']) if axis=='same_subject_split_half' else int(.8*len(order))
                c,t=order[:cut][:64],order[cut:][:512]
                p=predict(selected,task['items'][c],task['labels'][c],task['items'][t]); y=task['labels'][t]
                rows.append((float(np.mean((p-y)**2)),float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,ep['weight']))
            prefix=f'diagnostic/{population}/{axis}'
            diagnostic[prefix+'/brier_score']=float(np.average([r[0] for r in rows],weights=[r[2] for r in rows]))
            valid=[r for r in rows if r[1] is not None]
            if valid:
                diagnostic[prefix+'/roc_auc']=float(np.average([r[1] for r in valid],weights=[r[2] for r in valid]))
    result.update(identity=identity,config=config,selected_parameters=selected,history=history,diagnostics=diagnostic,wandb_url=wb.url)
    done.write_text(json.dumps(result,indent=2)); volume.commit()
    wb.log({'validation/brier_alc':result['brier_alc'],**{f'validation/brier_budget_{b}':v for b,v in result['by_budget'].items()},**diagnostic})
    wb.summary['final/full_validation_alc']=result['brier_alc']; wb.finish()
    print(f'{kind}: validation_ALC={result["brier_alc"]:.6f}',flush=True)
    return {k:v for k,v in result.items() if k!='tasks'}

@app.function(cpu=.125,memory=256,timeout=8*3600)
def workflow(group):
    checks=[run.spawn(k,group+'-smoke',True) for k in ('anp_correction','gp_classifier')]
    for call in checks:
        call.get()
    calls=[run.spawn(k,group) for k in ('anp_correction','gp_classifier')]
    for call in calls:
        print(json.dumps(call.get()),flush=True)

@app.local_entrypoint()
def main(group: str='bayesian-adapters-20261029-v1'):
    print(workflow.spawn(group).object_id)
