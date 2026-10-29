"""Four causal acquisition arms, frozen validation-selected SAM ANP."""
from pathlib import Path
import hashlib
import json
import pickle
import modal

app = modal.App('aims-anp-acquisition')
volume = modal.Volume.from_name('aims-neurips-data')
HERE = Path(__file__).resolve().parent
PREPARED = Path('/data/datasets/full/tabpfn-lora-benchmark-pca64-s20261018-v1')
RAW = Path('/data/datasets/full/anp-raw-bgem3-benchmark-s20261018-v1')
CHECKPOINT = Path('/data/results/anp-ood/anp-ood-bge1024-20261029-v1/sam/best.pt')
image = modal.Image.debian_slim(python_version='3.11').pip_install(
    'torch==2.5.1', index_url='https://download.pytorch.org/whl/cpu'
).pip_install('numpy>=2,<3', 'wandb>=0.25,<1').env({'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
for name in ('anp_acquisition.py','anp_crossattn_blocks.py','cnp_blocks.py','tabpfn_lora_core.py'):
    image = image.add_local_file(HERE/name, '/root/'+name)

@app.function(image=image, cpu=2, memory=8192, timeout=6*3600, volumes={'/data':volume},
              secrets=[modal.Secret.from_name('wandb', required_keys=['WANDB_API_KEY'])])
def evaluate(policy, group, smoke=False, full=False):
    import numpy as np
    import torch
    import wandb
    import anp_acquisition as core
    from anp_crossattn_blocks import AttentiveNeuralProcess
    from tabpfn_lora_core import make_bank
    torch.set_num_threads(2)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert state['best_step'] == 700 and state['config']['variant']=='sam'
    checkpoint_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    model = AttentiveNeuralProcess(1024, width=64, self_attention_blocks=1, cross_attention_heads=4, latent_dim=32)
    model.load_state_dict(state['state_dict'])
    model.eval().requires_grad_(False)
    with (PREPARED/'tasks.pkl').open('rb') as f:
        tasks = pickle.load(f)
    features = torch.from_numpy(np.load(RAW/'features.npy'))
    bank = make_bank(tasks, 'validation', 20264018, limit=3 if smoke else (0 if full else 24))
    train_ids = np.unique(np.concatenate([t['items'] for t in tasks if t['role']=='train']))
    heldout_ids = set(np.concatenate([t['items'] for t in tasks if t['role']!='train']).tolist())
    assert not heldout_ids.intersection(train_ids.tolist())
    rng = np.random.default_rng(20261018)
    reference = features[rng.choice(train_ids, 32, replace=False)]
    probes = features[rng.choice(train_ids, 32, replace=False)]
    config = {'policy':policy, 'checkpoint_sha256':checkpoint_hash, 'checkpoint_step':700,
              'split':'validation', 'tasks':len(bank), 'full_validation':full, 'stream_seeds':[17] if smoke else [17,43,89],
              'prediction':'latent prior mean (matches checkpoint selection)', 'bald_samples':16,
              'references':32, 'calibration_probes':32, 'calibration':'context-specific train-only probe score upper-tail rank',
              'budgets':core.BUDGETS, 'smoke':smoke,
              'code_sha256':hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()}
    identity = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    output = Path('/data/results/anp-acquisition')/group/policy
    output.mkdir(parents=True, exist_ok=True)
    done = output/'completed.json'
    if done.exists():
        old=json.loads(done.read_text()); assert old['identity']==identity; return old
    wb = wandb.init(project='aims-predictive-evaluation',group=group,name='acquisition-'+policy,
                    id='acq-'+hashlib.sha256((group+'/'+policy).encode()).hexdigest()[:18],resume='allow',config=config,
                    tags=['acquisition','frozen-anp','validation',policy])
    progress = output/'progress.json'
    rows=[]
    if progress.exists():
        saved=json.loads(progress.read_text()); assert saved['identity']==identity; rows=saved['rows']
    done_keys={(r['task'],r['stream_seed']) for r in rows}
    for ep in bank:
        task=tasks[ep['task']]
        for seed in config['stream_seeds']:
            if (ep['task'],seed) in done_keys:
                continue
            stream=np.random.default_rng(seed+ep['task']).permutation(ep['acquisition'])
            row=core.trajectory(model,features,task['items'],stream,ep['targets'],
                    lambda i: task['labels'][i],task['labels'][ep['targets']],policy,reference,probes,seed+ep['task'])
            row.update(task=ep['task'],stream_seed=seed,subject=task['subject'],benchmark=task['benchmark'],weight=ep['weight'])
            rows.append(row)
            progress.write_text(json.dumps({'identity':identity,'rows':rows})); volume.commit()
            summary=core.aggregate(rows)
            wb.log({'episodes':len(rows),'acquisition/brier_alc':summary['brier_alc'],
                    'acquisition/forced_purchases':summary['mean_forced_purchases'],
                    **{f'acquisition/brier_budget_{k}':v for k,v in summary['by_budget'].items()}})
            print(f'{policy}: {len(rows)}/{len(bank)*len(config["stream_seeds"])} ALC={summary["brier_alc"]:.6f}',flush=True)
    result={'identity':identity,'config':config,'wandb_url':wb.url,**core.aggregate(rows)}
    done.write_text(json.dumps(result,indent=2)); volume.commit()
    wb.summary['final/brier_alc']=result['brier_alc']; wb.finish()
    return {k:v for k,v in result.items() if k!='tasks'}

@app.function(cpu=.125,memory=256,timeout=8*3600)
def workflow(group, full=False):
    policies=('random','uncertainty','expected_brier') if full else ('random','uncertainty','bald','expected_brier')
    checks=[evaluate.spawn(p,group+'-smoke',True) for p in policies]
    for call in checks:
        call.get()
    calls=[evaluate.spawn(p,group,full=full) for p in policies]
    for call in calls:
        print(json.dumps(call.get()),flush=True)

@app.local_entrypoint()
def main(group: str='anp-sam-acquisition-full-20261029-v2', full: bool=True):
    print(workflow.spawn(group,full).object_id)
