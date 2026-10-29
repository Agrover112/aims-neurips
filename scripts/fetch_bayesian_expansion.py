"""Fetch completed Bayesian models and full acquisition progress/results."""
import json
from pathlib import Path
import modal
from modal.config import config

client=modal.Client.from_credentials(config.get('token_id',profile='arshiaswe',use_env=False),
                                     config.get('token_secret',profile='arshiaswe',use_env=False))
volume=modal.Volume.from_name('aims-neurips-data').hydrate(client)
output=Path(__file__).resolve().parents[1]/'reports'/'bayesian-expansion-20261029-v1'
output.mkdir(parents=True,exist_ok=True)
def fetch(remote,local):
    data=b''.join(volume.read_file(remote)); path=output/local
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    return json.loads(data)

summary={}
for kind in ('anp_correction','gp_classifier'):
    try:
        result=fetch(f'/results/bayesian-adapters/bayesian-adapters-20261029-v1/{kind}/completed.json',Path(kind)/'completed.json')
        summary[kind]={k:result[k] for k in ('brier_alc','by_budget','by_benchmark','selected_parameters','wandb_url')}
    except FileNotFoundError:
        summary[kind]={'status':'running'}
arms={}
for kind in ('random','uncertainty','expected_brier'):
    root=f'/results/anp-acquisition/anp-sam-acquisition-full-20261029-v2/{kind}/'
    try:
        result=fetch(root+'completed.json',Path('acquisition')/kind/'completed.json'); arms[kind]=result
        summary['acquisition_'+kind]={k:result[k] for k in ('brier_alc','by_budget','by_benchmark','benchmark_macro_alc','mean_forced_purchases','total_runtime_seconds','wandb_url')}
    except FileNotFoundError:
        try:
            p=fetch(root+'progress.json',Path('acquisition')/kind/'progress.json')
            summary['acquisition_'+kind]={'status':'running','episodes_completed':len(p['rows']),'episodes_total':1170}
        except FileNotFoundError:
            summary['acquisition_'+kind]={'status':'waiting'}
if len(arms)==3:
    baseline=arms['random']; keys={(r['task'],r['stream_seed']) for r in baseline['tasks']}
    zero={(r['task'],r['stream_seed']):r['by_budget']['0'] for r in baseline['tasks']}
    for kind,result in arms.items():
        assert {(r['task'],r['stream_seed']) for r in result['tasks']}==keys
        # CPU float32 kernels can differ by a few ulps across containers.
        assert all(abs(r['by_budget']['0']-zero[r['task'],r['stream_seed']])<1e-7 for r in result['tasks'])
        summary['acquisition_'+kind]['delta_vs_random']=result['brier_alc']-baseline['brier_alc']
        summary['acquisition_'+kind]['benchmark_deltas']={b:v-baseline['by_benchmark'][b] for b,v in result['by_benchmark'].items()}
(output/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
