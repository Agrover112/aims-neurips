"""Fetch small OOD/acquisition JSONs and paired, benchmark-level differences."""
import hashlib
import json
from pathlib import Path
import modal
from modal.config import config

client=modal.Client.from_credentials(config.get('token_id',profile='arshiaswe',use_env=False),
                                     config.get('token_secret',profile='arshiaswe',use_env=False))
volume=modal.Volume.from_name('aims-neurips-data').hydrate(client)
output=Path(__file__).resolve().parents[1]/'reports'/'anp-acquisition-20261029-v1'
def fetch(remote, local):
    data=b''.join(volume.read_file(remote))
    path=output/local; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    return json.loads(data)

training=[]
for name in ('balanced_erm','groupdro','swa','sam'):
    base=f'/results/anp-ood/anp-ood-bge1024-20261029-v1/{name}/'
    c=fetch(base+'completed.json',Path('training')/name/'completed.json')
    h=fetch(base+'curves.json',Path('training')/name/'curves.json')
    f=fetch(base+'final-validation.json',Path('training')/name/'final-validation.json')
    training.append(dict(variant=name,best_step=c['best_step'],selection_alc=c['best_validation_alc'],
                         full_validation_alc=f['brier_alc'],last_selection_alc=h[-1]['validation/brier_alc']))
print(json.dumps({'training':training},indent=2))
results={}
for name in ('random','uncertainty','bald','expected_brier'):
    base=f'/results/anp-acquisition/anp-sam-acquisition-20261029-v1/{name}/'
    try:
        results[name]=fetch(base+'completed.json',Path(name)/'completed.json')
    except (modal.exception.NotFoundError, FileNotFoundError):
        try:
            progress=fetch(base+'progress.json',Path(name)/'progress.json')
            print(f'{name}: {len(progress["rows"])}/72 complete')
        except (modal.exception.NotFoundError, FileNotFoundError):
            print(f'{name}: waiting for first full-pilot trajectory')
if len(results)==4:
    random=results['random']; reference={(r['task'],r['stream_seed']):r for r in random['tasks']}
    summary=[]
    for name,result in results.items():
        assert result['config']['checkpoint_sha256']==random['config']['checkpoint_sha256']
        assert {(r['task'],r['stream_seed']) for r in result['tasks']}==set(reference)
        for row in result['tasks']:
            assert abs(row['by_budget']['0']-reference[row['task'],row['stream_seed']]['by_budget']['0'])<1e-9
        summary.append(dict(policy=name,alc=result['brier_alc'],delta_vs_random=result['brier_alc']-random['brier_alc'],
                            by_budget=result['by_budget'],runtime_seconds=result['total_runtime_seconds'],
                            mean_forced_purchases=result['mean_forced_purchases'],
                            benchmark_deltas={b:v-random['by_benchmark'][b] for b,v in result['by_benchmark'].items()}))
    (output/'summary.json').write_text(json.dumps({'training':training,'acquisition':summary},indent=2))
    print(json.dumps(summary,indent=2))
