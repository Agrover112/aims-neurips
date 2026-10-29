"""Stage the evaluated expected-Brier policy alongside the selected SAM ANP."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NAME='anp-sa1-w64-bgem3-benchmarkholdout-sam-step700-expected-brier'
source=ROOT/'dist'/'anp-sa1-w64-bgem3-benchmarkholdout-sam-step700-random'
stage=ROOT/'dist'/NAME
shutil.copytree(source,stage,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
submission=ROOT/'organized'/'submission'
shutil.copy2(submission/'aims_predictor'/'predictor.py',stage/'aims_predictor'/'predictor.py')
shutil.copy2(submission/'expected_brier_labeling.py',stage/'labeling.py')
shutil.copy2(ROOT/'organized'/'training'/'anp_acquisition.py',stage/'aims_predictor'/'acquisition_core.py')
shutil.copy2(submission/'assets'/'sam-acquisition-bank.npz',stage/'assets'/'acquisition-bank.npz')
bank=json.loads((submission/'assets'/'sam-acquisition-bank.json').read_text())
assert hashlib.sha256((stage/'assets'/'acquisition-bank.npz').read_bytes()).hexdigest()==bank['bank_sha256']
metadata=json.loads((stage/'assets'/'metadata.json').read_text())
metadata['acquisition']='expected_brier_change_with_train_probe_rank_pacing'
metadata['acquisition_bank']=bank
metadata['provenance']['expected_brier_acquisition']=True
metadata['acquisition_validation']={'tasks':390,'stream_orders':3,'brier_alc':.19054668554895315,
    'random_brier_alc':.1937440300836548,'uniform_ties':'stable per-candidate hash instead of offline seeded RNG'}
(stage/'assets'/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
archive=ROOT/'dist'/(NAME+'.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(stage.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
            z.write(path,path.relative_to(stage))
report={'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'bytes':archive.stat().st_size,'metadata':metadata}
(ROOT/'dist'/(NAME+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
