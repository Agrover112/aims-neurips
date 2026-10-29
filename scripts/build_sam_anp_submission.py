"""Fetch the selected SAM ANP and build a minimal cached inference archive."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile
import modal
from modal.config import config
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from organized.training.anp_crossattn_blocks import AttentiveNeuralProcess

REMOTE='/results/anp-ood/anp-ood-bge1024-20261029-v1/sam/best.pt'
NAME='anp-sa1-w64-bgem3-benchmarkholdout-sam-step700-random'
client=modal.Client.from_credentials(config.get('token_id',profile='arshiaswe',use_env=False),
                                     config.get('token_secret',profile='arshiaswe',use_env=False))
volume=modal.Volume.from_name('aims-neurips-data').hydrate(client)
data=b''.join(volume.read_file(REMOTE))
source_sha=hashlib.sha256(data).hexdigest()
source=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
assert source['best_step']==700 and source['config']['variant']=='sam'
assert source['weight_source']=='online'
model=AttentiveNeuralProcess(1024,width=64,self_attention_blocks=1,cross_attention_heads=4,latent_dim=32)
model.load_state_dict(source['state_dict'],strict=True)
model.eval()
config={'model_family':'anp_cross_attention','width':64,'n_blocks':1,'cross_attention_heads':4,
        'use_latent':True,'latent_dim':32,'residual_proj':False,'dropout':0.}
out=ROOT/'dist'; out.mkdir(exist_ok=True)
stage=out/NAME
stage.mkdir(exist_ok=True)
common=ROOT/'organized'/'submission'
for name in ('model.py','labeling.py','models.txt','requirements.txt','prediction_config.json'):
    shutil.copy2(common/name,stage/name)
(stage/'models.txt').write_text('BAAI/bge-m3\n')
package=stage/'aims_predictor'; package.mkdir(exist_ok=True)
for name in ('__init__.py','predictor.py'):
    shutil.copy2(common/'aims_predictor'/name,package/name)
for name in ('cnp_blocks.py','anp_crossattn_blocks.py'):
    shutil.copy2(ROOT/'organized'/'training'/name,package/name)
assets=stage/'assets'; assets.mkdir(exist_ok=True)
filename='anp-sam-step700.pt'
torch.save({'config':config,'state_dict':source['state_dict']},assets/filename)
metadata={'checkpoint':filename,'embedding':{'model':'BAAI/bge-m3','item_text':'content_only',
    'native_dimension':1024,'output_dimension':1024,'max_length':8192,'pooling':'cls','l2_normalized':True},
    'model':{'family':'anp_cross_attention','width':64,'latent_dimension':32,
             'cross_attention_heads':4,'context_self_attention_blocks':1,'max_context':31},
    'acquisition':'random_streaming_with_reserve_rule',
    'provenance':{'workspace':'arshiaswe','source_checkpoint':REMOTE,'source_sha256':source_sha,
        'export_sha256':hashlib.sha256((assets/filename).read_bytes()).hexdigest(),
        'variant':'sam','selected_step':700,'seed':20261018,
        'benchmark_split':'11 train / 3 validation / 3 test',
        'selection_brier_alc':source['best_validation_alc'],
        'correction_adapter':False,'expected_brier_acquisition':False}}
(assets/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
archive=out/(NAME+'.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(stage.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            z.write(path,path.relative_to(stage))
report={'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'bytes':archive.stat().st_size,'source_sha256':source_sha,'metadata':metadata}
(out/(NAME+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
