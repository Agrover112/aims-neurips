"""Verify cache persistence, isolation and direct model inference without fallback."""
import importlib
import os
from pathlib import Path
import sys
import tempfile
import numpy as np

stage=Path(__file__).resolve().parents[1]/'dist'/'anp-sa1-w64-bgem3-benchmarkholdout-sam-step700-random'
sys.path.insert(0,str(stage))
os.environ['PREDICTIVE_EVAL_LOCAL_SMOKE_TEST']='1'
p=importlib.import_module('aims_predictor.predictor')
with tempfile.TemporaryDirectory() as temporary:
    os.environ[p.CACHE_DIR_ENV]=temporary
    first=p._DiskCache()
    vector=np.arange(p.EMBEDDING_DIM,dtype=np.float32)
    key=p._text_key('persistent cache test')
    first.put(key,vector)
    second=p._DiskCache()
    np.testing.assert_array_equal(second.get(key),vector)
    embedder=p._ItemEmbedder()
    embedder.local_smoke=False
    def fail_load():
        raise AssertionError('Warm disk cache must not load an encoder')
    embedder._load=fail_load
    np.testing.assert_array_equal(embedder.encode_many(['persistent cache test'])[0],vector)
    original=p.MODEL_ID
    p.MODEL_ID='different/encoder'
    isolated=p._DiskCache()
    assert isolated.get(key) is None
    p.MODEL_ID=original
    first._path(key).write_bytes(b'corrupt')
    assert first.get(key) is None
    first.put(key,np.full(p.EMBEDDING_DIM,np.nan,dtype=np.float32))
    assert first.get(key) is None
    predictor=p.NeuralPredictor()
    subject={'normalized_name':'test-subject'}
    item={'benchmark_id':'test-benchmark','item_content':'What is two plus two?'}
    probability=predictor.predict([subject,item],[])
    assert np.isfinite(probability) and 0<=probability<=1
    assert predictor.model.width==64 and predictor.model.latent_dim==32
print('PASS: persistent cache, encoder namespace isolation, corruption handling, lazy warm-cache path, strict ANP load and direct prediction')
