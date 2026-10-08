"""Real offline BPE evidence; no complete-backbone/AllenNLP admission claim."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from baselines.retacred import luke_tokenizer_fidelity as gate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def measured(tmp_path_factory):
    output = tmp_path_factory.mktemp('luke-real-bpe') / 'evidence.json'
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    code = '''
import runpy,sys
def audit(event,args):
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo'):
        raise RuntimeError('offline fixture attempted network access')
    if event=='open' and str(args[0]).endswith(('pytorch_model.bin','.safetensors')):
        raise RuntimeError('tokenizer gate attempted weight read')
sys.addaudithook(audit)
runpy.run_module('baselines.retacred.luke_tokenizer_fidelity',run_name='__main__')
'''
    process = subprocess.run([sys.executable, '-c', code, '--output', str(output)],
                             cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)
    assert process.returncode == 0, process.stdout + process.stderr
    return json.loads(output.read_text()), json.loads(process.stdout)


def test_actual_offline_tokenizer_proof_and_backend_guard(measured):
    result, summary = measured
    assert result['status'] == 'tokenizer_interface_verified'
    assert summary['positive_cases'] == 9 and summary['negative_cases'] == 5
    assert result['source_files_verified'] == 15 and len(result['tokenizer_asset_sha256']) == 7
    assert result['marker_ids'] == [50265, 50266] and result['vocab_size'] == 50267
    assert result['model_loaded'] is False and result['torch_imported'] is False
    assert result['weight_file_read'] is False and result['network_requests'] == 0
    assert result['peak_rss_bytes'] > 0


@pytest.mark.parametrize('name', [row['id'] for row in gate.fixtures()])
def test_real_bpe_role_spans_and_full_position_arrays(measured, name):
    rows = {row['case']: row for row in measured[0]['positive_cases']}
    row = rows[name]
    assert len(row['bpe_tokens']) == len(row['input_ids']) == row['token_count']
    assert row['input_ids'][0] == 0 and row['input_ids'][-1] == 2
    assert row['entity_ids'] == measured[0]['mask_ids']
    assert row['source_span_match']
    for marker, span, positions in zip([50265,50266], row['entity_spans'], row['entity_position_ids']):
        assert row['input_ids'][span[0]] == row['input_ids'][span[1]] == marker
        assert [p for p in positions if p != -1] == list(range(span[0],span[1]+1))


def test_512_boundary_and_no_30_token_cap(measured):
    rows = {row['case']: row for row in measured[0]['positive_cases']}
    assert rows['exact512']['token_count'] == 512
    assert len(rows['long_entity']['entity_position_ids'][0]) == 37
    assert len([p for p in rows['multisubword']['entity_position_ids'][0] if p!=-1]) > 3
    assert measured[0]['negative_cases'][0]['case']=='over512'
    assert 'exceeds512' in measured[0]['negative_cases'][0]['reason']


def test_gold_isolation_batch_padding_and_honest_scope(measured):
    result = measured[0]
    assert result['gold_input_isolation_verified'] and result['batch_padding_verified']
    assert result['source_span_method_executed'] and not result['allen_nlp_runtime_checked']
    assert not result['strict_comparison_eligible'] and result['jobs_launched'] == 0
    assert 'actual_AllenNLP_wrapper_indexer_runtime_equivalence' in result['pending']
    assert all(not row['accepted'] for row in result['negative_cases'])


def test_existing_output_preserved_before_loading(tmp_path, monkeypatch):
    output = tmp_path/'old.json'; output.write_text('existing-evidence')
    monkeypatch.setattr(gate,'report',lambda *args:pytest.fail('must reject before tokenizer load'))
    assert gate.main(['--output',str(output)])==1
    assert output.read_text()=='existing-evidence'


def test_fresh_process_and_forbidden_backends(monkeypatch):
    with pytest.raises(ImportError,match='backend import'):
        gate.NoModelImports().find_spec('torch.nn')
    monkeypatch.setitem(sys.modules,'torch',object())
    with pytest.raises(ValueError,match='fresh process'):
        gate.assert_no_backend()


def test_source_reader_tampering_not_executed(tmp_path):
    spec=gate.lock()['source']['files']['examples/relation_classification/reader.py']
    (tmp_path/spec['file']).write_text('raise RuntimeError("not executed")')
    with pytest.raises(ValueError,match='identity mismatch'):
        gate.source_reader(tmp_path,None)


def test_fixtures_are_bounded_and_independent():
    first=gate.fixtures(); original=copy.deepcopy(first); first[0]['token'][0]='changed'
    assert gate.fixtures()==original
    assert len(first)==9 and max(len(row['token']) for row in first)==505
