"""LUKE foundation adapter; not an admitted CoRE/CFIE/resampling/focal model.

The pinned author's parser and HF conversion classifier are executed directly.
AllenNLP batching is replaced at an explicit, tested interface boundary only.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

PACKAGE = Path(__file__).resolve().parent
LOCK_NAME = 'luke_foundation.lock.json'
ENT, ENT2 = '<ent>', '<ent2>'


def lock():
    return json.loads((PACKAGE / LOCK_NAME).read_text(encoding='utf-8'))


def file_digest(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def checked_file(directory, relative):
    root = Path(directory).absolute()
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('unsafe relative asset path')
    if path.resolve() != path.absolute() or root.resolve() != root:
        raise ValueError('symlinked source or asset')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_source(source):
    result = {}
    for name, spec in lock()['source']['files'].items():
        path = checked_file(source, spec['file'])
        actual = file_digest(path)
        if actual != spec['sha256']:
            raise ValueError('pinned LUKE source hash mismatch: ' + name)
        result[name] = actual
    return result


def verify_checkpoint(checkpoint):
    """Local streamed integrity checks; no downloads or torch imports."""
    result = {}
    for name, spec in lock()['checkpoint']['files'].items():
        path = checked_file(checkpoint, name)
        if path.stat().st_size != spec['size']:
            raise ValueError('checkpoint asset size mismatch: ' + name)
        if spec.get('sha256'):
            actual = file_digest(path)
            expected = spec['sha256']
        else:
            value = hashlib.sha1()
            value.update(b'blob ' + str(path.stat().st_size).encode() + b'\0')
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    value.update(block)
            actual, expected = value.hexdigest(), spec['git_blob']
        if actual != expected:
            raise ValueError('checkpoint asset hash mismatch: ' + name)
        result[name] = actual
    return result


def require_runtime_memory(minimum_bytes=12 * 1024**3):
    """Pre-import memory gate. Never changes cgroups or other processes."""
    limits = []
    cgroup_root = Path('/sys/fs/cgroup')
    membership = Path('/proc/self/cgroup')
    if membership.exists():
        for row in membership.read_text().splitlines():
            if row.startswith('0::'):
                group = cgroup_root / row[3:].lstrip('/')
                if '..' in group.parts:
                    raise RuntimeError('invalid cgroup membership')
                while group == cgroup_root or cgroup_root in group.parents:
                    if (group / 'memory.max').exists():
                        maximum = (group / 'memory.max').read_text().strip()
                        if maximum != 'max':
                            current = int((group / 'memory.current').read_text())
                            limits.append(max(0, int(maximum) - current))
                    if group == cgroup_root:
                        break
                    group = group.parent
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        for row in meminfo.read_text().splitlines():
            if row.startswith('MemAvailable:'):
                limits.append(int(row.split()[1]) * 1024)
    if not limits:
        raise RuntimeError('runtime memory identity is not verified')
    available = min(limits)
    if available < minimum_bytes:
        raise RuntimeError(f'LUKE full-weight runtime gate pending: headroom={available}, required={minimum_bytes}')
    return available


def parse_records(source, records):
    """Execute the exact pinned parse_tacred_file function with JSON I/O bound.

    Source mutates the decoded token list. A private copy preserves caller data.
    Neither labels nor ids influence sentence construction.
    """
    spec = lock()['source']['files']['examples/relation_classification/reader.py']
    path = checked_file(source, spec['file'])
    if file_digest(path) != spec['sha256']:
        raise ValueError('pinned LUKE reader hash mismatch')
    tree = ast.parse(path.read_bytes())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_tacred_file']
    if len(nodes) != 1:
        raise ValueError('unexpected upstream parser identity')
    namespace = {'Path': Path, 'ENT': ENT, 'ENT2': ENT2,
                 'json': SimpleNamespace(load=lambda _: copy.deepcopy(records)),
                 'open': lambda *_args: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return list(namespace['parse_tacred_file']('in_memory.json'))


def validate_records(records, labels):
    if len(labels) != 40 or len(set(labels)) != 40 or 'no_relation' not in labels:
        raise ValueError('explicit canonical40 label mapping required')
    for item in records:
        tokens = item.get('token', [])
        if not tokens or any(not isinstance(t, str) or not t for t in tokens):
            raise ValueError('invalid raw tokens')
        if any(ENT in t or ENT2 in t for t in tokens):
            raise ValueError('raw tokens collide with LUKE markers')
        for prefix in ('subj', 'obj'):
            start, end = item[prefix + '_start'], item[prefix + '_end']
            if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(tokens):
                raise ValueError('invalid inclusive raw entity span')
        if max(item['subj_start'], item['obj_start']) <= min(item['subj_end'], item['obj_end']):
            raise ValueError('overlapping entity spans need a separately declared reader adaptation')
        if item.get('relation') not in labels:
            raise ValueError('unknown canonical relation label')
        if 'id' not in item:
            raise ValueError('sample identity required')


def encode_records(source, tokenizer, records, labels):
    validate_records(records, labels)
    label_to_id = {label: index for index, label in enumerate(labels)}
    marker_ids = [tokenizer.convert_tokens_to_ids(token) for token in (ENT, ENT2)]
    expected = lock()['tokenizer']['marker_ids']
    if marker_ids != [expected[ENT], expected[ENT2]]:
        raise ValueError('LUKE marker vocabulary identity mismatch')
    mask_ids = [tokenizer.entity_vocab[token] for token in ('[MASK]', '[MASK2]')]
    if len(set(mask_ids)) != 2:
        raise ValueError('subject/object entity mask identity mismatch')
    features = []
    for raw, parsed in zip(records, parse_records(source, records)):
        encoded = tokenizer(parsed['sentence'], add_special_tokens=True, truncation=False)
        ids = list(encoded['input_ids'])
        if len(ids) > 512:
            raise ValueError('LUKE input exceeds512; no silent crop or entity truncation')
        if len(ids) < 2 or ids[0] != tokenizer.bos_token_id or ids[-1] != tokenizer.eos_token_id:
            raise ValueError('LUKE BOS/EOS identity mismatch')
        spans = []
        for marker in marker_ids:
            positions = [i for i, value in enumerate(ids) if value == marker]
            if len(positions) != 2:
                raise ValueError('missing or duplicated entity markers')
            spans.append(positions)
        features.append({'input_ids': ids, 'entity_spans': spans, 'entity_ids': mask_ids[:],
                         'labels': label_to_id[raw['relation']], 'sample_id': raw['id'],
                         'input_sentence': parsed['sentence']})
    return features


def collate_features(features, pad_id):
    """Author's inclusive span positions, padded -1 to this batch's maximum.

    No HuggingFace max_mention_length truncation: both entity markers are kept.
    Returns lists; conversion to device tensors belongs to the adapter.
    """
    if not features:
        raise ValueError('empty LUKE batch')
    width = max(len(item['input_ids']) for item in features)
    mention = max(end - start + 1 for item in features for start, end in item['entity_spans'])
    output = {key: [] for key in ('input_ids', 'attention_mask', 'entity_ids', 'entity_attention_mask',
                                  'entity_position_ids', 'labels')}
    for item in features:
        ids = item['input_ids']
        if len(item['entity_spans']) != 2 or len(item['entity_ids']) != 2:
            raise ValueError('exactly two role-ordered entities required')
        output['input_ids'].append(ids + [pad_id] * (width - len(ids)))
        output['attention_mask'].append([1] * len(ids) + [0] * (width - len(ids)))
        output['entity_ids'].append(item['entity_ids'][:])
        output['entity_attention_mask'].append([1, 1])
        positions = []
        for start, end in item['entity_spans']:
            if not 0 <= start < end < len(ids):
                raise ValueError('invalid encoded entity span')
            positions.append(list(range(start, end + 1)) + [-1] * (mention - (end - start + 1)))
        output['entity_position_ids'].append(positions)
        output['labels'].append(item['labels'])
    return output


def loading_keys_checked(info):
    allowed_missing = {'classifier.weight', 'classifier.bias', 'luke.pooler.dense.weight', 'luke.pooler.dense.bias'}
    missing = set(info.get('missing_keys', []))
    unexpected = info.get('unexpected_keys', [])
    if not {'classifier.weight', 'classifier.bias'} <= missing or missing - allowed_missing:
        raise ValueError('unexpected missing pretrained keys')
    if any(not name.startswith(('lm_head.', 'entity_predictions.')) for name in unexpected):
        raise ValueError('unexpected pretrained key')
    if info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('pretrained weight shape/loading error')


class LUKE:
    protocol = 'luke_foundation_canonical40_v1_not_formally_admitted'
    amp_enabled = False
    optimizer_contract = 'pending_common_loop_binding'
    precision_contract = 'foundation_fp32_runtime_gate_pending'

    def __init__(self, source, checkpoint, labels, *, device='cpu', checkpointing=False):
        validate_records([], labels)
        self.source = Path(source)
        self.source_hashes = verify_source(self.source)
        # Check memory before hash-reading2.2GB weights or importing torch.
        require_runtime_memory()
        self.checkpoint_hashes = verify_checkpoint(checkpoint)
        import torch
        from transformers import AutoConfig, LukeTokenizer
        self.torch, self.device = torch, torch.device(device)
        self.labels = list(labels)
        self.tokenizer = LukeTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                                       additional_special_tokens=[ENT, ENT2], task=None)
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        if config.model_type != 'luke' or not config.use_entity_aware_attention or config.vocab_size != 50267:
            raise ValueError('pretrained LUKE configuration identity mismatch')
        config.num_labels, config.classifier_bias = 40, True
        config.id2label = dict(enumerate(self.labels))
        config.label2id = {label: i for i, label in enumerate(self.labels)}
        spec = lock()['source']['files']['examples/relation_classification/luke_for_entity_pair_classification.py']
        path = checked_file(source, spec['file'])
        namespace = ModuleType('retacred_pinned_luke_classifier')
        exec(compile(path.read_bytes(), str(path), 'exec'), namespace.__dict__)
        self.model, self.loading_info = namespace.LukeForEntityPairClassification.from_pretrained(
            checkpoint, config=config, local_files_only=True, output_loading_info=True)
        loading_keys_checked(self.loading_info)
        # AllenNLP RelationClassifier uses nn.Linear's reset_parameters, unlike
        # the HF converter's normal post_init. Preserve that initialization law;
        # RNG call order/common seeds remain a declared adaptation, not bitwise identity.
        self.model.classifier.reset_parameters()
        if checkpointing:
            self.model.luke.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
        self.model.to(self.device)

    def encode(self, records):
        return encode_records(self.source, self.tokenizer, records, self.labels)

    def batch(self, features):
        lists = collate_features(features, self.tokenizer.pad_token_id)
        return {key: self.torch.tensor(value, dtype=self.torch.long, device=self.device) for key, value in lists.items()}

    def forward(self, batch):
        output = self.model(**batch, return_dict=True)
        return output.loss, output.logits

    def source_manifest(self):
        return dict(self.source_hashes)

    def finalize_vocabulary(self):
        if self.model.luke.get_input_embeddings().num_embeddings != len(self.tokenizer):
            raise ValueError('LUKE vocabulary resizing is not a permitted hidden adaptation')

    def trace_fields(self, batch, row):
        return {'entity_ids': batch['entity_ids'][row].detach().cpu().tolist(),
                'entity_position_ids': batch['entity_position_ids'][row].detach().cpu().tolist(),
                'representation': 'role_ordered_subject_then_object_entity_hidden_states'}

    def parameter_trace(self):
        return {'classifier.parameter_gradient.' + name: parameter.grad.detach().cpu().clone()
                for name, parameter in self.model.classifier.named_parameters() if parameter.grad is not None}
