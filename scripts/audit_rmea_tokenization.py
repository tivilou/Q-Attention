"""Observe full-population word/subword alignment without truncation or a model."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def marked_words(row):
    """Use pretrained punctuation tokens; retain a map to original word indices.

    Every relation candidate receives the same words and ordered entity spans.
    Labels/types/metadata never enter this construction. Markers are excluded
    from evidence pooling; the contextualized original words retain their effect.
    """
    words, original = [], []
    starts = {row['subject'][0]: '@', row['object'][0]: '#'}
    ends = {row['subject'][1]: '@', row['object'][1]: '#'}
    for index, token in enumerate(row['tokens']):
        if index in ends:
            words.append(ends[index]); original.append(None)
        if index in starts:
            words.append(starts[index]); original.append(None)
        words.append(token); original.append(index)
    if len(row['tokens']) in ends:
        words.append(ends[len(row['tokens'])]); original.append(None)
    return words, original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous audit')
    from transformers import RobertaTokenizerFast
    tokenizer=RobertaTokenizerFast.from_pretrained(args.assets,add_prefix_space=True,local_files_only=True)
    splits={}
    for split in ('train','valid','test'):
        lengths=Counter();missing_word_rows=missing_entity_rows=over512=0;words_total=0;count=0
        with (args.data_dir/(split+'.jsonl')).open(encoding='utf8') as handle:
            for line in handle:
                row=json.loads(line);count+=1
                words,original=marked_words(row)
                encoded=tokenizer(words,is_split_into_words=True,truncation=False,return_attention_mask=False)
                ids=encoded.word_ids();lengths[len(ids)]+=1;over512+=len(ids)>512
                covered={original[i] for i in ids if i is not None and original[i] is not None}
                expected=set(range(len(row['tokens'])))
                entity=set(range(*row['subject'])) | set(range(*row['object']))
                missing_word_rows+=covered!=expected;missing_entity_rows+=not entity<=covered
                words_total+=len(row['tokens'])
        splits[split]=dict(records=count,max_subwords=max(lengths),subword_histogram=dict(sorted(lengths.items())),
                           over_encoder_limit512=over512,missing_word_rows=missing_word_rows,
                           missing_entity_rows=missing_entity_rows,word_tokens=words_total,
                           word_fp16_bytes=words_total*1024*2)
    result=dict(schema='qattn.rmea.tokenization-inspection.v1',status='observations_only',
                tokenizer='FacebookAI/roberta-large@722cf37b1afa9454edce342e7895e588b6ff1d59',
                policy='add_prefix_space=True; pretokenized entity punctuation; no truncation; word mean pooling planned',
                splits=splits,model_loaded=False,training=False,
                tokenizer_sha256=hashlib.sha256((args.assets/'tokenizer.json').read_bytes()).hexdigest())
    payload=(json.dumps(result,indent=2)+'\n').encode('utf8')
    with args.output.open('xb') as handle:handle.write(payload)
    print(json.dumps(dict(output=str(args.output),sha256=hashlib.sha256(payload).hexdigest(),
                         splits={s:{k:v for k,v in r.items() if k!='subword_histogram'} for s,r in splits.items()})))
    if any(r['over_encoder_limit512'] or r['missing_word_rows'] or r['missing_entity_rows'] for r in splits.values()):raise SystemExit(2)


if __name__=='__main__':main()
