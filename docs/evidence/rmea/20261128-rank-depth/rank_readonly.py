"""Read four already sealed logit matrices in place, emit only aggregate ranks."""
import hashlib,itertools,json,sys,time
from pathlib import Path
import numpy as np

def integer(x):return int(np.count_nonzero(x))
def digest(b):return hashlib.sha256(b).hexdigest()
def chk(pin):
    p=Path(pin['path']);assert p.is_absolute() and p.is_file() and not p.is_symlink()
    assert p.stat().st_size<=64*1024**2
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==pin['sha256'],p.name
    return p

def levels(z,labels):
    pos=labels>0; y=labels[pos]; p=z[pos,1:]; sg=p[np.arange(len(y)),y-1]
    ids=np.arange(1,z.shape[1])[None,:]
    rank=1+np.sum((p>sg[:,None])|((p==sg[:,None])&(ids<y[:,None])),axis=1)
    # Independent check: stable comparator sort, with positive ID order fixed.
    another=1+np.array([int(np.flatnonzero(order==lab-1)[0]) for order,lab in zip(np.argsort(-p,axis=1,kind='stable'),y)])
    assert np.array_equal(rank,another)
    pred=z.argmax(1); yp=pred[pos]
    exact=yp==y; null=yp==0; wrong=(~exact)&(~null)
    assert (rank[exact]==1).all() and not integer(wrong&(rank==1))
    code=np.select([exact,null&(rank==1),null&(rank<=3),null&(rank>=4),wrong&(rank<=3),wrong&(rank>=4)],np.arange(6),default=-1)
    assert len(code)==integer(pos) and code.min()>=0
    return rank,pred,code,dict(positive_gold_tied_with_another_positive=integer(np.sum(p==sg[:,None],axis=1)>1),
                              tied_null_with_gold=integer(z[pos,0]==sg))

def cross(a,b,k):return np.bincount(k*a+b,minlength=k*k).reshape(k,k).tolist()

def audit(cfg):
    start=time.monotonic();path={k:chk(v) for k,v in cfg['pins'].items()}
    cohort=json.loads(path['cohort'].read_bytes());idx=np.asarray(cohort['assessment'],dtype=np.int64)
    yall=np.load(path['train_targets'],allow_pickle=False)
    assert yall.shape==(58465,) and yall.dtype.kind=='i' and yall.min()==0 and yall.max()==39
    assert idx.shape==(4096,) and len(set(idx))==4096 and not set(idx)&set(cohort['fit']) and len(cohort['fit'])==8192
    assert cohort['target_sha256']==cfg['pins']['train_targets']['sha256']
    labels=yall[idx];pos=labels>0;row={};rankdata={};states={};preds={}
    for name,zs in [('distinct','Q D G'.split()),('reference',['Gcov'])]:
        with np.load(path[name],allow_pickle=False) as a:
            assert np.array_equal(a['assessment_indices'],idx) and np.array_equal(a['labels'],yall)
            assert np.array_equal(np.flatnonzero(a['calibration']),np.sort(cohort['fit']))
            for short in zs:
                z=a[cfg['keys'][short]+'_logits']
                assert z.shape==(4096,40) and z.dtype.kind=='f' and z.dtype.itemsize in (4,8) and np.isfinite(z).all()
                rank,pred,code,ties=levels(z,labels)
                cm=cross(labels,pred,40);assert cm==cfg['confusions'][short]
                hist=np.bincount(code,minlength=6);positive_errors=integer(pred[pos]!=labels[pos]);fp=integer(pred[~pos]!=0)
                assert sum(hist[1:])==positive_errors and sum(hist[1:4])==sum(cm[i][0] for i in range(1,40)) and fp==sum(cm[0][1:])
                rankdata[short]=rank;states[short]=code;preds[short]=pred
                row[short]=dict(dtype=str(z.dtype),positive_codes=list(map(int,hist)),positive_to_null=int(sum(hist[1:4])),positive_to_wrong_positive=int(sum(hist[4:])),null_false_positive=fp,
                    null_correct=integer(pred[~pos]==0),positive_rank1_all=int(hist[0]+hist[1]),positive_rank_top3_all=int(sum(hist[[0,1,2,4]])),
                    positive_rank_histogram_1_to_39=np.bincount(rank,minlength=40)[1:].tolist(),ties=ties)
    assert len(states)==4
    pair={}
    for a,b in itertools.combinations(('Q','D','G','Gcov'),2):
        aa,bb=preds[a],preds[b];ra,rb=rankdata[a],rankdata[b]
        pair[a+'__'+b]=dict(positive_codes_matched=cross(states[a],states[b],6),gold_positive_rank_a_better=integer(ra<rb),gold_positive_rank_b_better=integer(ra>rb),gold_positive_rank_equal=integer(ra==rb),
          top3_a_only=integer((ra<=3)&(rb>3)),top3_b_only=integer((ra>3)&(rb<=3)),
          positive_correct_a_only=integer((states[a]==0)&(states[b]!=0)),positive_correct_b_only=integer((states[b]==0)&(states[a]!=0)),
          null_fp_status_a_by_b=cross((aa[~pos]!=0).astype(int),(bb[~pos]!=0).astype(int),2),
          all_correct_a_only=integer((aa==labels)&(bb!=labels)),all_correct_b_only=integer((aa!=labels)&(bb==labels)))
    assert pair['Q__Gcov']['all_correct_a_only']==241 and pair['Q__Gcov']['all_correct_b_only']==300
    return dict(status='PASS_RETAINED_RANK_DEPTH_AND_SAME_ROW_ERROR_COUNTS',cohort='existing exposed TRAIN4096, not independent confirmation',total=len(labels),positive=integer(pos),null=integer(~pos),classes=40,
        code_order=['correct','null_rank1','null_rank2to3','null_rank4plus','wrong_positive_rank2to3','wrong_positive_rank4plus'],
        source_hashes={k:v['sha256'] for k,v in cfg['pins'].items()}, source_label_hash=cfg['public_label_file_sha256'], exact_confusion_parity=4,
        reports=row,pairwise=pair, tie_rule='Original argmax chooses lowest label ID on exact ties. Positive-only rank orders original positive IDs ascending after descending stored logit.',
        read_label_rows_in_sealed_existing_records=len(labels),checked_train_label_array_length=len(yall),task_forwards=0,optimizer_updates=0,gpu_allocations=0,remote_writes=0,
        checkpoint_decodes=0, feature_arrays_read=0,official_splits_read=0,individual_records_exported=0,seconds=time.monotonic()-start)

if __name__=='__main__':print(json.dumps(audit(json.load(sys.stdin)),allow_nan=False))
