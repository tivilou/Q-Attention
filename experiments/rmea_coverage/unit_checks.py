"""Selection and preprocessing fixtures without real training or model assets."""
import numpy as np
from coverage_data import rank,select,sid,roles_for
from marking import marked_words
def run_checks():
    rows=[dict(metadata={'id':f'{c}-{i}'},label=c,tokens=['Alice','met','Bob'],subject=[0,1],object=[2,3]) for c,n in [('a',6),('b',2),('c',1)] for i in range(n)]
    old=[sid(rows[i]) for i in [0,1,2,8]];counts={'a':3,'b':0,'c':1}
    picked,original,census=select(iter(rows),old,['a','b','c'],counts,cap=4)
    assert census=={'a':6,'b':2,'c':1} and len(picked)==3
    for order in [list(reversed(rows)),rows[2:]+rows[:2]]:
        other,_,_=select(iter(order),old,['a','b','c'],counts,cap=4)
        assert [sid(x) for x in picked]==[sid(x) for x in other]
    assert [x['label'] for x in picked]==['a','b','b'] and set(original)==set(old)
    eligible=sorted((r for r in rows if r['label']=='a' and sid(r) not in old),key=lambda r:(rank(r),sid(r)))
    assert sid(picked[0])==sid(eligible[0])
    duplicate=False
    try:select(iter(rows+[rows[0]]),old,['a','b','c'],counts,4)
    except AssertionError:duplicate=True
    assert duplicate
    missing=False
    try:select(iter(rows[1:]),old,['a','b','c'],counts,4)
    except AssertionError:missing=True
    assert missing
    words,mapping=marked_words(rows[0]);assert words==['@','Alice','@','met','#','Bob','#'] and mapping==[None,0,None,1,None,2,None]
    roles=roles_for(rows[0]);assert roles.dtype==np.float32 and roles.shape==(3,4)
    np.testing.assert_array_equal(roles[:,2:],[ [1,0],[0,0],[0,1]])
    checks=['train-only cap/availability handling','old sample preservation','order permutation invariance','exact hash rank selection','duplicate ID rejection','missing base ID rejection','unchanged marker alignment fixture','exact ordered role fixture']
    return dict(status='PASS',checks=checks,training_steps=0)
