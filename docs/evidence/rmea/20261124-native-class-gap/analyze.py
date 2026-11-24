"""Describe all40 native TRAIN F1 gaps using already published confusion counts."""
from pathlib import Path
import hashlib,json
import numpy as np
r=Path(__file__).resolve().parent;source=r.parent/'full-train-error-contrast-20261124/cpu-summary.json'
s=json.loads(source.read_bytes());labels=s['labels'];mat={m:np.array(v['confusion_matrix'],dtype=np.int64) for m,v in s['results'].items()}
scores={}
for m,c in mat.items():
 den=c.sum(0)+c.sum(1);scores[m]=np.divide(2*np.diag(c),den,out=np.zeros(40),where=den!=0)
 assert abs(scores[m].mean()-s['results'][m]['all40_macro_f1'])<1e-12
assert all(np.array_equal(c.sum(1),mat['quantum'].sum(1)) for c in mat.values())
rows=[]
for i in range(40):
 c=mat['quantum'];qwrong=c[i].copy();qwrong[i]=0
 j=int(qwrong.argmax())
 rows.append(dict(index=i,label=labels[i],support=int(c[i].sum()),f1={m:float(v[i]) for m,v in scores.items()},
  q_minus_real12=float(scores['quantum'][i]-scores['real12'][i]),
  macro_contribution=float((scores['quantum'][i]-scores['real12'][i])/40),
  quantum_errors=int(c[i].sum()-c[i,i]),real12_errors=int(mat['real12'][i].sum()-mat['real12'][i,i]),
  quantum_wrong_to_null=int(c[i,0]) if i else 0,real12_wrong_to_null=int(mat['real12'][i,0]) if i else 0,
  most_frequent_quantum_error=dict(label=labels[j],count=int(qwrong[j]))))
delta=float(scores['quantum'].mean()-scores['real12'].mean())
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),population='All58465 native selected-model TRAIN replay; not new heldout or causal evidence.',
 rows=rows,q_minus_real12=delta,per_class_contributions_sum=float(sum(v['macro_contribution'] for v in rows)),
 classes_q_better=sum(v['q_minus_real12']>0 for v in rows),classes_q_worse=sum(v['q_minus_real12']<0 for v in rows),
 classes_tied=sum(v['q_minus_real12']==0 for v in rows),new_model_forwards=0,training_updates=0,new_valid=0,new_test=0,cos_writes=0)
assert abs(out['per_class_contributions_sum']-delta)<1e-12
(r/'class-gap.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
inverse=[];missing=[]
for a,b in [('per:parents','per:children'),('org:parents','org:subsidiaries'),('org:members','org:member_of')]:
 if a not in labels or b not in labels:
  missing.append(dict(pair=[a,b],reason='Not both present in actual40-label inventory'));continue
 i,j=labels.index(a),labels.index(b);v=dict(pair=[a,b],counts={})
 for m,c in mat.items():
  v['counts'][m]=dict(a_to_b=int(c[i,j]),b_to_a=int(c[j,i]),a_to_null=int(c[i,0]),b_to_null=int(c[j,0]),support=int(c[i].sum()+c[j].sum()))
 inverse.append(v)
(r/'inverse-confusion.json').write_text(json.dumps(dict(scope='All native complete TRAIN aggregate confusions; taxonomy inverse pairs do not imply reversed natural examples exist or encoder covariance.',pairs=inverse,not_in_inventory=missing),indent=2)+'\n',encoding='utf8')
lines=['# 原生完整TRAIN差距分解：不是少数方向混淆','',
 f"复用完整TRAIN58465/全部40类的五路selected模型既有混淆矩阵，零新模型调用/训练/VALID/TEST/COS。Q对real12 Macro差{100*delta:+.6f}百分点：{out['classes_q_worse']}类更低、{out['classes_q_better']}类更高、{out['classes_tied']}类持平。全部逐类贡献之和精确复原总差。",'',
 '| 最大Q相对劣势 | 支持 | Q F1 | real12 F1 | 差/百分点 | Q/real错例 |',
 '| --- | ---: | ---: | ---: | ---: | ---: |']
for v in sorted(rows,key=lambda v:v['q_minus_real12'])[:8]:
 lines.append(f"| {v['label']} | {v['support']} | {v['f1']['quantum']:.6f} | {v['f1']['real12']:.6f} | {100*v['q_minus_real12']:+.3f} | {v['quantum_errors']}/{v['real12_errors']} |")
extra=sum(v['counts']['quantum']['a_to_b']+v['counts']['quantum']['b_to_a']-v['counts']['real12']['a_to_b']-v['counts']['real12']['b_to_a'] for v in inverse)
errors=int(sum(v['quantum_errors']-v['real12_errors'] for v in rows))
lines+=['',f"真实标签表中parents/children与members/member_of两组的互逆错分，Q共{sum(v['counts']['quantum']['a_to_b']+v['counts']['quantum']['b_to_a'] for v in inverse)}、real12共{sum(v['counts']['real12']['a_to_b']+v['counts']['real12']['b_to_a'] for v in inverse)}，差{extra}；全体Q与real12错例差{errors}。这不是额外修复23条的可部署收益或上界：模型错例可不同、其他标签/FP会变化，但只看这两对不足以解释广泛差距。候选org:parents/org:subsidiaries在本实际标签表不存在，明确保留为未纳入，不替换配对救援。",'',
 'Q最大正F1差来自per:country_of_death，但支持仅6例；全部正负逐类结果保留class-gap.json，不把极小类优势作稳定机制。F1包含列向FP，不能只按gold类错例数判断哪一方F1更高。',
 '这些是已完成模型在TRAIN的后验描述，不是独立泛化或量子根因。角色互换自然样本是否存在、标签是否一致以及冻结marked-input编码是否协变尚未检查，不能从关系名直接推对称性。先核对自然匹配输入与既有预测，再决定是否值得一个角色绑定候选，不能据这两个小差额直接推新全量实验。']
(r/'class-gap.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
ranked=sorted(rows,key=lambda v:v['q_minus_real12'])
print(json.dumps(dict(delta=delta,q_better=out['classes_q_better'],q_worse=out['classes_q_worse'],tied=out['classes_tied'],
 largest_disadvantages=[{k:v[k] for k in ('label','support','q_minus_real12','quantum_errors','real12_errors','quantum_wrong_to_null','real12_wrong_to_null','most_frequent_quantum_error')} for v in ranked[:8]],
 largest_advantages=[{k:v[k] for k in ('label','support','q_minus_real12','quantum_errors','real12_errors')} for v in ranked[-5:]])))
