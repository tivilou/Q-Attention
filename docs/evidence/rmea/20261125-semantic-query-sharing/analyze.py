"""Analyze every fixed outcome without changing the promotion metric or budget."""
from pathlib import Path
import json

r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
s,cfg,verified,closed=(read(n) for n in ('summary.json','design.json','verification.json','allocation-closeout.json'))
assert verified['status']=='PASS' and closed['state']=='released'
rows={'new/'+k:v for k,v in s['results'].items()}
rows.update({'reused/'+k:v for k,v in s['reused_results'].items()})
assert len(rows)==8
save=lambda n,v:(r/n).write_bytes((json.dumps(v,indent=2)+'\n').encode())
metrics={}
for k,v in rows.items():
    cm=v['confusion_matrix']
    metrics[k]=dict(macro=v['all40_macro_f1'],ce=v['ce'],
        correct=sum(cm[i][i] for i in range(40)),null_f1=v['per_class_f1'][0],
        positive39_macro=sum(v['per_class_f1'][1:])/39,
        positive_to_null=sum(cm[i][0] for i in range(1,40)),null_false_positive=sum(cm[0][1:]),
        predicted_null=sum(x[0] for x in cm))
q=metrics['new/quantum_shared']['macro']
contrasts=dict(Qshared_minus_Qnative=q-metrics['reused/quantum']['macro'],
    Qshared_minus_Qpermuted=q-metrics['new/quantum_permuted']['macro'],
    Qshared_minus_Dshared=q-metrics['new/dephased_shared']['macro'],
    Qshared_minus_Gshared=q-metrics['new/gaussian_shared']['macro'],
    Qshared_minus_Gcov_practical=q-metrics['reused/gaussian_covariance']['macro'],
    Dshared_minus_Dnative=metrics['new/dephased_shared']['macro']-metrics['reused/dephased']['macro'],
    Gshared_minus_Gnative=metrics['new/gaussian_shared']['macro']-metrics['reused/gaussian_base']['macro'])
names=read('label-codes.json')['labels']
counts=read('../prior-adjusted-native-training-20261125/design.json')['fit_class_counts']
classes=[]
for i in range(40):
    f={k:v['per_class_f1'][i] for k,v in rows.items()}
    classes.append(dict(class_id=i,label=names[i],fit_count=counts[i],
        assessment_support=rows['new/quantum_shared']['class_support'][i],f1=f,
        quantum_sharing_minus_native=f['new/quantum_shared']-f['reused/quantum'],
        quantum_sharing_minus_permuted=f['new/quantum_shared']-f['new/quantum_permuted']))
save('class-analysis.json',classes)
save('contrast-analysis.json',dict(descriptive_macro_differences=contrasts,metrics=metrics,
    gates=s['gates'],stage_decision=s['gate'],limitation='Development screen, known low-rank sharing; no causal variance/coherence or independent confirmation claim.'))
save('primary-disposition.json',dict(stage='small_single_seed',decision=s['gate'],gates=s['gates'],
    all_eight_conditions_retained=True,new_fits=4,physical_updates=4096,
    full_data='NOT_STARTED',multi_seed='NOT_STARTED',new_L3='NOT_ACCEPTED',new_L4_T='NOT_ACCEPTED',
    no_rank_tokenization_threshold_lr_seed_cohort_budget_control_rescue=True,goal='ACTIVE'))
table='\n'.join(f"| {k} | {v['macro']:.12f} | {v['ce']:.12f} | {v['positive_to_null']} | {v['null_false_positive']} |" for k,v in metrics.items())
deltas='\n'.join(f'- {k}: {v*100:+.6f}个百分点。' for k,v in contrasts.items())
decision=('七门全部通过，只允许设计完整单seed协议，尚未启动全量/多seed。' if all(s['gates'].values()) else
          '七门未全部通过，暂缓这个确切rank7/name-token/初始化/1024配置，不进入全量/多seed。')
text=f'''# 关系语义共享查询：固定四路小规模筛选

## 问题与实际干预

上一轮原生模型诊断显示，Q在8192条训练窗口比D拟合更好，但在固定4096条评估上
更差，稀有关系贡献了较大Macro差距，部分关系在训练集也弱。这没有隔离统计方差、
query、相干或读出的原因。本轮检验一个已知共享结构是否有帮助的假设，不把它包装
成新理论：公开关系名的固定低秩编码，将40×12自由查询改成40×8固定码乘8×12系数。
null独占第1列，正类使用其余7维；原草稿的“第8坐标”措辞已澄清，编码字节未改。

Q/D再使用原pi*tanh与3qubit circuit；Gaussian仍使用原12维各向异性距离。
key/value/decoder/bias与原始初始化及训练方法一致；W由各原seed13自由query的
最小二乘投影初始化，不声称初始路由输出与自由query模型相同。原query仅为buffer。
三个结构臂同样获得公开关系名信息，均80644训练参数，原81028；registered count
相同不证明有效容量相同。Qpermuted按固定public-name hash循环移位全部39正类行，
null不动，code.T@code谱与秩保持，语义对应被打乱；不声称同标签两两距离保持。
精确code b7622c7b固定，无任务文本、训练targets、支持数或表现参与编码构造。

## 全部八路结果

| 条件 | 全40类Macro-F1 | CE | 正关系→null | null→正关系 |
|---|---:|---:|---:|---:|
{table}

{deltas}

固定七门{sum(s['gates'].values())}/7通过。{decision}
结果分析必须同时保留三个新对照和全部四个旧参照；不改成正类条件排序、低支持组
或平均CE作为成功判据。Gaussian covariance是更强函数类的实用参照，不是同参数
共享消融。仅胜原nativeQ不能证明量子独立贡献；小筛选不确认L3或T。
不扫rank、分词、阈值、LR、seed、训练预算、cohort或删除对照救援；本确切配置
未晋级也不证明所有语义共享/量子方案不可能。所有40类CM/F1/support均保留。

## 执行与边界

同一原TRAIN8192fit/4096assessment、sentence-group无重叠、全部40类、seed13，
1024updates/b128/micro4/FP32-complex64/TF32off，原AdamW参数不变；四路fit
全部完成后统一最终评估。四旧参照cohort/order/初始化方法/预算相同，无重复baseline。
{s['seconds']:.3f}秒/4096实际更新，双2080，gen{closed['generation']}已可信归还。
稳定screen50ce858f/host238f7e68字节未改；只做新code/buffer、最小二乘、查询梯度
链式/有限差分、原生公式替换/Gaussian NumPy、padding和克隆actual8更新检查。
Astra6.2/high首轮BLOCK保留：修正继承的总更新数字和训练顺序union检查；完整runtime
design原本含四reused rows，审阅精简packet遗漏已澄清，第二轮确切语义PASS。
该静态审核不是实际fixtures或独立理论审核。实际fixtures结果另存fixtures.json。
八路40类CM独立重算最大误差{verified['independent_macro_max_error']:.3g}。
无串行/暂停/多卡/COS恢复矩阵，无新VALID/TEST或COS写入，Windows无模型/特征/
逐例数组。短筛选只记录Git代码/配置/聚合结果，不自动上传checkpoint。

已知语义参数共享与低秩编码不是本项目新T，没有证明量子优势、泛化界、方差因果、
硬件加速或实质近邻区别。新L3/L4-T仍NOT_ACCEPTED，目标ACTIVE，原限定L4及
所有历史结果保留。下一步依据全部结果选择，不选择性追加seed或升级算力。
'''
(r/'analysis.md').write_bytes(text.encode())
print(json.dumps(dict(stage_decision=s['gate'],gates=s['gates'],contrasts=contrasts,metrics=metrics)))
