"""Analyze every frozen result without promoting a control or changing gates."""
from pathlib import Path
import json
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
save=lambda n,v:(r/n).write_bytes((json.dumps(v,indent=2)+'\n').encode())
s,verified,closed=(read(n) for n in ('summary.json','verification.json','allocation-closeout.json'))
assert verified['status']=='PASS' and closed['state']=='released'
rows={'new/'+k:v for k,v in s['results'].items()}
rows.update({'reused/'+k:v for k,v in s['reused_results'].items()})
assert len(rows)==14 and len(s['gates'])==13
metrics={}
for name,row in rows.items():
    cm=row['confusion_matrix']
    metrics[name]=dict(macro=row['all40_macro_f1'],ce=row['ce'],correct=sum(cm[i][i] for i in range(40)),
        positive39_macro=sum(row['per_class_f1'][1:])/39,null_f1=row['per_class_f1'][0],
        positive_to_null=sum(cm[i][0] for i in range(1,40)),null_false_positive=sum(cm[0][1:]),parameters=row['parameters'])
q=metrics['new/quantum_distinct']['macro']
contrasts={k:q-v['macro'] for k,v in metrics.items() if k!='new/quantum_distinct'}
labels=read('label-codes.json')['labels']
counts=read('../prior-adjusted-native-training-20261125/design.json')['fit_class_counts']
classes=[dict(class_id=i,label=name,fit_count=counts[i],assessment_support=rows['new/quantum_distinct']['class_support'][i],
    f1={k:v['per_class_f1'][i] for k,v in rows.items()}) for i,name in enumerate(labels)]
save('class-analysis.json',classes)
save('contrast-analysis.json',dict(metrics=metrics,candidate_macro_minus_each_control=contrasts,
    gates=s['gates'],stage_decision=s['gate'],limitation='Exposed development cohort/single seed; not semantic binding, causal coherence, significance or independent confirmation.'))
classical=metrics['new/gaussian_distinct']['macro']>metrics['reused/earlier/native_gaussian_covariance']['macro']
save('primary-disposition.json',dict(stage='small_single_seed',decision=s['gate'],gates=s['gates'],
    all_fourteen_conditions_retained=True,new_fits=5,physical_updates=5120,
    full_data='NOT_STARTED',multi_seed='NOT_STARTED',new_L3='NOT_ACCEPTED',new_L4_T='NOT_ACCEPTED',
    classical_derivative=dict(identity='gaussian_distinct',candidate_for_own_protocol=classical,
        grade='DEVELOPMENT_SCREEN_ONLY',replaces_quantum_goal=False),
    no_normalization_pair_rank_scale_lr_seed_budget_cohort_comparator_rescue=True,goal='ACTIVE'))
table='\n'.join(f"| {k} | {v['macro']:.12f} | {v['ce']:.9f} | {v['positive_to_null']} | {v['null_false_positive']} |" for k,v in metrics.items())
passed=sum(s['gates'].values())
decision='十三门全过，只允许设计完整单seed协议。' if all(s['gates'].values()) else '十三门未全过，本确切配置暂缓，不进全量或多seed。'
text=f'''# 删除同token自乘：固定五路小规模筛选

## 实际流程与本轮差异

项目输入是句子、指定实体对、服务器已有冻结RoBERTa词特征及实体角色；目标为40类
关系识别，包括no_relation。关系条件attention对value加权，输出40个分类分数，
以全40类Macro-F1（每类F1等权平均）为主指标。当前只在原TRAIN8192条拟合、
另4096条评估，句子组隔离；没有新读取VALID/TEST，也不是全量或独立确认。

[已测事实] 上一轮Q均值乘积 .504045671350好于Q中心化 .478786491123，
仍低于强GnativeCov .512054050292。该控制结果未被重命名为成功候选。
本轮冻结一个新开发假设：删除均值乘积中的同token位置自乘项是否有帮助。

| 部分 | 上一轮 | 本轮 |
|---|---|---|
| 查询/基础分类 | 固定语义共享查询和原读出 | 原样复用 |
| 二次项 | M=mu_even*mu_odd，或中心化covariance | U=M−Σalpha²*v_even*v_odd |
| 额外控制 | 无 | QcollisionMass=(1−Σalpha²)*M |
| 初始化/核心/预算 | 零moment decoder、原稳定核心 | 原样复用，各81924 trainables |

mu=Σalpha*v，U等于Σ_{{t!=u}}alpha_t*alpha_u*v_t,even*v_u,odd；不归一、不
调整温度或通道pair。Q/D/G及置换Q都使用相同U，QcollisionMass用于区分去自乘
与单纯注意力集中度调制。五个fit全部结束后才统一评估，旧九个参照直接复用。

## 全部结果与固定判定

| 条件 | 全40类Macro-F1 | CE | 正关系→null | null→正关系 |
|---|---:|---:|---:|---:|
{table}

[已测事实] 固定十三门{passed}/13通过。{decision} 差值和全部40类CM/F1/support
见contrast-analysis.json与class-analysis.json，不选类别、换主指标或删除不利对照。

## 认识与边界

[机制解释] U是已知非对角二次统计，适用于任意归一attention；不能仅凭公式或成绩
称其为新T、语义subject/object绑定、token顺序建模或量子独占资源。Q相对去相干及
Gaussian对照的结果才涉及本轮相对贡献，经典衍生另记自己的方法身份和证据等级。
[尚未证明] 本轮仍是暴露开发集的单seed筛选，不接受新L3/L4-T，不转移原限定L4。
逐例语义、attention和训练轨迹没有纳入聚合报告；不从CM反推机制案例。

新增U/mass独立双token公式、全参梯度、非零query差分、单token=0、同value恒等式、
padding、零moment共享端点和clone actual8检查记录于fixtures.json。首轮Astra/high
BLOCK指出差分可在两边均0时假通过；已保留原记录，加入两边均有限且绝对值>1e-10
约束，继续要求误差<1e-7；第二轮PASS。静态审核与实际执行证据分别保存。
代码修正仅涉及fixture，没有改变科学公式、数据、预算或十三门。

seed13，每路1024updates/b128/micro4，原AdamW/FP32-complex64/TF32off；
五fit5120物理更新，耗时{s['seconds']:.3f}秒，双GPU动态队列，gen{closed['generation']}
已可信归还。核心50ce858f/host238f7e68未改，不做续训/COS生命周期矩阵。
十四路CM独立误差≤{verified['independent_macro_max_error']:.3g}，0新COS，Windows0模型/
特征/逐例数组。短筛选仅Git保留代码/配置/聚合报告，未自动归档checkpoints。

## 下一步

唯一后继决策见next-direction.json。若任一门失败，不对本确切profile追加归一化、
集中度、通道pair、rank、token、scale、LR、seed、budget、cohort或删除对照救援。
所有先前负结果保持原状，不能据本轮否定整个方法族或接受量子/经典新L3/T。
'''
(r/'analysis.md').write_bytes(text.encode())
print(json.dumps(dict(gate=s['gate'],gates_passed=passed,metrics=metrics,contrasts=contrasts,classical_development_candidate=classical)))
