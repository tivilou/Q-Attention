"""Explain all frozen arms; keep utility, attribution and classical claims separate."""
from pathlib import Path
import json

r = Path(__file__).resolve().parent
read = lambda n: json.loads((r/n).read_bytes())
save = lambda n,v: (r/n).write_bytes((json.dumps(v,indent=2)+'\n').encode())
s,cfg,verification,closed = (read(n) for n in ('summary.json','design.json','verification.json','allocation-closeout.json'))
assert verification['status']=='PASS' and closed['state']=='released'
previous=read('../semantic-query-sharing-20261125/summary.json')
rows={'new/'+k:v for k,v in s['results'].items()}
rows.update({'reused/'+k:v for k,v in s['reused_results'].items()})
assert len(rows)==9 and len(s['gates'])==8
metrics={}
for name,row in rows.items():
    cm=row['confusion_matrix']
    metrics[name]=dict(macro=row['all40_macro_f1'],ce=row['ce'],correct=sum(cm[i][i] for i in range(40)),
        positive39_macro=sum(row['per_class_f1'][1:])/39,null_f1=row['per_class_f1'][0],
        positive_to_null=sum(cm[i][0] for i in range(1,40)),null_false_positive=sum(cm[0][1:]),
        parameters=row['parameters'])
q=metrics['new/quantum_shared_covariance']['macro']
d=metrics['new/dephased_shared_covariance']['macro']
g=metrics['new/gaussian_shared_covariance']['macro']
old={k:v['all40_macro_f1'] for k,v in previous['results'].items()}
differences=dict(Qjoint_minus_Djoint=q-d,Qjoint_minus_Gjoint=q-g,
    Qjoint_minus_Qshared=q-old['quantum_shared'],
    Qjoint_minus_Qmeanproduct=q-metrics['new/quantum_shared_mean_product']['macro'],
    Qjoint_minus_Qpermuted=q-metrics['new/quantum_permuted_covariance']['macro'],
    Qjoint_minus_QnativeCov=q-metrics['reused/native_quantum_covariance']['macro'],
    Qjoint_minus_GnativeCov=q-metrics['reused/native_gaussian_covariance']['macro'],
    Djoint_minus_Dshared=d-old['dephased_shared'],Gjoint_minus_Gshared=g-old['gaussian_shared'])
differences['sharing_cov_gain_Q_minus_D']=differences['Qjoint_minus_Qshared']-differences['Djoint_minus_Dshared']
differences['sharing_cov_gain_Q_minus_G']=differences['Qjoint_minus_Qshared']-differences['Gjoint_minus_Gshared']
labels=read('label-codes.json')['labels']
counts=read('../prior-adjusted-native-training-20261125/design.json')['fit_class_counts']
classes=[]
for i,label in enumerate(labels):
    f={k:v['per_class_f1'][i] for k,v in rows.items()}
    classes.append(dict(class_id=i,label=label,fit_count=counts[i],
        assessment_support=rows['new/quantum_shared_covariance']['class_support'][i],f1=f,
        Q_minus_D=f['new/quantum_shared_covariance']-f['new/dephased_shared_covariance'],
        Q_minus_G=f['new/quantum_shared_covariance']-f['new/gaussian_shared_covariance']))
save('class-analysis.json',classes)
save('contrast-analysis.json',dict(metrics=metrics,descriptive_macro_differences=differences,
    gates=s['gates'],stage_decision=s['gate'],
    limitation='Same exposed development cohort/single seed. Differences are descriptive, not interaction identification, significance or independent confirmation.'))
quantum_utility=q>old['quantum_shared']
classical_candidate=g>old['gaussian_shared'] and g>metrics['reused/native_gaussian_covariance']['macro']
save('primary-disposition.json',dict(stage='small_single_seed',decision=s['gate'],gates=s['gates'],
    all_nine_conditions_retained=True,new_fits=5,physical_updates=5120,
    quantum_baseline_relative_screen=quantum_utility,
    quantum_attributed_L3='NOT_ACCEPTED',classical_derivative=dict(
        identity='gaussian_shared_covariance',candidate_for_own_protocol=classical_candidate,
        grade='DEVELOPMENT_SCREEN_ONLY',replaces_quantum_goal=False),
    full_data='NOT_STARTED',multi_seed='NOT_STARTED',new_L3='NOT_ACCEPTED',new_L4_T='NOT_ACCEPTED',
    no_rank_token_pair_channel_scale_threshold_lr_seed_cohort_budget_control_rescue=True,goal='ACTIVE'))
table='\n'.join(f"| {k} | {v['macro']:.12f} | {v['ce']:.9f} | {v['positive_to_null']} | {v['null_false_positive']} |" for k,v in metrics.items())
deltas='\n'.join(f'- {k}：{v*100:+.6f}个百分点。' for k,v in differences.items())
passed=sum(s['gates'].values())
decision=('只允许设计完整单seed协议，未自动启动全量或多seed。' if all(s['gates'].values()) else
    '这个确切联合配置暂缓，不进入全量或多seed；不挑类别、换主指标或扫描参数救援。')
text=f'''# 共享关系查询与中心化证据：固定五路小筛选

## 实际工作流与这次改动

本项目从句子及指定实体对识别40种关系（包括no_relation）。使用服务器上已有的
冻结RoBERTa词特征、实体角色及词mask，训练关系条件注意力和分类读出，以全40类
Macro-F1衡量各类等权表现。当前阶段仅使用原TRAIN中固定8192条拟合、另4096条评估，
句子组不重叠；没有新读取VALID/TEST。这是开发筛选，不能代替全量或独立确认。

[已测事实] 上一轮语义共享使Q优于其原生版本，但仍输给同样共享的D/G。
本轮把该固定共享查询与先前的固定二阶读出合并，检验一个联合假设：是否能保留
公平经典/去相干对照下的收益。原先两轮确切配置仍关闭，没有改rank、分词、seed或预算。

| 部分 | 上一轮 | 本轮 |
|---|---|---|
| 关系查询 | 固定40×8 label-name code乘8×12参数 | 原样复用 |
| 读出 | 注意力加权value均值线性读出 | 加32对相邻value通道的中心化二阶统计 |
| 新参数 | 无moment读出 | 40×32 moment_decoder，全0初始化 |
| 训练核心/优化器/数据 | 固定版本 | 原样复用 |

同一attention权重alpha下，mu=Σalpha*v，新增C_j=Σalpha*v[2j]*v[2j+1]
−mu[2j]*mu[2j+1]，最后以relation-specific系数线性读出。mean-product对照固定用
正号mu[2j]*mu[2j+1]替代C；code置换对照打乱39正类对应、保持null行。Q/D/G同样
获得公开关系名结构及二阶统计，均81924 trainables；不声称有效容量相同。
0初始化使各模型初始输出精确等于对应gen96共享端点。

## 全部固定结果

| 条件 | 全40类Macro-F1 | CE | 正关系→null | null→正关系 |
|---|---:|---:|---:|---:|
{table}

{deltas}

[已测事实] 固定八门{passed}/8通过。{decision}
Q必须严格优于四个新对照及Qshared/QnativeCov/GnativeCov/Gshared四个固定参照；
仅优于自身Q基线不足以确认量子独立贡献。所有九路及40类CM/F1/support完整保存。
新增公式核对使用真实小批量，记录于fixtures.json；案例级预测、attention及优化轨迹
未纳入此轮聚合收尾，不补写或从CM反推个例。机制实例解释仍有这一限制。

## 已获得与尚未获得的认识

[已测事实] 各对照差值和完整类级结果见contrast-analysis.json与class-analysis.json。
共享前后增益差只是同一开发集上的描述，不能据此声称相干导致互补、统计交互被识别、
显著性或泛化规律。CE和null错分帮助描述行为，不能替换冻结的Macro-F1门槛。

[尚未证明] 小筛选尚不支持新L3/L4-T。低秩语义共享和二阶pooling是已知工具，组合
本身未建立实质理论创新、量子硬件优势或相干机制的独立贡献。原限定L4和全部历史保留。

按照新skill分别处理贡献：量子方案相对自身基线的效用、相干归因、经典衍生候选独立
记录。经典组合即使优于Q，也只成为自身协议的候选，仍需强基线、关键消融、公平经典
替代和分级投入后的适当确认，不能自动获得L3/L4或替代当前量子L4-T目标。

## 执行和投入

seed13，每路1024updates/b128/micro4，原AdamW/FP32-complex64/TF32off。
五fit全部完成后才统一评估，5120实际更新，耗时{s['seconds']:.3f}秒；双GPU动态队列，
gen{closed['generation']}已可信归还。四个旧参照完全复用，无重复baseline。
稳定核心50ce858f/host238f7e68未改，只检查新增组合端点、独立公式/梯度、padding和
clone actual8更新；A6API Astra6.2/high静态审核通过，实际fixture另记，二者不混称。
九路CM独立重算误差≤{verification['independent_macro_max_error']:.3g}。0新COS，Windows
0权重/特征/逐例数组；短且可重启的失败筛选仅Git保留代码、配置和聚合报告。

## 下一步

唯一推荐方向在next-direction.json冻结：检验删除均值乘积里的同token自乘项，
U=M−Σalpha²*v_even*v_odd，而非继续调covariance。新增Q/D/G/置换U及
Q集中度乘M五臂，后者区分纯注意力集中度效应；保留本轮全部九个旧参照，固定
十三个全40类Macro比较。原8192/4096、seed13、每臂1024updates；十三门全过
才设计完整单seed，任何门失败就暂缓该确切配置。尚未实现或启动这一后继实验。
U身份是已知非对角二次统计，不是新T，也尚未证明多个真实关系线索存在互补。
外部Astra/high仅提供一个建议，主模型加入集中度对照并独立冻结方案；未获文献新颖性。
具体已知二次读出等价和观察边界见mechanism-note.md；经典衍生路线未自动替换目标。
若八门未全过，不为本确切profile追加rank、token、pair、通道、scale、threshold、LR、
seed、cohort、budget或删除对照救援，也不据一次失败否定整个方法族。
'''
(r/'analysis.md').write_bytes(text.encode())
print(json.dumps(dict(gate=s['gate'],gates_passed=passed,contrasts=differences,
    metrics=metrics,classical_development_candidate=classical_candidate)))
