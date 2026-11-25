"""Keep all fixed outcomes and distinguish objective utility from Q attribution."""
from pathlib import Path
import json

r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
s,cfg,verified,closed=(read(n) for n in ('summary.json','design.json','verification.json','allocation-closeout.json'))
assert verified['status']=='PASS' and closed['state']=='released'
rows={'adjusted/'+k:v for k,v in s['results'].items()}
rows.update({'reused/'+k:v for k,v in s['reused_results'].items()})
assert len(rows)==7
save=lambda n,v:(r/n).write_bytes((json.dumps(v,indent=2)+'\n').encode())
metrics={}
for k,v in rows.items():
    cm=v['confusion_matrix']
    metrics[k]=dict(macro=v['all40_macro_f1'],raw_ce=v['ce'],
        correct=sum(cm[i][i] for i in range(40)),null_f1=v['per_class_f1'][0],
        positive39_macro=sum(v['per_class_f1'][1:])/39,
        positive_to_null=sum(cm[i][0] for i in range(1,40)),null_false_positive=sum(cm[0][1:]),
        predicted_null=sum(x[0] for x in cm))
contrasts={k+'_adjusted_minus_native':metrics['adjusted/'+k]['macro']-metrics['reused/'+k]['macro']
    for k in ('quantum','dephased','gaussian_base')}
q=metrics['adjusted/quantum']['macro']
contrasts.update(Qadjusted_minus_Dadjusted=q-metrics['adjusted/dephased']['macro'],
    Qadjusted_minus_Gadjusted=q-metrics['adjusted/gaussian_base']['macro'],
    Qadjusted_minus_Gcov_practical=q-metrics['reused/gaussian_covariance']['macro'])
classes=[]
for i in range(40):
    f={k:v['per_class_f1'][i] for k,v in rows.items()}
    classes.append(dict(class_id=i,fit_count=cfg['fit_class_counts'][i],assessment_support=rows['adjusted/quantum']['class_support'][i],
        f1=f,adjusted_minus_native={k:f['adjusted/'+k]-f['reused/'+k] for k in ('quantum','dephased','gaussian_base')}))
save('class-analysis.json',classes)
save('contrast-analysis.json',dict(descriptive_macro_differences=contrasts,metrics=metrics,gates=s['gates'],stage_decision=s['gate'],
    limitation='Known prior-aware training is not a quantum novelty or a Macro-F1 consistency theorem; no family impossibility or causal diagnosis from this development screen.'))
save('primary-disposition.json',dict(stage='small_single_seed',decision=s['gate'],gates=s['gates'],
    all_seven_conditions_retained=True,new_fits=3,physical_updates=3072,
    full_data='NOT_STARTED',multi_seed='NOT_STARTED',new_L3='NOT_ACCEPTED',new_L4_T='NOT_ACCEPTED',
    no_strength_threshold_lr_seed_cohort_budget_control_rescue=True,goal='ACTIVE'))
table='\n'.join(f"| {k} | {v['macro']:.12f} | {v['raw_ce']:.12f} | {v['positive_to_null']} | {v['null_false_positive']} |"
    for k,v in metrics.items())
deltas='\n'.join(f'- {k}: {v*100:+.6f}个百分点。' for k,v in contrasts.items())
decision=('六门全部通过，只允许设计完整单seed协议，尚未启动全量/多seed。' if all(s['gates'].values()) else
          '六门未全部通过，暂缓本确切full-strength/1024配置，不进入全量/多seed。')
text=f'''# 原生三路的类别先验训练修正

## 问题与实际干预

上一轮三路折叠读出均出现平均CE下降但Macro-F1降低，且更多正关系被判成null。
这只是损失与任务指标之间的开发观察，不能隔离不均衡、优化、正则或量子原因。
本轮恢复原生factorized value/decoder，只对训练目标做一个固定、已知的干预：
CE(raw_logits+log(fit_class_count),y)，评估时精确返回raw_logits，无阈值调参。
Q、原生去相干、Gaussian三路接受同样干预，各81028 trainables；无新增分类器、
特征、采样改变、折叠参数化或阶段冻结。

计数只来自原固定8192fit、全部40类正数，null5400/8192；最少类仅1条，需保留
这个小数据筛选限制。没有读取assessment标签设置prior，没有用VALID/TEST选方法。
本法是作者官方源码确认的Balanced Softmax，不是本项目的理论创新或完整BALMS。
它也不保证Macro-F1最优；balanced accuracy与Macro-F1不同。

## 全部七路固定结果

| 条件 | 全40类Macro-F1 | 原始输出CE | 正关系→null | null→正关系 |
|---|---:|---:|---:|---:|
{table}

{deltas}

固定六门{sum(s['gates'].values())}/6通过。{decision}
必须保留同样受到干预的D/G，不能只用旧经典对照归因于量子。所有40类support/F1/CM
和学习曲线均保留，null错误只作描述，不能替代预声明主Macro。
Gaussian covariance是较强函数类别的实用参照，不是loss路径/容量匹配的消融。
不调prior强度、class weights、阈值、LR、seed、cohort、预算或删控制救援。
小筛选未晋级不等于证明整个先验修正/量子模型家族不可能。

## 预算与执行

同一原TRAIN8192fit/4096assessment、同句组无重叠/全部40类、seed13/1024updates、
b128/micro4/FP32-complex64/TF32off/原AdamW；三新fit全完成后才统一评最终状态。
复用四路精确同cohort/顺序/初始化方法/预算的已完成参照，无重复baseline。
{s['seconds']:.3f}秒/3072实际更新，双2080，gen{closed['generation']}可信归还。
稳定screen50ce858f/host238f7e68字节未改，只检查新loss/训练评估切换、梯度、
计数buffer、NumPy公式、原生端点和克隆真实8行更新，Astra6.2/high确切语义审核。
没有单卡/暂停/多卡/COS恢复矩阵；0新VALID/TEST、0COS，Windows0模型/特征/逐例数组。
七路40类CM独立重算最大误差{verified['independent_macro_max_error']:.3g}。
训练曲线记录的是新Balanced Softmax CE，不能当作普通CE与旧训练曲线直接比较；
表中的assessment CE全部使用raw_logits，可以直接描述比较。

## 理论与评级边界

原作者README与完整loss源码已读，论文全文/定理未读；未冒称新定理、Macro一致性、
近邻区别或量子专属性。旧Q-PHV/Q-HARP toy修复、MGDA单步与calibration配置仍关闭，
本轮不是重命名或调参救援它们。新L3/L4-T仍NOT_ACCEPTED，目标ACTIVE；
原限定L4及全部历史结果保持。下一步由全部固定结果决定，不选择性补seed。

来源：[Ren等官方实现](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/main/loss/BalancedSoftmaxLoss.py)。
'''
(r/'analysis.md').write_bytes(text.encode())
print(json.dumps(dict(stage_decision=s['gate'],gates=s['gates'],contrasts=contrasts,metrics=metrics)))
