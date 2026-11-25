"""Explain the complete task error map and fixed-fit readout, without promotion."""
from pathlib import Path
import json

r=Path(__file__).resolve().parent
load=lambda n:json.loads((r/n).read_bytes())
v,fit,checks,closed=(load(n) for n in ('confusion-map.json','summary.json','fit-verification.json','allocation-closeout.json'))
assert checks['status']=='PASS' and closed['state']=='released'
modes=('quantum','dephased','gaussian_base','gaussian_covariance')
classes=[]
for row in v['classes']:
    item=dict(row)
    item['fit_f1']={k:fit['results'][k]['per_class_f1'][row['class_id']] for k in modes}
    if row['class_id']:
        i=row['class_id']-1
        item['fit_forced_recall']={k:fit['results'][k]['positive39_confusion'][i][i]/row['fit_count'] for k in modes}
    classes.append(item)
ordered=sorted(classes[1:],key=lambda x:(x['fit_count'],x['class_id']))
groups=[]
for j in range(3):
    rows=ordered[j*13:(j+1)*13]
    groups.append(dict(group='fit_support_tertile_'+str(j+1),class_ids=[x['class_id'] for x in rows],
        labels=[x['label'] for x in rows],fit_min=min(x['fit_count'] for x in rows),fit_max=max(x['fit_count'] for x in rows),
        assessment_support=sum(x['assessment_support'] for x in rows),
        fit_macro={k:sum(x['fit_f1'][k] for x in rows)/13 for k in modes},
        assessment_macro={k:sum(x['primary_f1'][k] for x in rows)/13 for k in modes},
        Q_minus_Gaussian_all40_macro_contribution=sum(x['Q_minus_Gaussian_macro_contribution'] for x in rows)))
save=lambda n,x:(r/n).write_bytes((json.dumps(x,indent=2)+'\n').encode())
save('class-analysis.json',classes)
save('support-analysis.json',dict(definition='Positive classes sorted by fit count,classID; three consecutive groups of13. Descriptive post-outcome summaries, no new primary or selection gate.',
    groups=groups,null_Q_minus_Gaussian_all40_macro_contribution=classes[0]['Q_minus_Gaussian_macro_contribution']))
save('primary-disposition.json',dict(status='DIAGNOSTIC_COMPLETE',new_trainable_plugin='NOT_SELECTED',
    optimizer_updates=0,forward_rows=33280,new_valid_test=0,cos_writes=0,
    promotion='NONE',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',goal='ACTIVE'))
table='\n'.join(f"| {k} | {fit['results'][k]['macro']:.6f} | {v['rows'][k]['primary_macro']:.6f} | {fit['results'][k]['positive_forced_correct']} / 2792 | {v['rows'][k]['positive_forced_correct']} / 1365 |" for k in modes)
tertiles='\n'.join(f"| {g['group']} ({g['fit_min']}–{g['fit_max']}) | {g['assessment_support']} | "+' | '.join(f"{g['fit_macro'][k]:.4f} / {g['assessment_macro'][k]:.4f}" for k in modes)+' |' for g in groups)
top=sorted(classes[1:],key=lambda x:x['Q_minus_Gaussian_macro_contribution'])[:8]
rare='\n'.join(f"| {x['label']} | {x['fit_count']} / {x['assessment_support']} | "+' | '.join(f"{x['fit_f1'][k]:.4f} / {x['primary_f1'][k]:.4f}" for k in modes)+' |' for x in top)
edges=v['all_observed_directed_errors'][:10]
confusions='\n'.join(f"| {e['true_label']} → {e['predicted_label']} | "+' | '.join(str(e['counts'][k]) for k in modes)+' |' for e in edges)
text=f'''# 原生模型的关系混淆与稀有类拟合诊断

## 问题与证据范围

gen94已知prior修正关闭：漏检下降但误报大增、主Macro降低。本轮先区分关系类别排序
与null判断，再检查原固定8192fit上的表现，避免把“少样本所以过拟合”当既定事实。
模型、source、最终checkpoint和原训练预算都未改，没有新增训练或学习规则。
该诊断只定位下一问题；不晋级任何候选，不接受新L3/L4-T。

## 四路完整结果

| 模型 | 原fit全40Macro | 原assessment全40Macro | fit强制正类排序正确 | assessment强制正类排序正确 |
|---|---:|---:|---:|---:|
{table}

强制正类排序只分析真实正关系，使用gold标签，不能部署为分类器或替代all40主指标。
assess来自已有4096输出，fit为同一已完成1024-update模型的8192原训练例，无更新。
训练/评估差距是描述，不隔离优化、数据难度、统计方差、query、readout或量子因果。

## 全39类按训练支持排序的三等分

固定按fit_count、classID排序，每13类一组，仅作事后描述；跨边界相同count按ID划分。
完整class-analysis保留全部40类，不能只用下面的稀有组或最高负差类决定成功。

| 支持分组（每类fit数量） | assessment总数 | Q fit / assess | D fit / assess | G fit / assess | Gcov fit / assess |
|---|---:|---:|---:|---:|---:|
{tertiles}

最稀少13类只有59条assessment，它们的离散F1波动不能当独立机制确认或总体理论。
这些是已有固定小筛选的性质；没有因结果换样本、补seed或为旧配置放宽晋级门。

## Q相对Gaussian的最大主Macro负贡献类

按全40主指标逐类F1差/40排序，下面只用于解释，完整40类均已保存。

| 关系 | fit / assessment支持 | Q fit / assess F1 | D | G | Gcov |
|---|---:|---:|---:|---:|---:|
{rare}

## 条件化关系排序的常见混淆

下列按Q在已知正类中的错误数量排序，不是按主Macro贡献排序，不能混为同一问题。
完整39×39及全部观测到的有向错误边、Q独对/对照独对配对均保留。

| 真关系 → 误判关系 | Q | D | G | Gcov |
|---|---:|---:|---:|---:|
{confusions}

## 身份与执行

关系ID从原source的q_rmea_entity_labels.json按原loader的sorted规则恢复：
40类/no_relation在0，contract SHA {v['label_contract_sha256']}。
三个已有private npz SHA、同4096indices和原cohort8fb8393b已核对；四路主CM精确等于
已有公共报告，逐类Macro误差≤{load('verification.json')['max_macro_error']:.3g}。
fit回放先复现每路原assessment前128条输出与argmax，再按原batch128/micro4、FP32、
TF32off回放8192fit。全部checkpoint SHA前后保持、fit CM独立核对，最大Macro误差
{checks['max_macro_error']:.3g}；没有新模型语义、optimizer或checkpoint格式变化。

运行{fit['seconds']:.3f}秒、33280条forward、0更新，gen{closed['generation']}可信归还。
复用原factory和host238f7e68，检查仅针对新窗口的身份、stored-output parity与CM，
没有调用外部代码审核，也没有将本机检查冒称独立实证/内容审核；没有生命周期矩阵。
0新VALID/TEST、0COS；Windows仅代码、类名、聚合计数与指标，无模型/特征/逐例数组。

原限定L4与全部历史保留，新L3/L4-T未接受，目标ACTIVE。只在明确任务缺口及其
区分预测后选下一插件；不重开已关闭null-bias、prior-strength或旧完整配置。
'''
(r/'analysis.md').write_bytes(text.encode())
print(json.dumps(dict(fit_macros={k:x['macro'] for k,x in fit['results'].items()},groups=groups,
    selected_problem='UNDECIDED; interpret fit and assessment jointly before choosing mechanism')))
