"""Summarize every fixed condition after common-state verification."""
from pathlib import Path
import json

r = Path(__file__).resolve().parent
read = lambda n: json.loads((r / n).read_bytes())
s, freeze, checked, closed = (read(n) for n in ('summary.json', 'utility-freeze-verification.json',
    'verification.json', 'allocation-closeout.json'))
assert checked['status'] == freeze['status'] == 'PASS' and closed['state'] == 'released'
rows = dict(s['results'], **{'reused/' + k: v for k, v in s['reused_results'].items()})
q, d, g = (s['results'][k]['all40_macro_f1'] for k in ('quantum', 'dephased', 'gaussian_base'))
contrasts = dict(Q_minus_dephased=q-d, Q_minus_Gaussian=q-g,
                 Q_minus_Gaussian_covariance_reference=q-s['reused_results']['gaussian_covariance']['all40_macro_f1'])
classes = []
for i in range(40):
    f1 = {k: v['per_class_f1'][i] for k, v in rows.items()}
    classes.append(dict(class_id=i, support=s['results']['quantum']['class_support'][i], f1=f1,
        Q_minus_D=f1['quantum'] - f1['dephased'], Q_minus_G=f1['quantum'] - f1['gaussian_base']))
save = lambda n, v: (r / n).write_bytes((json.dumps(v, indent=2) + '\n').encode())
save('class-analysis.json', classes)
save('contrast-analysis.json', dict(descriptive_macro_differences=contrasts, gates=s['gates'],
    stage_decision=s['gate'], utility_hashes_equal_and_unchanged=True,
    limitation='This utility map was learned under uniform pooling. Diagnostic development result, not a universal router ordering or independent L3/T.'))
learning = read('learning-progress.json')
warm = [x['block_mean_ce'][:4] for x in learning.values()]
assert all(x == warm[0] for x in warm), 'Actual identical first256 warmup losses required'
table = '\n'.join(f"| {k} | {v['all40_macro_f1']:.12f} | {v['ce']:.12f} | "
    f"{'前轮不同训练路径实用参照' if k.startswith('reused/') else '本轮共同utility训练'} |" for k, v in rows.items())
ordering = ('Q在这份共同冻结utility上优于两个新路由对照。' if q > d and q > g else
            'Q未同时超过两种新路由对照；共同utility没有消除这份固定协议下的路由差距。')
decision = ('三门均通过，只能列为小数据开发阶段有前途的训练路线，不能直接声称独立贡献。'
    if all(s['gates'].values()) else
    '三个固定门未全部通过，关闭本确切256/768训练配置，不投入全量或追加seed。')
text = f'''# 固定共同词评分后，量子路由能否领先？

## 为什么做这一步

上一轮同词协方差给Q局部收益，但Gaussian更强。各模型同时学习了不同的value和decoder，
所以不能把全部差距直接归因于注意力分配。本轮不再添加读出功能，而是控制词评分。

先用均匀注意力训练value、decoder和bias256步，再完全冻结这些参数，让Q、去相干和
Gaussian各自只训练原生router768步。三路独立重复相同的256步，以复用稳定核心和
现有两卡队列，不引入跨线程可变状态。总共3072次实际更新，各路径1024步。
这是一份有界诊断，不是选定新量子操作或宣称理论创新。

## 全部四路结果

| 条件 | 全40类Macro-F1 | CE | 来源 |
|---|---:|---:|---|
{table}

原TRAIN8192fit/4096assessment、同句组不重叠/40类完整、seed13、b128/micro4、
FP32/complex64、TF32off、原AdamW保持；所有三新fit完成后才评最终1024状态。
gen91 Gaussian-covariance仅作相同总路径更新预算的实用参照，不是相同有效容量、
相同训练路径或相同冻结utility的归因对照。各新模型81028注册参数，阶段一只更新
68200共享参数，阶段二只更新12828 router参数；去相干仍有不变方向。

## 关键控制与新增认知

本机只收集hash和小报告；GPU侧读取三份最终state，确认step257记录的共同utility
SHA完全相同，并且每路final的utility字节仍与各自boundary完全相同。
三路真实前256步训练loss也逐块完全一致。冻结控制通过，故该诊断可解释。

{ordering}
Q−D={100*(q-d):+.6f}个百分点，Q−Gaussian={100*(q-g):+.6f}个百分点。
{decision}

固定共同utility下，任意两路logit差满足sum_i(alpha_A-alpha_B)u_ri，
因此差别确实来自word权重而非重新学习的词评分或bias。
这是已知加权平均恒等式，不是量子专属理论，也不保证分类正确。
若所有有效词utility相等，所有归一化路由严格相同。
均匀预训练可能压掉本来依赖共同训练才能保留的线索；本轮不能排除联合训练、
整个量子家族或未来新机制。不能将该诊断推广为量子无用/经典普遍更优。

全部40类、正负结果、confusion、CE与学习趋势保留。没有换warmup长度、seed、
split、预算、主指标或对照救援；没有启动新的完整实验。

## 执行事实

{s['seconds']:.3f}秒，双2080Ti；gen{closed['generation']}可信归还、exit0、进程终态。
稳定screen50ce858f和host238f7e68字节不变；新增插件使用既有进度/optimizer接口。
只检查阶段切换、inactive参数零grad/零LR、router Adam计数reset及克隆真实8行两阶段更新。
Astra/high确切代码检查PASS，实际fixture PASS，所有四路CM/40类独立重算误差
{checked['independent_macro_max_error']:.3g}。没有训练核心、checkpoint或COS恢复矩阵。
0新VALID/TEST、0COS；Windows没有特征、权重或逐例数组。

新L3/L4-T仍NOT_ACCEPTED，目标ACTIVE，原限定L4及历史保留。
'''
(r / 'analysis.md').write_bytes(text.encode())
print(json.dumps(dict(stage_decision=s['gate'], Q=q, D=d, Gaussian=g,
    contrasts=contrasts, exact_warmup_loss_identity=True)))
