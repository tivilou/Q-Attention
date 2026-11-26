from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
r=Path(__file__).resolve().parent;root=Path('E:/Claude.Code/Paper.Writing/Q-Attention')
read=lambda n:json.loads((r/n).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda n,x:(r/n).write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
s=read('summary.json');v=read('verification.json');cfg=read('design.json');review=read('review.json')
assert v['status']=='PASS_INDEPENDENT_SCREEN_AUDIT' and read('allocation-closeout.json')['state']=='released'
assert review['response']['verdict']=='PASS_WITH_RISK' and review['model']=='gpt-6.2-astra' and review['effort']=='high'
assert s['gate']==v['gate'] and v['gate_count']==10
positive=all(v['gates'].values())
next_action=('冻结同一原核心完整单seed协议：确认全量数据/训练预算/最终checkpoint及公平native-Q、D、real8、real12、Gaussian/Gcov关键对照的实际可复用范围；先前置独立确认规则与机制预测，沿用同一插件及runner。当前小筛选仅准协议设计，不直接认定L3或启动补seed。' if positive else
 '只审计gen106固定最终Q/D/real8三臂的逐类与配对错误：复用已存4096 TRAIN logits，0新模型前向/0训练/0容量；报告所有40类及Q独有修复/损害，检验此前合成路径可分性是否存在任务证据。不得改变插入位置、深度、初始化、温度、seed、数据或控制来恢复此已关闭profile；审计后再择至多一个实质不同的原量子核心假设或NONE。')
next_dir=dict(status='NEXT_FULL_SINGLE_SEED_PROTOCOL' if positive else 'NEXT_FROZEN_SCREEN_ERROR_AUDIT_NO_NEW_FORWARD',action=next_action,
 exact_profile=cfg['profile'],screen_gate=s['gate'],source_pins={n:sha(r/n) for n in ('plugin.py','adapter.py','design.json','summary.json','verification.json','validation.json','input-binding.json')},
 resources_next=dict(GPU=0,new_task_forward=0,fit=0,new_capacity=0),
 stop_scope='Exact middle-placement/ansatz/budget fixed profile; retain all earlier positive and negative history.',
 retained_stops=['gen98','gen102','gen104','gen105']+([] if positive else ['gen106']),
 claim_ceiling='Exposed TRAIN one-seed development screen; no independent task confirmation, free-family separation, new theory or L3/L4-T.',
 originalL4='RETAINED',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE')
write('next-direction.json',next_dir)
rows='\n'.join('| '+k+' | '+format(row['all40_macro_f1'],'.12f')+' | '+format(row['ce'],'.9f')+' | '+str(row['parameters'])+' |' for k,row in s['results'].items())
references='\n'.join('| '+k+' | '+format(row['all40_macro_f1'],'.12f')+' | '+str(row['parameters'])+' |' for k,row in cfg['references'].items())
fails='、'.join(k for k,x in v['gates'].items() if not x) or '无'
text=f'''# 中间关系电路固定小筛选：{s['gate']}

本项目用固定文本编码器产生每个原词的1024维特征，加4维实体角色信息，经12角词电路和40个类别关系操作计算词权重。词权重加EPS后在句内归一化，池化h-only线性64维值，经40类decoder与bias得到logits，以交叉熵训练。当前筛选只使用既有TRAIN中8192条拟合、4096条评估，最终1024更新计算包含全部40类的宏平均F1；VALID/TEST未读。本轮保留同一原量子核心，不引入新的编码输入、深度、词值分支或训练优化器。

| 项目 | 原生位置消融 | 本轮中间关系位置 |
|---|---|---|
| 概率 | abs(<0|Q† W2 W1|0>)² | abs(<0|W2 Q† W1|0>)² |
| 共同项 | 原prepare/effect、pi*tanh、线性值、EPS1e-6、decoder/bias | 同左；完整初始tensor逐项相同 |
| 控制 | 已完成native Q/D、real12、Gaussian及强矩参考 | 新D块边界去相干、新real8同位置，固定映射 |

[已测事实] 插件先经一次A6API gpt-6.2-astra/high受影响机制审查PASS_WITH_RISK（{review['elapsed_seconds']:.3f}s）。按审查要求补上FP32实际参数变化断言及缺失参考源pins；科学公式未改。56项源码/输入/参考文件全hash通过，八个参考原始初始化摘要复现、共享value/decoder/bias逐tensor相同；native Q全状态与新三臂一致。旧参考均未重训。

[已测事实] 原Linux训练环境的CPU独立dense算子/相位还原复振幅/显式density-D/real8有序实矩阵检查通过。32词角×40关系角，FP64概率最大误差1.1102230246251565e-15，FP32最大5.779827247920366e-7，低于预设1e-11/2e-6。三个头各用固定前8条真实FIT样本及labels执行丢弃的FP64与FP32 clone一步更新：梯度有限、有限差分通过、参数确实变化，原初始模型未动。Windows QT的首次初始摘要核对失败被原样保留；训练环境复现原冻结摘要，未替换期望摘要或重新选择初始化。

任务样本逐层训练轨迹/完整sample-trace.v2未采集；本轮是开发效用筛选，不声称trace-complete机制归因。已保存的真实8条clone有限差分/零梯度/更新记录见validation.json；全4096三臂logits、完整checkpoint仅留原服务器，私有diagnostic摘要见summary.json。人工角独立oracle不代表自然文本或编码器训练轨迹。

## 固定结果

seed13，AdamW lr.001/wd.01/betas(.9,.999)/eps1e-8/clip1，batch128，FP32/complex64、deterministic、TF32off。三臂各1024更新，共3072；全部拟合完成后才做最终评估。microbatch配置继承4，但复用的dense_batches路径实际整batch执行，与旧参考路径一致。强Gcov是82308注册参数，新三臂81028，差异保留。

| 新臂 | all40 Macro-F1 | CE | 注册参数 |
|---|---:|---:|---:|
{rows}

| 复用参考 | all40 Macro-F1 | 注册参数 |
|---|---:|---:|
{references}

[已测事实] 冻结十门通过{v['pass_count']}/10；失败门：{fails}。独立用40×40整数混淆矩阵及Fraction重新计算每类和总Macro，十个严格大于判定全部重现；平局视为失败。CE为次指标，全部类及零F1保留，无择seed或择指标补救。[verification.json](verification.json)

[机制解释] Q/D末词段三个RZ只带整体相位，路由梯度精确0；D还失去首词段三个RZ及query末层三个RZ。AdamW仍可衰减这些已注册参数。real8采用两段各2层RY+ring、关系4层RY+ring，门数与几何不同。注册数相同不能推出有效容量或复杂度相同。完整Q矩阵的真实adjoint及D块边界含义见plugin.py与review.json。

[尚未证明] 合成终态相同而中置响应不同，只是固定preparer/utility可分性，未证明自由重参数化模型族、自然任务误差原因或非平凡新理论。旧vec/Choi/realification等价仍成立，无计算优势结论。当前没有新L3/L4-T，原限定L4保留。

## 决策与唯一下一步

{s['gate']}。{next_action}

成功判据及kill gate保持事先冻结：Q必须严格高于新D/real8和全部八个参考；任一失败关闭本确切位置/ansatz/预算，不位置-depth-seed-temperature-cohort-budget-control救援。下一步0GPU/新forward/fit/容量，仅分析或前置协议；后续任何新实验仍走同一共享调度。gen106 allocation已trusted released，旧gen98/102/104/105及全部正负历史保留。

证据：[design.json](design.json)、[summary.json](summary.json)、[validation.json](validation.json)、[input-binding.json](input-binding.json)、[review.json](review.json)、[execution.json](execution.json)、[next-direction.json](next-direction.json)。原公式数学证据沿用上一checkpoint，未作为新任务正效应。
'''
(r/'analysis.md').write_bytes(text.encode())
# Exact fresh progress snapshot; publication refuses concurrent drift.
w=root/'.ai-progress/workstreams/quantum-projector-v2'
backup=r/'progress-backup';backup.mkdir(exist_ok=False)
baseline=[]
for name in ('state.md','critical-invariants.md','plans/qattn-l5-theory-empirical-v006.json'):
 p=w/name;b=backup/name;b.parent.mkdir(parents=True,exist_ok=True);b.write_bytes(p.read_bytes())
 baseline.append(dict(path=str(p),backup=str(b),sha256=sha(p),recorded_at=datetime.now(timezone.utc).isoformat()))
write('progress-baseline.json',baseline)
names=['plugin.py','adapter.py','screen.py','design.json','design-pre-admission.json','adapter-reviewed.py','adapter-checks.diff',
 'review.json','review-adoption.json','selection.json','dispatch.json','consult.py','consult-execution.json',
 'preflight.json','model_preflight.py','model_preflight-execution.json',
 'prepare.py','adopt_review.py','adopt_review-execution.json','validate.py','validate-execution-initial-mismatch.json',
 'validation.json','operator-validation.json','input-binding.json','checks-execution.json','bind_inputs.py',
 'admission.json','summary.json','fixtures.json','RUN_COMPLETE','execution.json','resource-return.json','allocation-closeout.json',
 'verification.json','verify_results.py','verify_results-execution.json','analysis.md','next-direction.json','execute.py']
for n in names:
 b=(r/n).read_bytes()
 if n.endswith('.json'):json.loads(b)
 if n.endswith('.py'):ast.parse(b.decode())
assert all('.private' not in n and not n.endswith(('.pt','.npy','.npz')) for n in names)
write('package-verification.json',dict(status='PASS_INTERLEAVED_SCREEN_EVIDENCE_PACKAGE',public_evidence_sha256={n:sha(r/n) for n in names},
 newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE'))
print(json.dumps(dict(status='PASS_INTERLEAVED_SCREEN_EVIDENCE_PACKAGE',gate=s['gate'],science_files=len(names),newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')))
