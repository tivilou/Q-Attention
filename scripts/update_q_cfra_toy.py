"""Merge only audited Q-CFRA results; preserve unrelated history and public privacy."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--project-root',required=True)
parser.add_argument('--audit',required=True)
args=parser.parse_args()
project=Path(args.project_root).resolve()
if not (project/'research-web/apps/experiment-results/data.json').is_file():
    raise SystemExit('This incremental updater requires an initialized portal. Archive the audited Q-CFRA documents separately in a code-only checkout.')
audit_path=Path(args.audit).resolve()
a=json.loads(audit_path.read_text(encoding='utf-8'))
assert a['status']=='passed' and a['jobs']==35 and a['sample_trace_records']==350
assert a['config']['seeds']==[13,29,53,71,97]
assert a['implementation_commit']=='6307f3e037ca94b723b5acbce331f03a3b37cc33'
assert a['contract']['source_sha256']=='49bf9de390a160920df4aa177c65890157fa34eed72e242c59455cea956ff5d3'
now=datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')
g=a['aggregates'];d=a['decision'];seeds=a['config']['seeds'];modes=a['config']['modes']
assert d['classification'] in ('kill','inconclusive','promote_to_formal_preparation')
idea='q-cfra-counterfactual-fidelity-reallocation-attention-v1'
family='family-q-cfra-counterfactual-fidelity-reallocation-attention'
baseline='retacred-q-cfra-disabled-toy-5seed-v1'
log_id='q-cfra-counterfactual-fidelity-reallocation-attention-toy'
case_id='q-attention-case-study-q-cfra-toy'
revision='toy-5seed-6307f3e-r1'
ids={mode:(idea if mode=='quantum' else f'q-cfra-{mode.replace("_","-")}-toy-v1') for mode in modes if mode!='disabled'}
labels=dict(disabled='Disabled',quantum='Q-CFRA',classical_cosine='Classical cosine',classical_rbf='Classical RBF',diagonal='Diagonal',shuffled='Shuffled',entity_permuted='Entity-permuted')
outcome={'kill':'按预声明规则关闭本冻结候选','inconclusive':'未达到晋级门槛，结论为 inconclusive','promote_to_formal_preparation':'通过 toy 晋级门槛，可进入正式单 seed 准备'}[d['classification']]
claim=f"仅为630/631/587条真实数据 toy；{outcome}。未进行完整数据正式实验，不构成L1/L2、创新性或量子优势结论。"
headline=f"Q-CFRA平均Macro-F1 {g['quantum']['metrics']['macro_f1']['mean']:.9f}，Disabled {g['disabled']['metrics']['macro_f1']['mean']:.9f}，配对平均差 {d['mean_delta']:+.9f}；{d['positive_seeds']}/5正向。{outcome}。"
log_rel='docs/research-log/20261020T-073818Z-q-cfra-toy.md'
audit_rel='docs/validation/q_cfra_toy_audit.json'
explain_rel=f'docs/idea-explanations/q-cfra-counterfactual-fidelity-reallocation-attention/{revision}.md'
manifest_rel=f'docs/idea-explanations/q-cfra-counterfactual-fidelity-reallocation-attention/{revision}.json'
links=dict(results=f'/apps/experiment-results/?family={family}',log=f'/apps/research-log/?id={log_id}',case_study=f'/apps/experiment-results/?case={case_id}#case-study-title')
explain_href=f'/apps/idea-explanations/?id={idea}'
def content(path):
    return json.loads((project/path).read_text(encoding='utf-8'))
def emit(path,text,immutable=False):
    path=project/path;path.parent.mkdir(parents=True,exist_ok=True)
    data=text.encode('utf-8')
    if immutable and path.exists():
        assert path.read_bytes()==data,f'Immutable artifact already differs: {path}'
        return
    temp=path.with_name(path.name+'.cfra-tmp')
    temp.write_bytes(data);temp.replace(path)
def put(path,value,immutable=False):
    emit(path,json.dumps(value,ensure_ascii=False,indent=2)+'\n',immutable)
def upsert(rows,key,row):
    found=[i for i,x in enumerate(rows) if x[key]==row[key]]
    assert len(found)<=1
    if found: rows[found[0]]=row
    else: rows.append(row)

# Only audited aggregates/counts/hashes enter mounted public folders.
put(audit_rel,a,True)
put('docs/validation/q_cfra_executed_source_manifest.json',dict(implementation_commit=a['implementation_commit'],
    source_sha256=a['contract']['source_sha256'],source_file_sha256=a['source_file_sha256']),True)
put('docs/validation/q_cfra_run_config.json',a['config'],True)

data_path='research-web/apps/experiment-results/data.json'
before=content(data_path);data=copy.deepcopy(before)
dataset=next(x for x in data['datasets'] if x['id']=='real-data-screen')
max_order=max(row.get('added_order',0) for ds in data['datasets'] for row in ds['rows'])
source=f"Q-CFRA bounded toy / source6307f3e / frozen5seeds / {audit_rel} / source identity {a['contract']['source_sha256']}"
b=dict(id=baseline,model='Disabled / Q-CFRA toy',family='classical',dataset_id='real-data-screen',split='test',
    objective='Profile-specific reference;630/631/587; warmup8 + classifier6epochs',
    metrics={name:s['mean'] for name,s in g['disabled']['metrics'].items()},aggregate=g['disabled'],
    seed_results=[dict(seed=r['seed'],metrics=r['test']['metrics']) for r in a['conditions'] if r['mode']=='disabled'],
    seed_protocol='13/29/53/71/97; bounded Re-TACRED only',protocol_label='Q-CFRA fixed real-data toy',
    provenance=source,provenance_status='complete')
upsert(data['baselines'],'id',b)
new_rows=[]
for i,mode in enumerate(ids):
    row=dict(idea_id=ids[mode],title='Q-CFRA · '+labels[mode],added_order=max_order+i+1,added_at='2026-10-20T07:38:18Z',
        status='screen-complete',stage='screen',split='test',metrics={name:s['mean'] for name,s in g[mode]['metrics'].items()},
        aggregate=g[mode],delta={name+'_vs_disabled':s['mean'] for name,s in g[mode]['delta_vs_disabled'].items()},
        seed_results=[dict(seed=r['seed'],metrics=r['test']['metrics'],delta_vs_disabled={name:r['test']['metrics'][name]-next(x for x in a['conditions'] if x['seed']==r['seed'] and x['mode']=='disabled')['test']['metrics'][name] for name in ('macro_f1','accuracy','loss')}) for r in a['conditions'] if r['mode']==mode],
        evidence_tier='L0_diagnostic' if mode=='quantum' else 'screen_control_reference',claim_ceiling=claim,
        configuration=f"Q-CFRA toy · {mode} · dim32/2layers/4heads · logical16/micro2 · five fixed seeds",
        configuration_fingerprint=a['contract']['config_sha256'],dataset_identity=a['contract']['subset_sha256'],
        result_note=headline if mode=='quantum' else f"预声明对照；均值Macro-F1 {g[mode]['metrics']['macro_f1']['mean']:.9f}，配对差 {g[mode]['delta_vs_disabled']['macro_f1']['mean']:+.9f}。",
        provenance=source,provenance_status='complete',idea_family_id=family,
        variant_type='quantum' if mode in ('quantum','diagonal') else 'classical' if mode.startswith('classical') else 'auxiliary',
        variant_role='candidate' if mode=='quantum' else 'matched_control' if mode.startswith('classical') else 'auxiliary',
        paired_idea_id=ids['classical_cosine'] if mode=='quantum' else idea if mode.startswith('classical') else None,
        baseline_id=baseline,research_log_href=links['log'])
    old=next((x for x in dataset['rows'] if x['idea_id']==row['idea_id']),None)
    if old:
        row['added_order']=old['added_order']
        if 'idea_explanation_href' in old:row['idea_explanation_href']=old['idea_explanation_href']
    upsert(dataset['rows'],'idea_id',row);new_rows.append(row)
f=dict(id=family,label='Q-CFRA · 反事实保真度完整注意力重分配',dataset_id='real-data-screen',idea_ids=list(ids.values()),
    baseline_id=baseline,evidence_scope='bounded real-data toy ·5seeds',pair_status='matched',
    matching_scope='nominal_parameters_and_updates_only',stage='screen',status='screen-complete',
    result_note=headline+' 经典控制仅匹配名义参数和更新数；存在冗余仿射方向，不等同于有效容量匹配。',
    provenance_status='complete',provenance_note=source,
    variants=[dict(id=r['idea_id'],idea_id=r['idea_id'],variant_type=r['variant_type'],variant_role=r['variant_role'],
        paired_idea_id=r['paired_idea_id'],split='test',metrics=r['metrics'],aggregate=r['aggregate'],source_ref=source) for r in new_rows])
upsert(data['idea_families'],'id',f)
case=dict(id=case_id,title='Q-CFRA · 五 seed toy 完整私有数值链路',status='archived_not_projected',trace_completeness='complete',
    capture_status='complete_audited_private',viewer_status='archived_not_projected',
    trace_schema='sample-trace.v2',io_contract='Producer-owned stage I/O and complete private NPY arrays',
    semantic_contract_ref=audit_rel,source_ref=source,
    reason=f"350条轨迹、{a['full_value_artifacts']}份完整数值已逐项核验；原句、逐样本预测和完整张量保留私有，静态门户未提供公开原值投影。",
    provenance=dict(status='complete',reason='All35 job/phase identities, checkpoint/summary/trace hashes and numeric lineage validated.',missing_artifacts=[]),
    missing_required_evidence=['privacy-safe static portal projection for original samples and arbitrary numeric coordinates'],
    experiment=dict(run_id='q_cfra_toy/20261020-v1',dataset='bounded Re-TACRED train/valid/test',seeds=seeds,
        code_revision=a['implementation_commit'],target_access='Gold only enters post-prediction supervision/evaluation.'),
    sample_selection=dict(rule='First hash-ordered selected record of each split at initial/best/final plus actual train/update0.',population_scope='bounded toy630/631/587 only',selected_count_per_trace=10,portal_display_count=0),
    coverage={key:('not_applicable' if key in ('retrieval','selection','diagnosis') else 'observed') for key in ('data','preprocess','training','retrieval','scoring','selection','context','generation','evaluation','diagnosis')},
    representation_inventory=dict(records=350,full_numeric_artifacts=a['full_value_artifacts'],numeric_encoding='lossless private NPY'),
    tensor_policy='Private values and source samples remain outside static mounts.',
    entries=[dict(selector=mode,seeds=seeds,trace_count=50,source_ref=source) for mode in modes],samples=[],sample_previews=[])
upsert(data['case_studies'],'id',case)
data['generated_at']=now
note=' Q-CFRA五seed真实数据toy已审计：'+headline
if note.strip() not in data.get('source_note',''): data['source_note']=data.get('source_note','')+note
# Omitted scientific rows and baselines must survive byte-for-byte as JSON values.
for old_ds in before['datasets']:
    updated=next(x for x in data['datasets'] if x['id']==old_ds['id'])
    for row in old_ds['rows']:
        if row['idea_id'] not in ids.values(): assert next(x for x in updated['rows'] if x['idea_id']==row['idea_id'])==row
for row in before['baselines']:
    if row['id']!=baseline: assert next(x for x in data['baselines'] if x['id']==row['id'])==row
for key,current_id in (('idea_families',family),('case_studies',case_id)):
    for row in before[key]:
        if row['id']!=current_id: assert next(x for x in data[key] if x['id']==row['id'])==row
put(data_path,data)

seed_table='| seed | Disabled | Q-CFRA | 配对差 |\n| --- | ---: | ---: | ---: |\n'
for index,seed in enumerate(seeds):
    seed_table+=f"| {seed} | {g['disabled']['metrics']['macro_f1']['values'][index]:.9f} | {g['quantum']['metrics']['macro_f1']['values'][index]:.9f} | {d['paired_deltas'][index]:+.9f} |\n"
control_table='| 条件 | Macro-F1 均值 ± 样本SD | 相对Disabled均值差 |\n| --- | ---: | ---: |\n'
for mode in modes:
    s=g[mode]['metrics']['macro_f1'];delta=g[mode]['delta_vs_disabled']['macro_f1']['mean']
    control_table+=f"| {labels[mode]} | {s['mean']:.9f} ± {s['std']:.9f} | {delta:+.9f} |\n"
gate_text=f"预声明晋级要求平均差≥+.005、至少4/5正向、配对差SD≤.004或不大于Disabled SD。本次配对差SD={d['delta_sample_sd']:.9f}，Disabled SD={d['disabled_sample_sd']:.9f}；shuffled parity={d['shuffled_parity']}；梯度/干预塌缩={d['collapsed_plugin_or_intervention']}。最终classification=`{d['classification']}`。"
mechanism_table='| seed | 投影梯度L2 | 信道梯度L2 | 读出梯度L2 | 最大有效query注意力L1变化 |\n| --- | ---: | ---: | ---: | ---: |\n'
for record in d['mechanism_diagnostics']:
    norms=record['plugin_group_gradient_norms']
    mechanism_table+=f"| {record['seed']} | {norms['projection']:.6g} | {norms['channel']:.6g} | {norms['readout']:.6g} | {record['max_valid_attention_l1_shift']:.6g} |\n"
geometry=a['geometry_diagnostics']
coverage_text='；'.join(f"{split}有正支持的类别{len(a['data_identity'][split]['supports'])}/40，相对训练词表的零支持类别{sorted(set(a['data_identity']['train']['supports'])-set(a['data_identity'][split]['supports'])) or '无'}" for split in ('train','valid','test'))+'。Macro-F1按固定40类取均值，零支持类保留原指标定义，不改成39类口径。来源内因资格筛选而不可用的类别另见审计unavailable_labels，不能与跨split缺类混为一谈。'
geometry_text=(f"固定50条checkpoint/首更新捕获的token级保真度均值范围为 {min(x['fidelity_mean'] for x in geometry):.6g}–{max(x['fidelity_mean'] for x in geometry):.6g}，上界clamp比例范围 {min(x['fidelity_clamped_upper_fraction'] for x in geometry):.6g}–{max(x['fidelity_clamped_upper_fraction'] for x in geometry):.6g}。这些记录包含同一原句的重复回放，不能当作50条独立新测试句，也不能单凭此判定因果机制。")
next_action=('准备完整数据seed13的合作者运行协议，并完成正式runner/exporter/真实启动与失败重试验收' if d['classification']=='promote_to_formal_preparation' else '先审查Q-BCRA的可证伪设计与Bures共识路由接口，保留本次实现作为对照；不追加本冻结Q-CFRA协议的结果驱动调参')
authorization='由项目toy晋级规则授权准备；完整数据仍仅由合作者执行' if d['classification']=='promote_to_formal_preparation' else '仅为推荐，新候选实现和训练尚未授权'
case_rows='| 条件 | 固定sample ID | split/checkpoint/lane | 实际token数 | 私有trace SHA256 |\n| --- | --- | --- | ---: | --- |\n'
for case in a['case_observability']:
    case_rows+=f"| {case['mode']} | `{case['sample_id']}` | {case['split']}/{case['checkpoint']}/{case['lane']} | {case['token_count']} | `{case['trace_sha256']}` |\n"
workflow='''输入是一句话、词序和有序的主语/宾语实体跨度及类型，输出是40类关系中的一个。词表、类型中和token及其中和向量只用选定训练集拟合。每个seed先训练两层小Transformer，再冻结编码器；所有条件继续训练同一个规格的分类器，干预条件额外训练注意力模块。验证集Macro-F1选择checkpoint，测试集只在阶段结束后评估。Macro-F1是各类F1的等权平均，Accuracy是分类正确比例，交叉熵loss越低越好；关系micro-F1另由原项目指标函数计算。无检索、自由文本生成、硬件量子或finite-shot阶段。'''
mechanism='''保留原句，并把主语或宾语的每个token分别替换为同角色/类型的中和token，得到长度、位置和跨度对齐的三个视图。冻结的第一层分别编码它们。共享仿射投影给出两个复振幅，经归一化和随机幺正CPTP信道成为合法密度矩阵；固定去极化项为1e-4。比较原句与两个反事实视图的平方Uhlmann保真度，并计算Bloch向量的实对称协方差。每个query-key对把原始打分、两个保真度logit和协方差编码为两量子比特的旋转角；两层共享局部门与CZ后测量Z0+Z1+Z0Z1。这个读出取代第二层整张打分矩阵，masked softmax分配全部注意力，再使用原来的value与分类头。它没有小幅量子残差。'''
mechanism+=' 保真度是在同一token的原句/中和视图之间计算，并作为key侧特征共享给所有query、head与视图；它不是query与key之间的密度态保真度。各query/head的差异由原始qk分数提供。'
training='''模型dim32、2层、4heads、FF64、dropout0；每seed公共warm-start8epochs，随后每条件6epochs，均为Adam lr.001，逻辑batch16。内存修订后物理microbatch2，累计完整逻辑目标的梯度后只step一次；CE和预测JS按样本数加权，attention JS按有效query数加权。目标是原句CE+.05三视图预测JS+.05三视图attention JS；Disabled也接受同样视图损失和分类器更新。公共编码器被冻结，分类器3880参数；每个非Disabled模块另有150个名义参数：投影132、信道6、读出12。经典控制存在冗余仿射方向，只能称名义预算匹配。'''
cases=f'''预先固定每个split哈希排序后的第一条，在initial/best/final回放，并记录首个真实逻辑更新；每条件每seed10条，共350条。实际原句、实体、预测与完整数组保留在`{a['run_directory']}`的私有`seed*/<mode>/sample_trace.json`及NPY文件中。下表只公开长度、定位与哈希。

{case_rows}
审计确认这些病例在不同条件/seed间的输入角色哈希一致。训练记录的梯度/损失对应完整逻辑batch，逐样本内部张量仅捕获第一条，不能把它称为完整逐样本微批次轨迹。尚未提供静态门户原句/任意坐标查看器，故viewer状态是archived_not_projected；捕获本身完整，公开文档的案例可读性仍有限。个案与50条重复checkpoint观察只支持执行诊断，不替代总体效用或因果归因。'''
result=f"[已测事实] {headline}\n\n{seed_table}\n{control_table}\n{gate_text}\n\n{claim}"
lessons='''[已测事实] 数值、完整轨迹与恢复门禁通过；这是工程证据，不是任务收益。首次真实数据canary在1GiB容器被系统终止，原部分目录保留；之后通过有等价性测试的microbatch累积和共享门精确合成完成执行。

[诊断支持] 预声明的首次逻辑更新中，投影/信道/读出梯度和注意力移动按冻结阈值逐seed记录，实际值见审计decision.mechanism_diagnostics。保真度/协方差/对易子诊断来自固定选定案例，不外推为所有测试句子的机制规律。

[尚未证明] 实体中和是否保留关系线索、保真度是否过于饱和、完整重分配是否抹去基础注意力，都是竞争解释。区分它们需要预先设计的类型保留性标注、冻结checkpoint的几何敏感性/读出消融；本轮没有执行这些新实验。不得仅从F1反推失败的唯一原因。'''
lessons+='\n\n'+mechanism_table+'\n[诊断支持] '+geometry_text
lessons+='\n\n[已测事实] 本轮screen在seed29 shuffled第4轮后随服务器失联中断，日志未记录Python异常，具体原因未确认。重连后逐项核对源码、配置、子集、阶段、warm-start、Adam/RNG及12个完成marker，再按原目录检查点续跑；恢复回执见 /docs/validation/q_cfra_screen_recovery.json。中断与恢复不改变冻结实验或选模规则。'
next_text=(f'''推荐恰好一项：{next_action}。

- 假设：本轮冻结结构在完整数据seed13仍能得到正的预声明Macro-F1差；这是待验证外推。
- 最小实验：先准备并验收可移植runner、分支/数据/环境preflight、白名单exporter与中文合作者手册；随后仅由合作者执行正式seed13。
- 成功门槛：包内真实启动canary与注入失败后的同一完成run导出重试通过，正式结果满足项目单seed效用、干预与泄漏门槛。
- kill门槛：正式主指标非正或数据/干预/证据门禁失败；不以辅助条件替换主候选。
- 资源：正式训练预算与GPU适配在handoff中单独冻结，不从本次CPU toy推算。
- 不确定性：完整数据效用、多seed复现、量子归因和创新性尚未建立。
- 授权：{authorization}。''' if d['classification']=='promote_to_formal_preparation' else f'''推荐恰好一项：{next_action}。

- 假设：Bures几何的共识路由能表达跨视图稳定信息；这是未验证的下一候选，不继承Q-CFRA效用。
- 最小检查：先锁定独立路由接口、退化条件、同预算经典/对易对照和无标签数据流，再决定是否产生新的toy协议。
- 成功门槛：数学与梯度/掩码检查通过，明确可区别当前候选的中间量及完整轨迹契约。
- kill门槛：路由退化、参数不可观测、标签泄漏或无法构造公平对照。
- 资源：设计/只读证据检查约一个工作时段；后续训练预算尚未批准。
- 不确定性：尚无新候选实测结果或确认创新性。
- 授权：{authorization}。''')
ranked_next=('候选排序：1.冻结Q-CFRA正式单seed准备；2.Q-BCRA只保留设计候选；3.Q-MERA-T只保留设计候选。后两项均不在本轮执行范围。' if d['classification']=='promote_to_formal_preparation' else '候选排序：1.Q-BCRA的可证伪设计检查；2.Q-MERA-T的语义层级与公平性设计；3.保留Q-CFRA作为冻结结构对照。后两项均不在本轮执行范围。')
log=f'''# 20261020T-073818Z | {log_id} | Q-CFRA五seed真实数据toy

时间依据：启动回执2026-10-20T07:38:18.050323Z；本条在完整35job审计后生成。实验完成与公开Case Study阅读器接入是两件独立状态。

## 项目工作流总览

{workflow}

## 一句话结论

[已测事实] {headline}

## 这次想法与前一idea的差别

{mechanism}

| 项目 | Q-ECPA | Q-CFRA |
| --- | --- | --- |
| 注意力入口 | 实体条件相干路径的加性分数残差 | 对齐反事实密度几何生成完整分数 |
| 重点风险 | 跨seed收益未复制 | 中和视图语义、保真度敏感性和原注意力替换 |
| 当前比较单位 | 历史完整数据正式三seed | 本轮630/631/587条toy；指标不可直接横比 |

## 冻结验证协议

{training}

从官方train/valid/test按类别、长度≤64、非重叠实体和固定SHA排序各取最多16条；实际630/631/587。{coverage_text} 训练来源58465、验证19584、测试13418条，仅用于来源身份，不作为本轮评估样本量。固定seeds13/29/53/71/97；没有按结果加seed或改门槛。

## 真实案例与证据边界

{cases}

## 结果

{result}

## 从结果学到了什么

{lessons}

## 当前证据判断

阶段screen；项目full-real-data证据口径下L0_diagnostic。{claim} 独立复核为PASS_WITH_RISK；经典有效容量、完整数据效用、量子归因、finite-shot与资源优势未建立。

## 下一步方向

{ranked_next}

{next_text}

## 证据索引

- [审计与完整逐seed指标](/docs/validation/q_cfra_toy_audit.json)
- [执行源码身份](/docs/validation/q_cfra_executed_source_manifest.json)
- [实际冻结配置](/docs/validation/q_cfra_run_config.json)
- [逐步解释]({explain_href})
- [结果工作台]({links['results']})
- [Case Study状态]({links['case_study']})
'''
emit(log_rel,log,True)
explanation=f'''# Q-CFRA：反事实保真度如何生成整张注意力

## 一句话理解

{headline} 它检验的是“去掉实体身份后，表示怎样变化”能否用于完整注意力分配；本轮只到真实数据toy。

## 原任务与动机

{workflow}

前一Q-ECPA正式三seed没有复制历史seed13收益。本次改为更直接的反事实干预与完整分数替换；这是改变作用路径的假设，不是效果承诺。

## 一步步怎么做

{mechanism}

| 步骤 | 输入与输出 | 掩码/边界 |
| --- | --- | --- |
| 三视图 | 词ID B×N → B×3×N | 中和只改实体词身份，不改位置和长度 |
| 冻结第一层 | B×3×N×32 | padding不可作key；真实实体跨度用于池化 |
| 密度几何 | 复振幅2、密度2×2；每token两个F与一个协方差 | 不接收正确关系标签 |
| 两量子比特读出 | B×3×4heads×N×N个四振幅状态 →完整logits | masked softmax，仅有效key |
| 分类 | 原value加权与原第二层 →三种池化拼接96维 →40类 | 标签只在预测后进入loss/评估 |

N≤64，实际训练张量按物理microbatch填充；逻辑batch由梯度累计定义。检索和自由文本生成不适用。

## 原理与公式

归一化复向量ψ生成ρ=ψψ†。E(ρ)=ΣwᵢUᵢρUᵢ†，w=softmax，Uᵢ=Ry(π/4)Rᵢ(θᵢ)，再加固定1e-4去极化。因每个U幺正且权重是概率，输出保持半正定与单位迹。

平方保真度F(ρ,σ)=tr(ρσ)+2√(detρdetσ)衡量两个qubit密度态接近程度。Bloch向量r与两个反事实向量rₛ/rₒ给出a=r−rₛ、b=r−rₒ，协方差c=a·b−(r·a)(r·b)。输入角为πtanh([qk/√d,logitFₛ/4,logitFₒ/4,c])，读出z=〈Z₀+Z₁+Z₀Z₁〉/.7。

共享门先精确合成为4×4算子；与逐门线路的状态和梯度已按float64检查。这里是经典CPU上的量子态模拟，不提供硬件加速结论。

## 哪些参数会学习

{training}

信道混合权重3、旋转角3和读出12都进入优化器。固定项包括去极化、保真度clamp、编码缩放、CZ及温度。非Disabled参数总数4030，Disabled3880；名义模块150，不能据此声称有效容量相同。

## 训练与推理

训练只更新分类器及可选模块，公共编码器冻结。验证集选best epoch；测试只报告该best的原句分类。initial/best/final诊断回放另有各split轨迹，不参与选模或调参。实际Adam更新每条件240次，物理拆批不增加step。

## 帮助理解的例子

{cases}

本说明没有用教学样例替换真实Case Study，也没有公开私有原句或预测。读取完整状态需在原私有run使用带哈希校验的坐标reader。

## 实验设置与对照

630/631/587条、40类输出、五固定seed。{coverage_text} Disabled保留原第二层注意力；cosine/RBF控制用实向量几何；diagonal移除相干；shuffled错配反事实token；entity-permuted中和错位置而保留真实池化跨度。所有模式数据、公共warm-start、逻辑batch、loss与更新数相同。经典仅名义预算匹配，不能承担严格量子归因。

## 实际效果

{result}

## 成功与失败原因

{lessons}

## 一分钟口述

我们的任务是根据句子里的两个实体判断关系。Q-CFRA分别保留原句、去掉主语身份和去掉宾语身份，用同一个冻结编码器得到三份表示，再把它们变成合法密度态。保真度和协方差描述表示对实体身份的敏感性，一个小的两量子比特线路把这些信号变成整张注意力。我们用五个固定种子和六个对照，保留全部关键私有数值并验证了恢复。{headline} 因此当前只按冻结toy门槛作决定，不能把实现通过等同于完整数据效用或量子优势。

## 面试追问

- 为什么用CPTP？共享随机幺正混合给出可检查的合法密度信道，而不是随意命名的矩阵。
- 反事实一定保留关系吗？没有保证；类型保留、中和身份是待验证假设，实体错位和shuffle控制用于限制解释。
- 是否更准确地关注证据？未测直接证据对齐指标，不能从F1称作注意力更准确。
- 为什么不是量子优势？经典有效容量、finite-shot、真实硬件与资源审计都未完成。
- 是否多seed即L2？本项目要求完整数据正式评估与独立确认；五seed短预算toy不能自动升级。

## 证据与版本

执行代码6307f3e，source identity `{a['contract']['source_sha256']}`；config `{a['contract']['config_sha256']}`；subset `{a['contract']['subset_sha256']}`。报告与源码身份分开固定。

- [完整审计摘要](/docs/validation/q_cfra_toy_audit.json)
- [执行源码文件哈希](/docs/validation/q_cfra_executed_source_manifest.json)
- [实际配置](/docs/validation/q_cfra_run_config.json)
- [结果]({links['results']}) · [日志]({links['log']}) · [案例状态]({links['case_study']})

当前文档缺少公开原样本/坐标页面及逐样本全微批次轨迹，失败机理仍需区分性证据；实验本身已完成并审计，文档限制不改变实验完成状态。
'''
emit(explain_rel,explanation,True)
def ref(path): return dict(path=path,sha256=hashlib.sha256((project/path).read_bytes()).hexdigest())
manifest=dict(schema_version='idea-explanation.v1',idea_id=idea,family_id=family,revision=revision,title='Q-CFRA：反事实保真度与完整注意力重分配',
    dataset_id='real-data-screen',checkpoint='五seed真实数据toy ·6307f3e',experiment_status='complete',completeness='partial',
    limitations=['完整原句/逐样本预测/张量保留私有，静态门户未接入原值页面。','仅记录选定第一条的内部状态及完整逻辑batch梯度/损失，没有逐样本全微批次轨迹。','尚无完整数据正式效用或失败原因的区分性消融。'],
    source_path=explain_rel,
    evidence=dict(code=dict(identity=a['implementation_commit']+' / '+a['contract']['source_sha256'],status='verified',reason='逐文件执行源码哈希与实际run source identity已核对；不以报告提交冒充执行代码。',refs=[ref('docs/validation/q_cfra_executed_source_manifest.json')]),
        config=dict(identity=a['contract']['config_sha256'],status='verified',reason='实际run配置和固定subset身份分别审计。',refs=[ref('docs/validation/q_cfra_run_config.json')]),
        report=dict(identity=a['summary_sha256'],status='verified',reason='35jobs、相同预声明指标与门槛、所有完成与阶段身份已审计。',refs=[ref(audit_rel)]),
        trace=dict(identity='350 sample-trace.v2 / '+str(a['full_value_artifacts'])+' full arrays',status='verified',reason='全部私有值已重哈希、解码和坐标回读；公开viewer尚未接入。',refs=[ref(audit_rel)])),
    result_refs=[dict(kind='idea',id=value) for value in ids.values()]+[dict(kind='baseline',id=baseline)],links=links,
    content_review=dict(reviewer='primary agent',checks=['workflow','parameters','results','causal_bounds','target_boundary','examples'],notes='已对照冻结源码、配置、审计结果与trace覆盖；保留案例公开投影和因果解释的限制。'))
put(manifest_rel,manifest,True)

log_data=content('research-web/apps/research-log/logs.json')
upsert(log_data['entries'],'idea_id',dict(idea_id=log_id,source_href='/'+log_rel,source_path=log_rel,date='2026-10-20',
    title='Q-CFRA · 五seed真实数据toy',status='L0_diagnostic / '+d['classification'],tag='bounded real-data toy',summary=headline,
    idea_explanation_href=explain_href,sections=[dict(title='审计结果',tone='fact',body=headline)],
    evidence=[dict(label='完整日志',href='/'+log_rel),dict(label='审计摘要',href='/'+audit_rel),dict(label='代码',value=a['implementation_commit'])]))
log_data['generated_at']=now;put('research-web/apps/research-log/logs.json',log_data)
readme=project/'docs/research-log/README.md'
old=readme.read_text(encoding='utf-8')
tail=old.split('## 时间线',1)[1]
timeline=f"| 2026-10-20 07:38:18 UTC（启动） | `{log_id}` | {headline} | [entry](20261020T-073818Z-q-cfra-toy.md) |"
if log_id not in tail: tail=tail.rstrip()+'\n'+timeline+'\n'
emit('docs/research-log/README.md','# 研究日志\n\n## 当前研究总览\n\n- 最新完成checkpoint：'+headline+'\n- 当前最高历史证据仍保留各idea原有边界；本次Q-CFRA仅为L0真实数据toy，不改写Q-ECPA/Q-EPVG历史单seed记录。\n- 当前推荐：'+next_action+'。\n\n## 时间线'+tail)
registry=content('research-web/registry.json')
for app in registry['apps']:
    if app['id'] in ('experiment-results','research-log','idea-explanations'):
        app['updated_at']=now[:10]
        if app['id']=='experiment-results': app['summary']='按Idea、Baseline和数据集查看审计结果；新增Q-CFRA五seed真实数据toy，保留历史正式证据。'
put('research-web/registry.json',registry)
for app_id in ('experiment-results','research-log','idea-explanations'):
    path=f'research-web/apps/{app_id}/web-app-manifest.json'
    app=content(path);app['updated_at']=now[:10];put(path,app)
receipt=dict(status='completed',idea_id=idea,family_id=family,classification=d['classification'],manifest=manifest_rel,
    rows=sum(len(ds['rows']) for ds in data['datasets']),families=len(data['idea_families']),baselines=len(data['baselines']),
    cases=len(data['case_studies']),logs=len(log_data['entries']),preserved_unrelated_rows=True,generated_at=now)
put('docs/validation/q_cfra_documentation_closeout.json',receipt)
print(json.dumps(receipt))
