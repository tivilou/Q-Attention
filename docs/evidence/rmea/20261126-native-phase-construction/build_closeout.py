from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((r/n).read_bytes())
v=read('verification.json');result=read('results.json')
assert v['status']=='PASS_NATIVE_CONSTRUCTION_RECORDS'
parent=Path('D:/CodexHome/task-artifacts/qattn-residual-fastweights-source-20261126')
metadata=parent/'knyazev-openalex.json';m=json.loads(metadata.read_bytes())
assert m['doi'].lower().endswith('10.1137/s1064827500366124')
author=m['authorships'][0]['author']
assert author['id']=='https://openalex.org/A5064449686' and author['orcid']=='https://orcid.org/0000-0002-1635-3711'
source=json.loads((parent/'source-receipts.json').read_bytes())
failed=[]
for a in source['knyazev']['attempts']:
    for x in a['attempts']:
        for key in ['source','url','final_url']:
            if x.get(key):failed.append(x[key])
next_=dict(decision='HOLD_CANDIDATE; EXACT_ORCID_AUTHOR_CHANNEL_ONLY',
 action='Resolve the author-profile route from the ORCID identifier already present in the exact-DOI OpenAlex record, then inspect at most its directly linked public official author/repository copy of this exact Knyazev paper.',
 reason='Native synthetic joint construction is now verified; unresolved Knyazev original remains a scientific source gap before task-method admission.',
 metadata_pin=dict(path=str(metadata),sha256=sha(metadata.read_bytes()),
                   DOI='10.1137/s1064827500366124',OpenAlex_author=author['id'],
                   reported_ORCID=author['orcid'],
                   identity_limit='ORCID association is index-reported, not independently verified as self-authored or exact-paper author until the public record/name and linked source are matched.'),
 first_request=dict(method='GET',url='https://pub.orcid.org/v3.0/0000-0002-1635-3711/person',
                    Accept='application/json',fields_to_inspect=['name','researcher-urls']),
 collection=dict(max_ordinary_requests=4,max_exact_PDF_transfers=1,max_browser_navigations=0,
                 max_model_API_inference=0,minimum_inter_request_seconds=1.5,max_retries=0,
                 followups='Only explicit public researcher-URL links from matched ORCID identity or exact-title/DOI links on that official profile. No path guessing.',
                 stop='Stop origin on403/429/challenge; no Google or closed SIAM retry; no repeat prior failed URLs or scheme-only variants. If name/link identity fails or budget ends, preserve UNRESOLVED.',
                 already_failed_targets=sorted(set(failed)),
                 sensitive_fields='No emails, addresses, browser profile or raw personal metadata in public report.'),
 success_gate='Acquire exact original or explicitly identified author manuscript, verify title/author/version/hash, read actual generalized Rayleigh residual/preconditioner/trial-space update and fixed-exponential relation with page evidence. Metadata alone never clears this gate.',
 kill_gate='No verified fresh source within budget: report exact gap, HOLD, and choose an independent evidence-grounded research action without inventing claims or repeating failed retrievals.',
 after_reading='Reassess substantive distinction and task-relevant prediction; known residual/spectral/outer-product identities remain known tools. Original-native fixture pass does not admit task fitting.',
 resources=dict(task_rows=0,feature_reads=0,checkpoint_reads=0,native_head_calls=0,circuit_calls=0,
                optimizer_updates=0,GPU_allocations=0,server_writes=0,COS=0,model_API=0),
 authorization='Existing sustained research goal; bounded public read-only metadata/fulltext research.',
 goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'next-direction.json').write_bytes((json.dumps(next_,ensure_ascii=False,indent=2)+'\n').encode())
report='''# 原生相位构造通过：人工输入可达，任务贡献仍待建立

## 项目实际工作流

原研究从冻结文本表示 h、四维角色和有效词 mask 出发，以三量子比特两层 RY/RZ+CNOT 生成词态；40个类 query 分别作二元测量，不是联合 POVM。原 Born 概率加 EPS=1e-6 后归一为 attention，池化64维词 value，由40类 decoder 与 bias 形成关系分类 logits。完整任务训练使用 TRAIN 和冻结的损失/选择规则，主要报告 Macro 指标；需要 dephased、强 Gaussian/经典对照、关键消融与适当独立确认。本轮不存在训练、自然文本推理、gold标签、任务 Macro 或 held-out 评估阶段，也未读取这些数据或 checkpoint。

## 本轮结论与变化

[已测事实] 在未改动的原生 Q-RMEA head 上，预先固定的两个合成 context 同时生成成功：原 attention、完整64维 value 和词态 populations 相同，原全部40类 logits 均为0；固定 signed-value 更新后，row0 分别为0和0.9640256520245128。稠密独立参考最大误差4.440892098500626e-16。这里只补上构造的原生实现可达性，候选继续 HOLD；新 L3/L4-T 未接受，原量子目标 ACTIVE。

当前候选保留原词 utility，以 B_t=rho_t+EPS I、B=sum B_t、A_c=sum u_ct B_t，固定原分数 z0 与 C=A_c-z0B，再归一 exp(sC)phi。此前已证每个正 s 高于起点的限定性质，也已保存非单调和分类变差反例；不能把它当作任务收益或新理论。上轮完成 FastWeights 的指定原文检查，但 Knyazev 全文仍未取得。此轮只检验同一解析构造能否兼容实际 prepare、effect、value 和 decoder。

| 上一检查点 | 本次增量 | 保持的限制 |
| --- | --- | --- |
| 一般8维纯态构造；联合原生可达未证 | 实际3比特2层线路和统一参数生成同一构造，支持位由预先声明的基重标变为0/2 | 人工 h/role 不代表文本 encoder 可产生它 |
| 原 alpha/value 相同的两 context 在固定更新后输出不同 | 实际 head 的 alpha、全64 value、populations 完全匹配，独立矩阵与解析参考一致 | 实际 FastWeights 和公平 real8 控制仍未被排除 |
| 无任务/理论准入 | 仅实现构造门通过 | 无任务方法选择、无新 T/L3/L4-T |

## 冻结设置与真实案例

两个 context 各2词，h形状[2,2,1024]，roles[2,2,4]全部0，mask[2,2]全部True；这是 head 接口允许的合成输入，不主张真实关系数据的角色语义。其余 h 坐标全0，h0在 A 为(+atanh(.5),+atanh(.5))，在 B 为(+atanh(.5),-atanh(.5))；两者 h1=(-1,+1)、h2=(2,-3)。全部参数置0后，仅 prepare.weight[0,0]、value.weight[0,1]、value.weight[1,2]、decoder[0,0]置1。所有40个 query角度为0，仅row0有非恒定 signed utility，另外39行恒为0。构造器seed0仅用于初始化记账，所有参数随后被覆盖，没有训练seed或择优。

固定案例是冻结 scope 中的 context A=plus/plus、B=plus/minus；不是从任务结果挑选。两次批量 forward 分别为 quantum、同一参数的 dephased；每次内部调用词态与 query circuit 各一次，共2次 head/4次批量 circuit API调用，不是4次物理量子实验。另有44个独立稠密参考态、80行独立谱计算和80行 PyTorch matrix_exp。s=1仅重复旧合成例，未选择任务 strength。数值门1e-12，内部 CPU 上限10秒；任何科学门失败即停止该构造主张，不作参数/EPS/norm/rank/seed救援。

### 案例 A：相同 plus 词态

[已测事实] 实际 psi=(e0+e2)/sqrt2 两词相同，phi=e0；原每类两词概率均为.5、attention均为.5。完整 value 两词分别以(-1,2)与(1,-3)为首两维、其余62维0。row0原分数0，C=0，固定更新后仍为0。

### 案例 B：第二词改为 minus

[已测事实] 实际第二词态为(e0-e2)/sqrt2，其 populations、原 probability/attention和全部value与A相同。row0 的 C 仅在(0,2)、(2,0)位置为-1，归一 query 为[cosh(1)e0-sinh(1)e2]/sqrt(cosh(2))，readout=tanh(2)/(1+2EPS)=0.9640256520245128。另外39类仍为0。原生Q、D全部原 logits 都保持0。

[解析对照] 首层 wire0 的 RY=±π/2 产生(e0±e4)/sqrt2；两轮CNOT把100依次映为011再010，最终支持0/2。prepare 的原 π*tanh 映射用 h0=±atanh(.5) 精确实现该角度。完整门级推导及 scalar readout 见 analytic-construction.md。

## 汇总与审计

| 项目 | 记录 |
| --- | --- |
| 最高绝对数值误差 | 4.440892098500626e-16，低于冻结1e-12 |
| 两 context 的 alpha / 全64value / populations 差异 | 三项均0 |
| quantum / dephased 原40类 logits | 两者全部0 |
| 更新后 row0，A / B | 0 / 0.9640256520245128 |
| head / circuit 批量调用 | 2 / 4，无重跑 |
| 内部 CPU / wall 秒 | 0.03125 / 0.06507379998220131 |
| 新任务、feature、checkpoint、optimizer、GPU、模型API、原文网络请求 | 全部0 |

CPU计时保守包含 native-stage.json 的中间打包/写入，不含最终trace/results序列化，未放宽10秒门；Windows process_time 的实际读数直接保留。QT Python3.11.10、PyTorch2.8.0+cpu、NumPy1.26.4，零安装。原生两个源文件与全部参数前后hash不变。独立参考用 Kronecker 门、逐列二进制CNOT矩阵和 NumPy eigh，未调用原生 circuit；独立指另一套计算实现，由同一agent执行，不是外部审核或独立任务确认。实际执行回执保留完整命令、stdout/stderr与exit0。

## Trace与证据上限

native-stage.json（synthetic-native-stage.v1）保存完整数值 h/roles/mask、参数完整的零填充+4项赋值配方与digest、实际Q/D各输出和调用记录；trace.json（synthetic-native-phase-trace.v1）保存 rho/B/A/z0/C/q/概率/attention/readout、独立参考和每门误差。输入、参数生成、原测量与固定更新各阶段均覆盖；无训练、优化、标签评估阶段。此为预声明的合成计算trace，不声称任务 sample-trace.v2 或训练后checkpoint案例已完成。验证器从记录另核闭式 C、query、readout 与源/trace hashes，没有再调用原生 head。

[正向经验] 泛化纯态例中担心的 prepare/value耦合，在这一固定实例上可由同一显式参数实现；相位几何确实可以改变固定更新后的readout。

[负向边界] 本例仍全为实数态，real8 可以精确表示；40类只有row0非恒定，其余39行常量。它未认证一般非零RZ、全部门/梯度或任意40类query联合可达，更未证明该几何与自然标签相关。已核实际 FastWeights 模型仍可保留此类几何，原文新颖性清关尚未完成。旧431个固定teacher数值障碍和所有分类反例照旧。

当前证据只是构造和实现一致性的一个限定检查点，非独立任务确认、量子必要性、硬件收益或新理论。本轮没有任务主指标，不能把缺少的 Macro/统计置信区间写为0。原限定L4与全部历史保持，新L3/L4-T均NOT_ACCEPTED。

## 唯一推荐下一步

本地已保留的精确DOI OpenAlex记录给出作者标识 A5064449686、ORCID0000-0002-1635-3711。下一步冻结最多4个公开请求：先读该 ORCID person 的姓名和 researcher-urls，仅跟随身份匹配的明确公开作者/仓储链接，最多取得1份这篇确切原文。不猜路径、不重试旧失败URL/协议变体，不访问已停止的Google或SIAM渠道；403/429/挑战即停止该源。ORCID只是索引报告的关联，仍须核姓名与实际论文身份。原文取得后核 title/author/version/hash，并实际阅读 residual/preconditioner/trial-space 和固定指数关系；仅metadata不算通过。

这一公开只读获取由持续科研目标覆盖，无额外确认。若预算内无新可靠来源，保留确切UNRESOLVED/HOLD并选择其他独立的证据驱动研究动作。合成构造通过不会自动准入任务诊断/训练；真实 gold-vs-rival 导数对齐、公平 scalar/diagonal/real-vector/强Gaussian控制、完整预声明及独立确认仍是后续必要工作。

证据：scope.json、receipt.before-execution.json、source-pins.json、construct.py、independent_reference.py、native-stage.json、trace.json、results.json、construct-execution.json、verification.json、analytic-construction.md、next-direction.json。原量子目标ACTIVE；gen98、gen102 STOP及全部正负历史保持。
'''
(r/'analysis.md').write_bytes(report.encode())
out=dict(status='BUILT_NATIVE_CONSTRUCTION_CLOSEOUT',analysis_sha256=sha((r/'analysis.md').read_bytes()),
         next_direction_sha256=sha((r/'next-direction.json').read_bytes()),
         metadata_identity_source_sha256=sha(metadata.read_bytes()))
(r/'closeout-build.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(out))
