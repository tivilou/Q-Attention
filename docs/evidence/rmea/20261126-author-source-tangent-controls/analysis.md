# 作者渠道仍缺原文；初始导数的方差与几何控制项已核

## 项目实际工作流

Q-RMEA用冻结文本表示h、四维role和有效词mask生成三比特两层词态与40个class query；分别作二元Born测量，加原EPS=1e-6归一，池化64维value，再用decoder/bias分类。原任务训练和Macro指标需要固定数据边界、公平dephased/强Gaussian等经典比较、关键消融、完整预声明及独立确认。本轮没有训练、自然文本推理、gold任务评估或新checkpoint；任务Macro/置信区间未提供。

当前未选定候选固定C=A-z0B，并归一exp(sC)phi；原utility和EPS保持。已知每类分数高于起点的性质不保证margin、Macro或随s单调，旧反例与固定teacher的431例数值障碍继续保留。上一轮只证了一个人工head输入/参数的原生联合构造，real8仍精确复现；没有因此升级L3/L4-T。

## 本轮结论

[已测事实] ORCID公开姓名匹配Andrew Knyazev，作者链接的arXiv索引可读，但已读页面未检出确切题名前缀/DOI。3次请求、0份PDF；Knyazev原文仍UNRESOLVED，算法未读。另完成一个预先冻结的代数检查，给后续标签对齐诊断补出更明确的scalar控制：0<=g<=2Gamma*Var_alpha(u)。这是标准线性代数工具，不计新T。候选HOLD、原量子目标ACTIVE，新L3/L4-T均NOT_ACCEPTED。

| 比较项 | 上一检查点 | 本轮新增 |
| --- | --- | --- |
| 作者来源 | 精确DOI索引报告ORCID | 公开ORCID姓名匹配及明确作者链接；仍无原文 |
| 初始导数g | 2||Cphi||²/D，存在分类反例 | tangent/词间Gram分解、scalar方差上界和signed交叉项 |
| 实现 | 原生人工构造可达 | 只重用保存的row0数值，不再调用native head/circuit |
| 证据等级 | 无任务/新理论接受 | 保持，不由代数或来源索引提升 |

## 作者来源检查

冻结预算最多4个普通请求、1次PDF、0browser/模型API。实际如下：

| 请求 | 结果 | 边界 |
| --- | --- | --- |
| pub.orcid.org/v3.0/0000-0002-1635-3711/person | HTTP200，姓名Andrew Knyazev | 与旧精确DOI OpenAlex作者关联匹配；不是论文全文身份核验 |
| ORCID明列Wikipedia作者页 | TLS握手超时，0重试 | 未读取页面内容，未换身份或绕限制 |
| ORCID明列arxiv.org/a/knyazev_a_1 | HTTP200，作者索引 | 该返回HTML内确切题名前缀和DOI命中均0；不声称全arXiv或其他仓储不存在 |

未识别可核的确切PDF链接后，在3/4请求处结束该渠道，余量不是需要消耗的配额。原Google/SIAM停止保持，未猜路径/请求旧失败URL。来源仅用于身份与导航；没有把索引题名、后续相关论文或metadata冒充2001原文。完整人物原始响应、索引HTML/cache留在私有artifact，公开包仅保留所需姓名/链接、hash和收据，不含联系方式。来源：[ORCID公开记录](https://pub.orcid.org/v3.0/0000-0002-1635-3711/person)、[已读arXiv作者索引](https://arxiv.org/a/knyazev_a_1.html)。

## 不依赖原文的导数检查

[推导结论] 对有效单位词态psi、单位phi，a_t=<psi_t|phi>，p_t=|a_t|²，D=sum(p_t+EPS)，delta_t=u_t-z0。令v_t=a_t(I-|phi><phi|)psi_t，则Cphi=sum delta_t v_t。由词间Gram矩阵K_st=<v_s|v_t>，g=(2/D)delta^T K delta=g_self+g_cross；g_self=(2/D)sum delta_t²p_t(1-p_t)，只依赖原p/utility。g_cross可正可负，保留词间几何。

再记sigma²=sum alpha_t delta_t²、Gamma=sum p_t(1-p_t)/(p_t+EPS)，标准Frobenius/Cauchy–Schwarz界给出0<=g<=2Gamma sigma²。上界非零时eta=g/(2Gamma sigma²)∈[0,1]；上界为零时eta未定义而g=0，未加新EPS救援。这里g_self/Gram对角不等于原生density去相干的dephased模型。完整证明见 tangent/derivation.md；Knyazev未读原文没有被该自有推导替代。

### 固定案例与trace

冻结10个数值case：两条既有人工原生context的row0，以及8个一般纯态代数例，涵盖p全0、全1、0/1混合、constant utility、复态、非均匀概率、独立词global phase和同步排列+无效padding。后8例未认证原生ansatz可达；39个常量行未被当作额外样本数。另用Fraction精确moment给两原生构造证书。全部case在计算前固化，不从结果择优。

| 保存的原生构造row0 | sigma² | Gamma | g_self | g_cross | g | eta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A plus/plus | 1 | 1/(1+2EPS) | 1/(1+2EPS) | -1/(1+2EPS) | 0 | 0 |
| B plus/minus | 1 | 1/(1+2EPS) | 1/(1+2EPS) | +1/(1+2EPS) | 2/(1+2EPS) | 1 |

[已测事实] 10个数值例和2个Fraction证书通过；maxidentity8.881784197001252e-16，global-phase/排列mask不变性误差各4.440892098500626e-16。CPU process_time读数0.0秒是计时粒度下未分辨，不代表无耗时；实际进程耗时保留在decompose-execution.json。无native重跑、无strength/EPS/norm/seed sweep。

tangent/scope.json与receipt.before-execution.json绑定病例、代码、源和证明hash；tangent/trace.json（tangent-control-algebra-trace.v1）保存实际a/p/D/alpha/z0/delta/A/B/C/residual/v/K和每例结果。原生数字来源是既有native-stage.json，未新读任务feature或checkpoint。此为合成代数trace，训练/自然任务/标签评估阶段不适用，不宣称sample-trace.v2任务回放完成。两种数值表达与精确moment均由同一agent执行，非外部审核或独立任务确认。

## 经验和证据上限

[正向经验] 预期的几何差异不能只看每类非负g。相同utility方差、p与Gamma下，signed词间交叉项仍可完全抵消或达到界。这说明后续诊断须同时控制方差、g_self与Gamma，避免尺度混淆。

[负向边界] p全为0/1时Gamma=0，即使utility有方差，当前更新初始导数仍为0；单类正g也不代表gold比rival上升更快。构造仍为real8可表示，不能排除实际FastWeights或强实算符模型，也未建立自然label alignment、任务效用、物理条件更新或实质新颖性。

本轮3普通来源请求、代数部分0网络；全轮0任务行/feature/checkpoint/nativehead/circuit/optimizer/GPU/serverwrites/COS/modelAPI。元数据用既有QT2，代数用QT/NumPy1.26.4，零安装。原限定L4、全部正负历史以及gen98/gen102 STOP保持。

## 唯一推荐下一步

重用gen102已有design.json（该artifact没有scope.json）、runner.py和input_probe.py，核同一nativeQ checkpoint8ec179f4/statec8f29a9d/cohort8fb8393b的输入映射与新的服务器/共享调度可用性，然后冻结一次无fit的TRAIN导数对齐诊断。旧cohort8192/4096及cache-metadata顺序需重核，不能当rawJSONL行；teacher已见全TRAIN，不能作为独立确认。本文只发现并pin已有契约，未完成fresh server核验或准入任务执行。

可证伪问题是：gold相对固定native strongest-other rival的g差，是否有超出sigma²/g_self/Gamma等已知scalar量的标签方向信息；同时报告原正确/错误例与全部40类，旧431组只在ID精确映射后使用。g是固定C的full-state方向，不是原PQC角度SGD导数。无fit诊断不会选择任务strength、重启旧配置或替代公平训练的real-vector/operator、dephased和强Gaussian控制。

原文缺口与探索性任务信息检查是不同前提：此诊断可独立排除缺乏label信息的假设，但通过也不能清关近邻、接受新T或自动启动训练。候选资源建议上限1共享GPU/600秒、0updates；确切调用数、主判据、trace和停止门在契约审计后、任何任务计算前另行冻结。持续科研授权覆盖准备，不新增确认流程。详见next-direction.json。

证据索引：source-summary.json、request-ledger.json、index-inspection.json、tangent/{scope.json,derivation.md,trace.json,results.json,decompose-execution.json}、verification.json、next-direction.json。
