# 研究日志

## 当前研究总览

gen105四checkpoint/4096真实TRAIN无更新诊断完成并归还；主S AUC.509391低于长度.541584/纯度.524882/同Q去相位.513610，优势-.032192 vs冻结+.02、主.55未达，4门2PASS/STOP确切预测。541discordants:Gcov独对300/Q独对241，0缺失；全部80strata和4路40CM保留，原预测argmax0差/Macro0差。512head512circuit0updates/11.292636s，real16 1.11e-15/NumPy2.11e-15/状态checkpoint未变，Fraction36829/72300与-931/28920核验。收集换行失败及字节修复保留、未重跑。原量子goalACTIVE/L3L4T未接受，下一步仅排重joint state/observable utility是否有未测任务假设，0GPU/forward/fit，不旧profile救援。

## 时间线

- [2026-11-26｜配对混叠预测真实数据未通过](20261125T-214027Z-context-pair-diagnostic.md)：4路重放一致但主AUC.5094低于长度；冻结预测STOP，gen105已归还。

- [2026-11-26｜输入词态配对诊断与充分性反例](20261125T-211848Z-context-pair-alignment.md)：6源排重、精确1/2差与输出不变；仅暂定诊断，冻结AUC门槛未运行。

- [2026-11-26｜实际FastWeights与原量子函数类审计](20261125T-205938Z-native-fw-class-audit.md)：完整FW也分开旧相位例；real16/Hermitian64核验，未选新方法，headroom口径分开。

- [2026-11-26｜原量子导数真实标签诊断未过门](20261125T-204006Z-tangent-label-diagnostic.md)：固定8192/4096零更新，g56.28%低于self57.09%；数值/trace已核，停止该方向，两分配归还。

- [2026-11-26｜作者原文缺口与导数的方差/几何控制](20261125T-201101Z-author-source-tangent-controls.md)：ORCID姓名匹配但无原文；tangent分解及界10例/2精确证书通过，下一步无fit诊断契约。

- [2026-11-26｜原生相位构造的实现可达性核验](20261125T-195236Z-native-phase-construction.md)：固定两context构造0/.9640，2head4circuit/独立实现误差4.44e-16；仅合成可达、HOLD，下一步精确作者源。

- [2026-11-26｜FastWeights原文核验与仍未完成的近邻审计](20261125T-193735Z-residual-fastweights-source.md)：补充包错配隔离；Knyazev全文缺口保留/HOLD，下一步合成原生线路CPU核验。

- [2026-11-26｜value引导effect的行为区别与分类边界](20261125T-190545Z-value-guided-effect-preflight.md)：960行/12精确例与反例；独立静态审核PASS，无GPU准入，下一步两近邻原文。

- [2026-11-26｜固定表示 value-span 诊断停止](20261125T-184709Z-observable-value-span.md)：三门0PASS；固定逐词替换失败，资源释放，下一步只作未选机制代数去重。

- [2026-11-26｜同态读值退化与信息必要条件](20261125T-181017Z-state-value-preflight.md)：两种指定读出停止，同态observable保留为未选组件；下一步固定表示的信息诊断，非任务训练晋级。

- [2026-11-26｜依存图输入资格与来源边界](20261125T-175200Z-dependency-interface.md)：原词特征对齐有据，依存图资格未建立；不重建parse/不开训练，保留producer未绑定缺口。

- [2026-11-26｜可训练量子嵌入目标与任务边界](20261125T-173055Z-trained-kernel-audit.md)：17页范围读、源完整表格及选择协议审计；KTA已知工具，未选新模型、0任务训练。

- [2026-11-26｜固定机制检索与GBQC原文边界](20261125T-171647Z-gbqc-source-audit.md)：0/1条命中、28页范围读；固定乘积核与流程控制资格审计，未选择新模型、0任务训练。

- [2026-11-26｜Pauli-Lego核与非负路由适用边界](20261125T-165822Z-trace-kernel-audit.md)：22页范围读、负质量及Mercer物理基边界、独立稠密门/有理反例PASS，0新任务训练。

- [2026-11-26｜结构量子核源协议与槽位重排审计](20261125T-164408Z-structured-kernel-audit.md)：22页范围读、关键消融测试集选择分级，原生latent重排为已知参数对称性，0新任务训练。

- [2026-11-26｜纠缠张量核与原生全梯度对应审计](20261125T-162609Z-entangled-tensor-audit.md)：20页范围读、完整Q/D对应、稠密门/差分独立核验，函数不变不计新贡献，0新任务训练。

- [2026-11-26｜CME目标与五臂平方风险审计](20261125T-160846Z-cme-objective-audit.md)：有限label已知损失，Q/G风险未反转，原门槛不变，0新训练；日期为新加坡本地日，文件名用UTC。

- [2026-11-25｜算符值核嵌入与集中界审计](20261125T-154216Z-operator-embedding-audit.md)：核原文/原生POVM/23维混态边界/集中界反例，0新训练，原量子目标继续。

- [2026-11-25｜相关干扰阈值源转移审计](20261125T-152839Z-correlated-distractor-audit.md)：核原文/320端口，直接转移停止，原量子目标继续。

- [2026-11-25｜gen98固定匿名内容审计](20261125T-151008Z-gen98-content-audit.md)：64判断/固定选择与盲化审计，未识别新机制，0新训练，L4-T继续研究。

- [2026-11-25｜原核心机制咨询与现有预测定位](20261125T-145137Z-native-mechanism-discovery.md)：咨询NONE；固定输入分层与配对误差核验，0新训练，L4-T继续研究。

- [2026-11-25｜角色交换输入资格审计](20261125T-143446Z-role-exchange-input-audit.md)：不假设缓存h不变或40类反向封闭，0新训练，L4-T继续研究。

- [2026-11-25｜量子嵌入原文转移边界审计](20261125T-141308Z-embedding-transfer-audit.md)：不转移二元风险/纯态对捷径，0新训练，L4-T继续研究。

- [2026-11-25｜完整输出局部一致性四路小筛选](20261125T-134905Z-output-consistency.md)：七门按冻结规则判定，L4-T继续研究。

- [2026-11-25｜关系查询联合框架测量三路小筛选](20261125T-131424Z-relation-query-frame.md)：六门0PASS，暂缓确切配置，L4-T继续研究。

- [2026-11-25｜删除同token自乘的五路小筛选](20261125T-105847Z-distinct-token-evidence.md)：原8192fit/4096assessment、seed13、五新九旧，本确切去自乘配置暂缓，不晋级全量。

- [2026-11-25｜共享查询与中心化证据的五路小筛选](20261125T-103308Z-semantic-connected-evidence.md)：原8192fit/4096assessment、seed13、五新四旧，本确切联合配置暂缓，不晋级全量。

- [2026-11-25｜关系语义共享查询的四路小筛选](20261125T-084938Z-semantic-query-sharing.md)：8192fit/4096assessment，四新fit及四固定参照，完整八路保留；本确切配置暂缓，不晋级全量。

- [2026-11-25｜关系类别混淆与原fit回放诊断](20261125T-074335Z-relation-confusion-map.md)：8192fit/4096assessment，四旧模型全类分析、0新fit，下一四臂设计固定；固定模型诊断完成；下一共享查询四路小筛选仅设计冻结、未实现/未运行。

- [2026-11-25｜原生三路的类别先验训练目标筛选](20261125T-072421Z-prior-adjusted-native-training.md)：8192fit/4096assessment，三新fit及四固定参照，完整七路保留；本确切配置暂缓，不晋级全量。

- [2026-11-25｜等价单层词评分的三路小数据筛选](20261125T-065902Z-common-utility-router.md)：8192fit/4096assessment，三新fit及四固定参照，完整七路保留；本确切配置暂缓，不晋级全量。

- [2026-11-25｜共同词评分下的三路注意力训练诊断](20261125T-064555Z-common-utility-router.md)：8192fit/4096assessment，三新fit、实用参照保留；关闭确切256/768配置，不晋级全量。

- [2026-11-25｜同词线索共现读出小数据筛选](20261125T-062818Z-connected-value-pooling.md)：8192fit/4096assessment，五新fit及三控制复用；不晋级全量，完整八路结果保留。

- [2026-11-25｜保持测量总量的上下文谱流筛选](20261125T-054302Z-context-spectral-flow.md)：8192fit/4096assessment，4新fit及3控制复用；局部改善但固定门失败，不晋级全量。

- [2026-11-25｜同checkpoint门控开关诊断](20261125T-043218Z-corroboration-gate-readout.md)：58465TRAIN、0训练/新heldout/COS，全部40类；固定门控有作用但未建立独立泛化贡献。

- [2026-11-25｜跨侧印证五路完整结果](20261125T-041309Z-cross-side-corroboration-full.md)：五新九旧全部保留；13门5PASS，确切配置关闭，新L3/L4-T未接受。

- [2026-11-24｜固定训练路径：局部改善但未超过必要对照](20261123T-170147Z-fixed-coherence-path-result.md)：Q路径提高3.83个百分点，但仍低于同路径实数和去相干；先检查后续筛选协议，不救回本轮固定候选。

- [2026-11-24｜新L4目标与公平经典参照](20261123T-163228Z-new-l4-matched-classical-references.md)：保留原限定L4，固定TRAIN四路比较；同时完成上一轮门槛诊断的聚合报告发布。

- [2026-11-23｜相对证据存在但绝对判定拒识](20261123T-145025Z-relative-evidence-and-rejection.md)：完整TRAIN分层相关性、模型盲选32句及自然正负配对；保留失败与反证，不新增训练。

- [2026-11-23｜先定位实体对正关系证据为何被拒识](20261123T-133704Z-entity-pair-evidence-problem.md)：固定完整模型、已有语义对照与完整TRAIN只读导数；不新增训练或原始数据发布。

- [2026-11-23｜正关系拒识瓶颈与词间条件读出](20261123T-114542Z-problem-and-conditional-readout.md)：完整模型32例实际读出及固定四路TRAIN实验；原始数据和权重留在私有计算节点。

- [2026-11-23｜完整模型的漏检问题与关系拒识](20261123T-104306Z-full-model-error-structure.md)：672条共同漏检中四个头均已把正确正关系排第一；新固定语义32与旧训练样本的适用范围已区分。

- [2026-11-23｜共享无关系参照的四路可训练筛选](20261123T-101940Z-shared-null-differential.md)：已定位任务问题，按冻结预算比较量子／去相干／经典差分与原生；以最终报告为准。

- [2026-11-23｜关系竞争与证据到分类的错位](20261123T-084918Z-relation-competition.md)：无标签联合归一化真实32条干预；更偏拒识，未修复支持明确的漏检；已知Slot Attention近邻，无新理论或全量晋级。

- [2026-11-23｜先确定有收益空间的任务问题](20261123T-082733Z-task-problem-priority.md)：共同漏检的保守合法子集721条，量化标签知情理想修复值与低频类别影响；无新训练或量子收益。

- [2026-11-23｜证据选择与关系竞争的任务定位](20261123T-081145Z-semantic-problem-audit.md)：固定32条盲判与词级回放，区分可聚焦修复、目标行上界障碍和标签歧义；没有新量子机制。

- [2026-11-23｜五模型同类型对照与真实关系漏检](20261123T-074445Z-pooled-type-reference.md)：通用类型修复成立，1070条类型合法的共同漏检成下一问题；没有新量子增益。

- [2026-11-23｜类型兼容修复与经典校准对照](20261123T-070113Z-type-context-baseline.md)：两组CPU偏置校准无收益；保留有效类型规则，进一步定位同类型关系证据。

- [2026-11-23｜先找到可修复的性能问题](20261123T-061718Z-task-bottleneck-discovery.md)：实体类型兼容性经典开发修复，完整VALID净多判对318条；尚不归因于量子。

- [2026-11-23｜先验证任务结构与筛选上限](20261123T-053853Z-quantum-structure-route.md)：绑定样本覆盖存在，固定40对的见证上限不足；不把它当作机制失败。

- [2026-11-23｜逐样本自由测量不能解释这轮正确性翻转](20261123T-050026Z-native-selective-attainability.md)：TRAIN零训练数学边界诊断，双向障碍计数均0；保留共享学习问题。

- [2026-11-23｜原版上下文的两个区域都不能简单当作相干噪声](20261123T-035900Z-native-context-dephasing.md)：四路零训练开发诊断，提分假设关闭；原文近邻与下一问题。

- [2026-11-23｜原版五种子的类别收益与损害](20261123T-022831Z-native-class-heterogeneity.md)：全39类历史开发VALID分析；不新增训练或独立确认。

- [2026-11-23｜原版固定一步的完整开发验证集对照](20261123T-021331Z-native-finite-step.md)：FAIL_CLOSE_EXACT_STEP，19.57秒；不构成独立确认。

- [2026-11-23｜原版两组共同下降方向](20261123T-014909Z-native-two-group.md)：PASS_LOCAL_DIRECTION_ONLY，100.89秒；不构成独立确认。

- [2026-11-23｜原版共享测量梯度的跨样本方向检验](20261123T-011806Z-native-effect-transfer.md)：零优化器更新，有界TRAIN诊断，125.37秒；不构成独立任务收益。

- [2026-11-23｜原版固定分类值的路由空间](20261123T-010252Z-native-routing-headroom.md)：零优化器更新，有界TRAIN诊断，39.07秒；不构成独立任务收益。

- [2026-11-23｜原模型错误与受限改进空间](20261123T-004438Z-native-error-headroom.md)：418/478错误跨有无关系边界；最佳共享偏置只净修复12条，选择输入依赖证据检验。

- [2026-11-23｜无关系参考的输入依赖测量滤波](20261123T-004041Z-null-relative-filter.md)：TRAIN内4192条同源权重续训／4000条评估，143.43秒，FAIL_CLOSE_EXACT_SCREEN；仅探索性诊断。

- [2026-11-23｜主体寻址的带符号值记忆](20261123T-000311Z-subject-value-memory.md)：TRAIN内4192条拟合／4000条评估，473.43秒，FAIL_CLOSE_EXACT_SCREEN；仅探索性诊断。

- [2026-11-23｜原模型接受率证据](20261122T-231217Z-native-mass-evidence.md)：8192条TRAIN，28.06秒，FAIL_CLOSE_EXACT_PROFILE；仅探索性诊断。

- [2026-11-23｜词元测量强度与分类值的顺序交互](20261122T-230109Z-native-ordered-mass-value.md)：8192条TRAIN，16.05秒，FAIL_CLOSE_EXACT_PROFILE；仅探索性诊断。

- [2026-11-23｜统一随机Pauli共用测量检查未过门槛](20261122T-224130Z-native-shared-measurement.md)：1.03秒；总体方差规划比值中位数8.04，128条仅1条改善，精确方案关闭。

- [2026-11-23｜128条TRAIN测量尺度资格检查](20261122T-222254Z-native-measurement-scale.md)：0.39秒；多数方差改善但固定预算规划支撑不足，未启动VALID方案。

- [2026-11-23｜双测量分量短测未获收益](20261122T-220744Z-native-mixed-normalization.md)：固定8192TRAIN／全部VALID，两个960参数头各512更新；候选不及原模型及同规模对照，68.94秒完成，资源释放。

- [2026-11-23｜共享池化排序证书未过门](20261122T-214718Z-native-interval-order.md)：19,584条VALID、0个决策对；16.73秒，工程通过，方案关闭。

- [2026-11-23｜同词出现位置对照覆盖不足](20261123T-date-only-occurrence-qualification.md)：原词／ID／角色精确匹配，支持key为0，未运行模型；日志日期精度仅一天。

- [2026-11-23｜直接循环相位读出未过验证收益门槛](20261122T-211540Z-native-triangle-readout.md)：27.33秒特征＋1.69秒拟合；数值／收敛通过，验证Macro下降1.37个百分点，关闭配方。

- [2026-11-23｜原模型决策纠缠证书未过门](20261122T-205127Z-native-decision-witness.md)：256固定VALID案例，9.14秒；数值工程通过，充分证书近零，停止晋级。

- [2026-11-23｜相干双返回三路完整seed13负结果](20261122T-205037Z-relative-return-full-seed13.md)：21936更新、全量评估和COS100完成；相干低于native及两个对照，关闭配方。

- [2026-11-23｜相对完成分支理论](../evidence/rmea/20261123-relative-process-orbit/theory.md)：数学推导与事前条件诊断，非全量结果。

- [2026-11-22｜固定上下文总权重的正向诊断](20261122-date-only-context-mass-held.md)：事前TRAIN96全保留，78正类门槛通过；不证明保护优势或泛化。仅日期命名。

- [2026-11-22｜实体端点散射三卡完整seed13负结果](20261121T-231433Z-entity-scattering-full-seed13.md)：21,936更新/三路全量评估完成；七文件和独立重算通过，无新任务收益，停止晋级。

- [2026-11-22｜实体端点散射实现与三卡检查](20261121T-184300Z-entity-scattering-implementation.md)：固定读出、匹配实数对照；CPU数值与96次真实前后向通过，无优化器更新。

- [2026-11-22｜端点散射理论候选](20261121T-181537Z-entity-scattering-theory.md)：直流退化边界和旁支风险已明确；复回路方向预测通过数学审查，尚未运行。

- [2026-11-22｜算子方差三卡诊断](20261121T-173646Z-operator-dispersion.md)：冻结原模型，三种标量读出全数据比较；Q方向为负且劣于原模型/均匀方差，联合门槛未过。

- [2026-11-21｜有限探针贡献边界](20261121T165242Z-finite-probe-scope.md)：9次条件计数成立；原17组实测保留，独立L5提名收束。

- [2026-11-21｜checkpoint有限value探针](20261121T162523Z-finite-value-ports.md)：实际端口/九维证书/17组干预通过，合成接口范围内的正向结构结果。

- [2026-11-21 15:49 UTC｜消去共同分数偏移](20261121T154942Z-centered-output.md)：44条配对仍成立，继续核查条件有限探针桥梁。

- [2026-11-21 15:30 UTC｜96条三卡输出匹配](20261121T153018Z-output-matching.md)：固定K8的44条有解；下一步排除共同分数偏移。

- [2026-11-21：配对信用方向三卡诊断](20261121T-date-only-credit-pairing.md)（固定一批/一次更新，完整开发验证；三卡分片完成，门槛未过）。

- [2026-11-21：同前向训练梯度三卡回放](20261121T-date-only-backward-credit.md)（短256更新；三卡实际同时训练；不支持联合归因门槛）。

- [2026-11-21：查询/上下文匹配三卡诊断](20261121T-date-only-query-alignment.md)（三卡真实重叠；全量validation门槛未过，partial trace）。

- [2026-11-21：词质量/方向配对诊断，停止质量补偿](20261121T-date-only-mass-direction.md)（仅日期命名；三项短诊断完成，partial trace 边界明确）。

- [2026-11-21：三阶输运全量结果与线性路径线索](20261121T-034306Z-token-transport-full-seed13.md)（原始运行开始UTC已记录；不扩大历史证据等级）。

- [2026-10-27：Q-RMEA完整seed13结果](20261027T-075438Z-q-rmea-seed13.md)（运行启动UTC已记录；L1候选，待独立确认）。

- [2026-10-26：Q-RMEA 实现验收](20261026T-162616Z-rmea-implementation.md)（实际关闭 UTC 收据；技术验收，不是正式效用）。
- [2026-10-26：前置检查](20261026-date-only-rmea-readiness.md)（条目时间精度仅日期；完整机制 trace 缺失）。
- [历史条目 20261026T-022952Z-q-srm-bounded-screen.md](20261026T-022952Z-q-srm-bounded-screen.md)（时间及结论以原文为准）。
- [历史条目 20261020T-073818Z-q-cfra-toy.md](20261020T-073818Z-q-cfra-toy.md)（时间及结论以原文为准）。
