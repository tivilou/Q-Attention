# 两近邻来源审计：Fast Weights 已核，Knyazev 全文仍缺

[已测事实] 本次是原文与证据边界检查，不是新实验结果。两篇文献身份已核；Fast Weights 的会议原文及同题 arXiv v1 附录已按范围阅读。Knyazev 只有官方元数据，原文审计未完成。候选继续 HOLD，新 L3/L4-T 未接受，原量子目标 ACTIVE。

## 项目实际工作流

项目用冻结文本表示 h、四维角色和有效词 mask，分别形成三量子比特词态与40个类 query；Born 概率加原 EPS=1e-6 后归一，池化64维词 value，再由各类 decoder 与 bias 输出关系分类分数。原协议训练只用 TRAIN，按冻结的验证与评估规则报告 Macro 指标；公平比较包括原 quantum、dephased 和强 Gaussian/经典控制。训练、独立确认、关键消融与独立审核共同决定证据等级。本轮不读取任务 feature/checkpoint，不训练、不评估任务 Macro，也不申请共享 GPU。历史 gen98 十三门10PASS 后 STOP、gen102 固定 teacher value 替换三门0PASS 后 STOP 均保持。

上一检查点把 signed utility 构成的 C=A-z0B 固定，再归一 exp(sC)phi；证明高于起点而不保证随 s 单调，更不保证分类改善。其960行代数、12个精确例、分类反例和独立静态审核均为既有证据。本次只补近邻来源检查，不把旧结果重计成新实验。

| 比较项 | 上一检查点 | 本次新增 |
| --- | --- | --- |
| 算符行为 | 特定构造能区分原 alpha/value 相同的两个 context | 对照实际 fast-memory 文献，收窄排除范围 |
| 最近邻依据 | reviewer 给出两篇题名 | 准确身份、版本、阅读页与获取失败记录 |
| 证据等级 | 无任务准入、无新 T | 仍 HOLD；未提升 |

## 原文核验结果

| 文献/版本 | 取得与实际阅读 | 可支持的范围 |
| --- | --- | --- |
| Ba 等，NIPS2016会议版，9页 | 全文1–9页；3/5/6/7页图像复核 | Eq1–5、原表1–3、控制与限制 |
| 同题 arXiv:1610.06258v1，11页 | 1页身份；3/4页指定公式窗口；10/11页全文，10页图像 | 单独的固定版本实验附录，不声称全篇与会议版相同 |
| Knyazev，2001，SIAM JSC23(2):517–541，DOI10.1137/s1064827500366124 | Crossref/索引身份核验；全文未取得、未读 | 不归因任何算法公式或固定指数更新有无 |
| 官网链接 Supplemental.zip | 23页PDF仅检查第1页文本与图像 | 与 fast weights 附录内容不符，隔离；其真实论文身份未核 |

来源：[会议官方页](https://papers.nips.cc/paper_files/paper/2016/hash/9f44e956e3a2b7b5598c625fcc802c36-Abstract.html)、[固定 arXiv v1](https://arxiv.org/abs/1610.06258v1)、[Knyazev DOI](https://doi.org/10.1137/s1064827500366124)。本地 hash、页码及状态见 source-receipts.json 和 reading-record.json。

[原文事实] Ba 的 Eq1–4 用随时间衰减的隐藏态外积存储序列历史；Eq5 的 query 内循环含持续输入边界、非线性及 layer normalization。默认 eta=.5、lambda=.95；v1附录的关联检索 lambda=.9 单独保留。v1附录还说明 RL 对随机超参数搜索的前5个模型取平均，不能解释成独立 seed 复现。会议版 Eq1 与 v1 的时间下标印法也分别保存。

[机制解释] 自有代数 C=sum_t(u_t-z0)rho_t + EPS sum_t(u_t-z0)I 中，标量 I 因子在归一指数 query 时抵消；EPS 仍影响 z0 和最终原 readout，未改 EPS。这里可解释为类相关 signed 外积算符与谱滤波。它与已读 Ba 模型的具体更新公式不同，但这些已知构件及其组合尚未构成实质近邻清关或新理论。

[边界更正] 先前的双 context 构造只排除了所列固定公式。实际 fast-memory 模型也可保留原 scalar attention 以外的上下文几何，因此该例没有排除实际 Fast Weights 模型；这补充了上一报告“有功能区别、未清关”的限制。不能把假设线性、去 LN 后的多项式内循环当作 Ba 实际模型，也不能由某公式不同推出模型表达力分离。给定词态与 utility 的求和不含顺序，不等于整个使用上下文 h/role 的 NLP 模型顺序不敏感。

## 结果与公平性

[原文结果、未复现] 会议表1是 test error%，列为20/50/100个 recurrent units：FastWeights 1.81/0/0，LSTM60.81/1.85/0，存在平局。表2是 MNIST error%，特征50/100/200：FastWeights7.21/1.30/.85，对 ConvNet1.81/1.00/.9 为两负一胜。表3 facial accuracy%：FastWeights86.34，ConvNet88.23。完整紧凑转录见 reported-tables.json；不同任务和指标不合并为 Q-RMEA 的 Macro。

[限制] 相同隐藏单元数不认证总临时记忆或计算公平；视觉 glimpse/store 策略预先指定。RL 曲线未数字化，不报告额外精确数值或置信区间。这些结果不替代本项目的强经典控制、独立确认及消融。

本轮18个普通元数据/下载请求与1次公开浏览器导航有预算记录。SIAM403后停止该源；Google200为 JS shell、无可用结果，单次普通浏览器导航429后关闭专用会话，未触碰挑战或其他浏览器会话。Bing返回无关词典页，不能据此声称论文不存在。web 工具后端404未提供任何源内容。最后 www.math.ucdenver 传输错误，未取得PDF。获取失败及补充包错配不代表科学反证，也不由解析页数推断已阅读。

## 案例与证据上限

本轮未执行模型训练、推理或量子仿真；sample-trace.v2 任务案例和批量因果轨迹不适用。不是任务诊断的已完成条目，Macro/置信区间/新 split 覆盖均未提供。取得与阅读证据独立保存：两份正式论文PDF按 hash 私有留存，公开包只含小型元数据、阅读记录与自有分析，不含全文、解析文本、浏览器 profile 或搜索日志。

[已测事实] 外积记忆已经是原文明确机制；官网补充包的来源链接与实际内容需分别验证。

[尚未证明] Knyazev 的实际算法关系、原生线路对构造的联合可达、自然任务中 gold 相对 rival 的导数对齐、强经典公平比较、实质新颖性和独立量子贡献。既有“每类分数不降”与固定 teacher 的431例障碍均不放宽。

## 唯一推荐下一步

先冻结并完成一个小型 CPU 原生线路构造核验：用现有 prepare/effect/value/decoder 的显式参数和两组人工 h，检验此前相位构造能否由实际三比特两层线路同时生成。只测合成可达性，不读取任务或 checkpoint；独立基向量/CNOT 实现做对照。成功要求1e-12精度的态、mask、全64 value、原 attention 匹配及0/.9640256520的构造 readout；任一失败停止该可达性主张。预算最多8次 native-head/16次 circuit、内部计算10 CPU秒，具体调用数执行前冻结。不得改 strength/norm/EPS/rank/seed 救援。

该检查由持续科研目标覆盖，独立于文献缺口；即便通过，也不准入任务方法或训练。Knyazev 原文仍是后续近邻清关条件。自然任务导数诊断排在其后，当前未准入；重新训练或 GPU 筛选不推荐。详见 next-direction.json。

原限定 L4、全部正负历史、gen98与gen102 STOP保持，原量子研究主线 ACTIVE。证据索引：source-receipts.json、reading-record.json、comparison.json、reported-tables.json、verification.json、next-direction.json。
