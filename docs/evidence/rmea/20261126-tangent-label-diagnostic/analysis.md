# 原生量子导数标签诊断：冻结门未通过，停止当前方向

本项目把句子与有序主客体转成固定 RoBERTa 词特征及角色特征。原 Q-RMEA 用3量子位、2层 RY/RZ 和 CNOT 环制备每个词与40个关系查询的状态，以重叠概率加 EPS=1e-6 得到关系内词权重，再对经典64维 value 的关系效用加权得到40分类 logits。本轮沿用已训练的 epoch17/seed13 量子模型，仅重放已冻结 TRAIN 缓存的8192／4096条，不训练、不拟合、不更新查询。原 teacher 已见完整 TRAIN，两组均是暴露的开发诊断数据。

本轮计算每个关系的固定 C 初始全状态导数 g=2||Cphi||²/D，比较真实类别与原模型最强其他类别的导数差。它度量一个理想化查询方向最初怎样改变分类间隔，不是原 PQC 角度 SGD，也没有执行有限步物理条件更新。方差 variance、token-Gram 对角 self 与上界 upper=2Gamma*variance 是同一原生输出上的简单诊断控制，均不是独立训练的公平经典基线。

| 相邻步骤 | 已做内容 | 本轮差别 |
| --- | --- | --- |
| 之前的tangent代数 | 人工原生两例与一般纯态，证明分解/已知上界 | 首次在同一真实缓存的固定12288条上检查标签方向 |
| gen102固定value替换 | 拟合替换效用并按原冻结门STOP | 本轮零拟合，未重启该方法 |

## 固定案例与可见范围

[已测事实] `sample-trace.v2`，hash `003c0f49b7f3f9c096752cd6a8645ea29d870e45b8e6a31c6235de6f460f2cba`；schema hash `b23b1db0e600229b786077f2a5c0748acb87e11dc21dc39d90d85c60365ff082`。选择规则是冻结 fit 前4及 assessment 前4，未按结果选例。完整私有 trace 与8个数值文件留在服务器 `/hy-tmp/Q-Attention/control/tangent-label-diagnostic-20261126-v2/output/`，不进入 Windows/Git。192个数值表示、411179个元素、560个任意坐标读取通过；原生角度的独立 Kronecker 电路及稠密 A/B/C 重放通过。[verification.json](verification.json) / [output-audit.json](output-audit.json) 给出完整性凭据，源码 `trace_io.py read_value` 提供源绑定读取。

缓存 data、原生 scoring、prediction、导数 diagnosis、标签 evaluation 均实际记录；仅在原生输出和无标签导数产生后读 gold。编码器/原训练轨迹未在本轮产生，标 unavailable；优化本轮不适用。原缓存仅2条有原始tokens，其余只有 id_hash/words，故8案例只有1条可追原词串，7条没有原词串/上游分词值。完整性仅针对声明的缓存重放链，不是从原始文本到编码器的端到端认证。旧431组未精确映射，不作该组结论。

| 预选案例 | 词数 | gold / 原预测 / rival | g间隔初始斜率 | self间隔初始斜率 | 原词串 |
| --- | ---: | --- | ---: | ---: | --- |
| fit:0 | 21 | 0 / 0 / 26 | -11.092230558 | -2.423409249 | 有 |
| fit:1 | 29 | 0 / 0 / 28 | 9.413289395 | -17.137850265 | 缺失 |
| fit:2 | 55 | 0 / 0 / 7 | -62.029924304 | -41.336230853 | 缺失 |
| fit:3 | 29 | 28 / 28 / 0 | 94.392402691 | 39.230241421 | 缺失 |
| assessment:0 | 30 | 13 / 13 / 0 | 258.825611391 | 181.155771664 | 缺失 |
| assessment:1 | 48 | 0 / 0 / 3 | -48.573667077 | -44.794791371 | 缺失 |
| assessment:2 | 39 | 0 / 0 / 22 | -33.842975860 | 23.069497089 | 缺失 |
| assessment:3 | 56 | 0 / 0 / 15 | -200.850360637 | -43.658232673 | 缺失 |

8条恰均为原分类正确，不代表原错误样本。g斜率正3／负5，均值0.780268130，总体标准差124.398918239，范围[-200.850360637,258.825611391]。同一个输入、同一模型、同一原生中间状态产生所有诊断量；例如 assessment:2 的 g 为负而 self 为正。该代数差异可观察，但没有执行实际干预，不能由这8条解释247个错误的因果来源或估计性能收益。

## 总体结果与决定

主门预声明：在原错误样本上，g正向比例至少比最佳简单控制高0.02；对原正确样本的反向比例至多比最优简单控制多0.01；两组各至少50条；数值门必须通过。[protocol.md](protocol.md) / [design.json](design.json)。符号容差为每行每量1e-10*max(1,max_class abs量)，rival来自原logits含bias且并列取最低类别。全部40类（包括零支持）均在结果中，未选择类别。

| 诊断量 | 原错误上的正向比例 | 原正确上的反向比例 |
| --- | ---: | ---: |
| g | 139/247 = 56.27530364% | 2121/3849 = 55.10522214% |
| variance | 140/247 = 56.68016194% | 2210/3849 = 57.41751104% |
| self | 141/247 = 57.08502024% | 2220/3849 = 57.67731878% |
| upper | 140/247 = 56.68016194% | 2210/3849 = 57.41751104% |

[已测事实] assessment4096条=247错误+3849正确；g相对最佳 self 的差为0.80971660个百分点的劣势，要求+2个百分点未达成。原正确保护门通过只说明相对这些控制，没有证明绝对安全：g仍有2121/3849条初始间隔反向。四门中3PASS，决定 **STOP_CURRENT_LABEL_DIRECTION**。[summary.json](summary.json)。8,192 fit有500错误；fit也完整报告，但未据其改门。

[已测事实] 在247错误上，g与self的符号配对为：95条均非正，13条仅self正，11条仅g正，128条均正；加入 signed cross 后净少2条正向。[机制解释] token间干涉确实改变方向，但此固定模型/冻结子集没有显示比对角量更好的错误方向。[尚未证明] 这不是所有联合训练量子模型的否定，也不是 g 与标签独立的统计证明。

每类全状态 g/upper 的163840个 assessment 比值均定义，均值0.040297834953，范围[0.001606084941,0.545945532945]；只是几何占界比例，没有量子优势含义。原模型 TRAIN assessment Macro-F1=0.878615999631，仅重放身份检查；本轮没有新方法 Macro-F1。

## 实现与资源审计

先有gen103执行缺陷：错误假定每个缓存metadata有tokens，在首个原生forward前KeyError退出1；当时已解码 checkpoint并复制少量首批特征，标签/导数/模型输出为0。其记录保持失败且分配已释放，见 `failed-v1/`。经只读核验12288条 metadata.words 与 offsets 全匹配后，显式冻结v2，唯一runner改动为用 words 检查长度；模型、cohort、seed和科学门不变，没有旧profile自动重启或基于结果救援。[implementation-review.json](implementation-review.json) / [runner.diff](runner.diff)。

gen104完成一次12288条重放，192次head／384次批电路调用，0 optimizer，诊断进程17.104545712秒，exit0。独立稠密表达误差 slope 9.022e-15、residual 1.241e-14；电路误差psi 5.953e-16、phi 5.558e-16；数值上界违例0。原存储 state hash c8f29a9d…，FP64模型有效state hash前后 `11e0f668605ab2838bd6e4c1222256b788a8d2a02d17242ea39e8d0d8071263a` 相同；checkpoint字节hash仍8ec179f4…。[execution.json](execution.json) / [verification.json](verification.json)。另一表达与汇总核验由同一agent完成，不等于外部独立审核。

共享整机视图后按不可变输入本地性使用2080Ti2的1张GPU；3060节点离线。仅2次本项目分配gen103/104，均精确generation/attempt归还，无扩容、COS、模型API或私有数组下载。[resource-verification.json](resource-verification.json)。未修改任何训练权重，全部旧正负历史与原限定L4保持。

## 证据边界与下一步

idea-maturity1：新L3 / L4-T **NOT_ACCEPTED**；原限定L4卡与历史标签不变。已知代数、人工可达性、内部数值一致性和本轮标签斜率诊断均不代替实质近邻区别、非平凡新T、公平训练对照、关键消融与独立确认。real8/operator精确经典表示继续保留；实际FastWeights尚未被旧matched-alpha构造排除。Knyazev原文仍未读，既有失败渠道关闭，不作算法归因。

推荐恰好一项：对原Q-RMEA与实际FastWeights递归及real8/operator嵌入做源绑定的函数类审计。假设是尚有一个非简单几何/重参数化的限制可导出任务预测；最小工作为复用已读原文和固定原生源码，列完整方程/假设，给出反例或等价以及可检验预测。只有非同义结构差别且预测明确才保留 exploratory_only；若仅已知恒等式、顺序记账或经典精确模拟，则记录没有合格候选，停止实现准入。预算0GPU／0任务forward／0新容量。此只读审计已在持续目标授权内；不启动已停flow、补seed或参数搜索。[next-direction.json](next-direction.json)。
