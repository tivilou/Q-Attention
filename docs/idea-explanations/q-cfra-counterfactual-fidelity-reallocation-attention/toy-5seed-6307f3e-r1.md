# Q-CFRA：反事实保真度如何生成整张注意力

## 一句话理解

Q-CFRA平均Macro-F1 0.061957234，Disabled 0.056808170，配对平均差 +0.005149064；4/5正向。通过 toy 晋级门槛，可进入正式单 seed 准备。 它检验的是“去掉实体身份后，表示怎样变化”能否用于完整注意力分配；本轮只到真实数据toy。

## 原任务与动机

输入是一句话、词序和有序的主语/宾语实体跨度及类型，输出是40类关系中的一个。词表、类型中和token及其中和向量只用选定训练集拟合。每个seed先训练两层小Transformer，再冻结编码器；所有条件继续训练同一个规格的分类器，干预条件额外训练注意力模块。验证集Macro-F1选择checkpoint，测试集只在阶段结束后评估。Macro-F1是各类F1的等权平均，Accuracy是分类正确比例，交叉熵loss越低越好；关系micro-F1另由原项目指标函数计算。无检索、自由文本生成、硬件量子或finite-shot阶段。

前一Q-ECPA正式三seed没有复制历史seed13收益。本次改为更直接的反事实干预与完整分数替换；这是改变作用路径的假设，不是效果承诺。

## 一步步怎么做

保留原句，并把主语或宾语的每个token分别替换为同角色/类型的中和token，得到长度、位置和跨度对齐的三个视图。冻结的第一层分别编码它们。共享仿射投影给出两个复振幅，经归一化和随机幺正CPTP信道成为合法密度矩阵；固定去极化项为1e-4。比较原句与两个反事实视图的平方Uhlmann保真度，并计算Bloch向量的实对称协方差。每个query-key对把原始打分、两个保真度logit和协方差编码为两量子比特的旋转角；两层共享局部门与CZ后测量Z0+Z1+Z0Z1。这个读出取代第二层整张打分矩阵，masked softmax分配全部注意力，再使用原来的value与分类头。它没有小幅量子残差。 保真度是在同一token的原句/中和视图之间计算，并作为key侧特征共享给所有query、head与视图；它不是query与key之间的密度态保真度。各query/head的差异由原始qk分数提供。

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

模型dim32、2层、4heads、FF64、dropout0；每seed公共warm-start8epochs，随后每条件6epochs，均为Adam lr.001，逻辑batch16。内存修订后物理microbatch2，累计完整逻辑目标的梯度后只step一次；CE和预测JS按样本数加权，attention JS按有效query数加权。目标是原句CE+.05三视图预测JS+.05三视图attention JS；Disabled也接受同样视图损失和分类器更新。公共编码器被冻结，分类器3880参数；每个非Disabled模块另有150个名义参数：投影132、信道6、读出12。经典控制存在冗余仿射方向，只能称名义预算匹配。

信道混合权重3、旋转角3和读出12都进入优化器。固定项包括去极化、保真度clamp、编码缩放、CZ及温度。非Disabled参数总数4030，Disabled3880；名义模块150，不能据此声称有效容量相同。

## 训练与推理

训练只更新分类器及可选模块，公共编码器冻结。验证集选best epoch；测试只报告该best的原句分类。initial/best/final诊断回放另有各split轨迹，不参与选模或调参。实际Adam更新每条件240次，物理拆批不增加step。

## 帮助理解的例子

预先固定每个split哈希排序后的第一条，在initial/best/final回放，并记录首个真实逻辑更新；每条件每seed10条，共350条。实际原句、实体、预测与完整数组保留在`/home/Q-Attention/q-cfra-toy/runs/q_cfra_toy/20261020-v1`的私有`seed*/<mode>/sample_trace.json`及NPY文件中。下表只公开长度、定位与哈希。

| 条件 | 固定sample ID | split/checkpoint/lane | 实际token数 | 私有trace SHA256 |
| --- | --- | --- | ---: | --- |
| disabled | `train-selected0:update0:training` | train/update0/training | 15 | `37d36629cf41237261773339accf2ac31c0cf735ad9a5000087ad31d0cc001fc` |
| disabled | `train-selected0:best:replay` | train/best/replay | 15 | `37d36629cf41237261773339accf2ac31c0cf735ad9a5000087ad31d0cc001fc` |
| disabled | `valid-selected0:best:replay` | valid/best/replay | 48 | `37d36629cf41237261773339accf2ac31c0cf735ad9a5000087ad31d0cc001fc` |
| disabled | `test-selected0:best:replay` | test/best/replay | 34 | `37d36629cf41237261773339accf2ac31c0cf735ad9a5000087ad31d0cc001fc` |
| quantum | `train-selected0:update0:training` | train/update0/training | 15 | `08255e4ee78e920a0df335cfadcacb4c0b150b65866458ad4adc15985a5c6333` |
| quantum | `train-selected0:best:replay` | train/best/replay | 15 | `08255e4ee78e920a0df335cfadcacb4c0b150b65866458ad4adc15985a5c6333` |
| quantum | `valid-selected0:best:replay` | valid/best/replay | 48 | `08255e4ee78e920a0df335cfadcacb4c0b150b65866458ad4adc15985a5c6333` |
| quantum | `test-selected0:best:replay` | test/best/replay | 34 | `08255e4ee78e920a0df335cfadcacb4c0b150b65866458ad4adc15985a5c6333` |

审计确认这些病例在不同条件/seed间的输入角色哈希一致。训练记录的梯度/损失对应完整逻辑batch，逐样本内部张量仅捕获第一条，不能把它称为完整逐样本微批次轨迹。尚未提供静态门户原句/任意坐标查看器，故viewer状态是archived_not_projected；捕获本身完整，公开文档的案例可读性仍有限。个案与50条重复checkpoint观察只支持执行诊断，不替代总体效用或因果归因。

本说明没有用教学样例替换真实Case Study，也没有公开私有原句或预测。读取完整状态需在原私有run使用带哈希校验的坐标reader。

## 实验设置与对照

630/631/587条、40类输出、五固定seed。train有正支持的类别40/40，相对训练词表的零支持类别无；valid有正支持的类别40/40，相对训练词表的零支持类别无；test有正支持的类别39/40，相对训练词表的零支持类别['per:country_of_birth']。Macro-F1按固定40类取均值，零支持类保留原指标定义，不改成39类口径。来源内因资格筛选而不可用的类别另见审计unavailable_labels，不能与跨split缺类混为一谈。 Disabled保留原第二层注意力；cosine/RBF控制用实向量几何；diagonal移除相干；shuffled错配反事实token；entity-permuted中和错位置而保留真实池化跨度。所有模式数据、公共warm-start、逻辑batch、loss与更新数相同。经典仅名义预算匹配，不能承担严格量子归因。

## 实际效果

[已测事实] Q-CFRA平均Macro-F1 0.061957234，Disabled 0.056808170，配对平均差 +0.005149064；4/5正向。通过 toy 晋级门槛，可进入正式单 seed 准备。

| seed | Disabled | Q-CFRA | 配对差 |
| --- | ---: | ---: | ---: |
| 13 | 0.038591811 | 0.044642952 | +0.006051141 |
| 29 | 0.064499286 | 0.073582549 | +0.009083263 |
| 53 | 0.065314186 | 0.075849021 | +0.010534836 |
| 71 | 0.070952296 | 0.061686694 | -0.009265602 |
| 97 | 0.044683271 | 0.054024952 | +0.009341681 |

| 条件 | Macro-F1 均值 ± 样本SD | 相对Disabled均值差 |
| --- | ---: | ---: |
| Disabled | 0.056808170 ± 0.014233858 | +0.000000000 |
| Q-CFRA | 0.061957234 ± 0.013142568 | +0.005149064 |
| Classical cosine | 0.057280148 ± 0.012627689 | +0.000471978 |
| Classical RBF | 0.056717721 ± 0.012284824 | -0.000090449 |
| Diagonal | 0.056892294 ± 0.018993323 | +0.000084124 |
| Shuffled | 0.058943399 ± 0.010528923 | +0.002135229 |
| Entity-permuted | 0.056668174 ± 0.013435464 | -0.000139996 |

预声明晋级要求平均差≥+.005、至少4/5正向、配对差SD≤.004或不大于Disabled SD。本次配对差SD=0.008225856，Disabled SD=0.014233858；shuffled parity=False；梯度/干预塌缩=False。最终classification=`promote_to_formal_preparation`。

仅为630/631/587条真实数据 toy；通过 toy 晋级门槛，可进入正式单 seed 准备。未进行完整数据正式实验，不构成L1/L2、创新性或量子优势结论。

## 成功与失败原因

[已测事实] 数值、完整轨迹与恢复门禁通过；这是工程证据，不是任务收益。首次真实数据canary在1GiB容器被系统终止，原部分目录保留；之后通过有等价性测试的microbatch累积和共享门精确合成完成执行。

[诊断支持] 预声明的首次逻辑更新中，投影/信道/读出梯度和注意力移动按冻结阈值逐seed记录，实际值见审计decision.mechanism_diagnostics。保真度/协方差/对易子诊断来自固定选定案例，不外推为所有测试句子的机制规律。

[尚未证明] 实体中和是否保留关系线索、保真度是否过于饱和、完整重分配是否抹去基础注意力，都是竞争解释。区分它们需要预先设计的类型保留性标注、冻结checkpoint的几何敏感性/读出消融；本轮没有执行这些新实验。不得仅从F1反推失败的唯一原因。

| seed | 投影梯度L2 | 信道梯度L2 | 读出梯度L2 | 最大有效query注意力L1变化 |
| --- | ---: | ---: | ---: | ---: |
| 13 | 0.57378 | 0.000322524 | 0.0804091 | 0.897248 |
| 29 | 2.00165 | 0.00741979 | 0.146671 | 1.82534 |
| 53 | 1.71346 | 0.00135682 | 0.110545 | 1.61197 |
| 71 | 1.4926 | 0.00500265 | 0.146601 | 1.63457 |
| 97 | 1.19689 | 0.00236086 | 0.22573 | 1.79871 |

[诊断支持] 固定50条checkpoint/首更新捕获的token级保真度均值范围为 0.966768–0.99569，上界clamp比例范围 0–0.441176。这些记录包含同一原句的重复回放，不能当作50条独立新测试句，也不能单凭此判定因果机制。

[已测事实] 本轮screen在seed29 shuffled第4轮后随服务器失联中断，日志未记录Python异常，具体原因未确认。重连后逐项核对源码、配置、子集、阶段、warm-start、Adam/RNG及12个完成marker，再按原目录检查点续跑；恢复回执见 /docs/validation/q_cfra_screen_recovery.json。中断与恢复不改变冻结实验或选模规则。

## 一分钟口述

我们的任务是根据句子里的两个实体判断关系。Q-CFRA分别保留原句、去掉主语身份和去掉宾语身份，用同一个冻结编码器得到三份表示，再把它们变成合法密度态。保真度和协方差描述表示对实体身份的敏感性，一个小的两量子比特线路把这些信号变成整张注意力。我们用五个固定种子和六个对照，保留全部关键私有数值并验证了恢复。Q-CFRA平均Macro-F1 0.061957234，Disabled 0.056808170，配对平均差 +0.005149064；4/5正向。通过 toy 晋级门槛，可进入正式单 seed 准备。 因此当前只按冻结toy门槛作决定，不能把实现通过等同于完整数据效用或量子优势。

## 面试追问

- 为什么用CPTP？共享随机幺正混合给出可检查的合法密度信道，而不是随意命名的矩阵。
- 反事实一定保留关系吗？没有保证；类型保留、中和身份是待验证假设，实体错位和shuffle控制用于限制解释。
- 是否更准确地关注证据？未测直接证据对齐指标，不能从F1称作注意力更准确。
- 为什么不是量子优势？经典有效容量、finite-shot、真实硬件与资源审计都未完成。
- 是否多seed即L2？本项目要求完整数据正式评估与独立确认；五seed短预算toy不能自动升级。

## 证据与版本

执行代码6307f3e，source identity `49bf9de390a160920df4aa177c65890157fa34eed72e242c59455cea956ff5d3`；config `f0b604d8af4d9e4325e519bbd8d62e16ac7023fde9443a87629595904eafb004`；subset `ab61365ed69a19323cfca927d2000d5c4d9ee9aa8d6c97b857b7cbeda839cb89`。报告与源码身份分开固定。

- [完整审计摘要](/docs/validation/q_cfra_toy_audit.json)
- [执行源码文件哈希](/docs/validation/q_cfra_executed_source_manifest.json)
- [实际配置](/docs/validation/q_cfra_run_config.json)
- [结果](/apps/experiment-results/?family=family-q-cfra-counterfactual-fidelity-reallocation-attention) · [日志](/apps/research-log/?id=q-cfra-counterfactual-fidelity-reallocation-attention-toy) · [案例状态](/apps/experiment-results/?case=q-attention-case-study-q-cfra-toy#case-study-title)

当前文档缺少公开原样本/坐标页面及逐样本全微批次轨迹，失败机理仍需区分性证据；实验本身已完成并审计，文档限制不改变实验完成状态。
