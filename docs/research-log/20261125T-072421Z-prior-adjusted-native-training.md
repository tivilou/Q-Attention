# 原生三路的类别先验训练修正

## 问题与实际干预

上一轮三路折叠读出均出现平均CE下降但Macro-F1降低，且更多正关系被判成null。
这只是损失与任务指标之间的开发观察，不能隔离不均衡、优化、正则或量子原因。
本轮恢复原生factorized value/decoder，只对训练目标做一个固定、已知的干预：
CE(raw_logits+log(fit_class_count),y)，评估时精确返回raw_logits，无阈值调参。
Q、原生去相干、Gaussian三路接受同样干预，各81028 trainables；无新增分类器、
特征、采样改变、折叠参数化或阶段冻结。

计数只来自原固定8192fit、全部40类正数，null5400/8192；最少类仅1条，需保留
这个小数据筛选限制。没有读取assessment标签设置prior，没有用VALID/TEST选方法。
本法是作者官方源码确认的Balanced Softmax，不是本项目的理论创新或完整BALMS。
它也不保证Macro-F1最优；balanced accuracy与Macro-F1不同。

## 全部七路固定结果

| 条件 | 全40类Macro-F1 | 原始输出CE | 正关系→null | null→正关系 |
|---|---:|---:|---:|---:|
| adjusted/quantum | 0.369463600921 | 1.505527496338 | 93 | 1182 |
| adjusted/dephased | 0.385477822812 | 1.795422077179 | 54 | 1479 |
| adjusted/gaussian_base | 0.405525099315 | 1.546728134155 | 79 | 1328 |
| reused/quantum | 0.389300602574 | 0.750126183033 | 450 | 333 |
| reused/dephased | 0.472501085293 | 0.707974433899 | 400 | 362 |
| reused/gaussian_base | 0.484228946800 | 0.705439209938 | 385 | 360 |
| reused/gaussian_covariance | 0.512054050292 | 0.695091843605 | 372 | 336 |

- quantum_adjusted_minus_native: -1.983700个百分点。
- dephased_adjusted_minus_native: -8.702326个百分点。
- gaussian_base_adjusted_minus_native: -7.870385个百分点。
- Qadjusted_minus_Dadjusted: -1.601422个百分点。
- Qadjusted_minus_Gadjusted: -3.606150个百分点。
- Qadjusted_minus_Gcov_practical: -14.259045个百分点。

固定六门0/6通过。六门未全部通过，暂缓本确切full-strength/1024配置，不进入全量/多seed。
必须保留同样受到干预的D/G，不能只用旧经典对照归因于量子。所有40类support/F1/CM
和学习曲线均保留，null错误只作描述，不能替代预声明主Macro。
Gaussian covariance是较强函数类别的实用参照，不是loss路径/容量匹配的消融。
不调prior强度、class weights、阈值、LR、seed、cohort、预算或删控制救援。
小筛选未晋级不等于证明整个先验修正/量子模型家族不可能。

## 预算与执行

同一原TRAIN8192fit/4096assessment、同句组无重叠/全部40类、seed13/1024updates、
b128/micro4/FP32-complex64/TF32off/原AdamW；三新fit全完成后才统一评最终状态。
复用四路精确同cohort/顺序/初始化方法/预算的已完成参照，无重复baseline。
182.445秒/3072实际更新，双2080，gen94可信归还。
稳定screen50ce858f/host238f7e68字节未改，只检查新loss/训练评估切换、梯度、
计数buffer、NumPy公式、原生端点和克隆真实8行更新，Astra6.2/high确切语义审核。
没有单卡/暂停/多卡/COS恢复矩阵；0新VALID/TEST、0COS，Windows0模型/特征/逐例数组。
七路40类CM独立重算最大误差1.11e-16。
训练曲线记录的是新Balanced Softmax CE，不能当作普通CE与旧训练曲线直接比较；
表中的assessment CE全部使用raw_logits，可以直接描述比较。

## 理论与评级边界

原作者README与完整loss源码已读，论文全文/定理未读；未冒称新定理、Macro一致性、
近邻区别或量子专属性。旧Q-PHV/Q-HARP toy修复、MGDA单步与calibration配置仍关闭，
本轮不是重命名或调参救援它们。新L3/L4-T仍NOT_ACCEPTED，目标ACTIVE；
原限定L4及全部历史结果保持。下一步由全部固定结果决定，不选择性补seed。

来源：[Ren等官方实现](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/main/loss/BalancedSoftmaxLoss.py)。

## 已有输出的关系类别排序诊断

不重跑模型，只在GPU主机上读取四份已有private logits（逐文件SHA与同4096indices核对），
固定1365条真实正关系，描述强制在39个正类中argmax的结果。该条件使用真实标签，
不是可部署推理方法或本轮主指标，也不是已知可获得的增益。

| 条件 | 实际正类正确 | 去掉null后正类排序正确 | 排序正确但原被null挡住 | 去掉null仍错正类 |
|---|---:|---:|---:|---:|
| adjusted/quantum | 967 | 1011 | 44 | 354 |
| adjusted/dephased | 1001 | 1019 | 18 | 346 |
| adjusted/gaussian_base | 1027 | 1066 | 39 | 299 |
| reused/quantum | 778 | 1064 | 286 | 301 |
| reused/dephased | 859 | 1129 | 270 | 236 |
| reused/gaussian_base | 867 | 1125 | 258 | 240 |
| reused/gaussian_covariance | 884 | 1147 | 263 | 218 |

原Q即使排除null仍有301条关系类别错误，原D236、Gaussian240、Gcov218。
同条配对中Q胜D50条、D胜Q115条；Q胜G66条、G胜Q127条。
因此当前开发差距不只表现为null判别：关系类别排序也有缺口，尚未隔离因果机制。
新目标虽然把Q的null挡住正确关系从286降到44，但正类排序正确从1064降到1011，
同时新增大量null误报；不能靠单项漏检下降宣称修复成功。

下一步回到输入证据如何区分易混淆关系的具体问题；先定位原Q相对D/G的
关系排序缺口，再提出对这一缺口有可区分预测的机制，公平经典对照同行。
不扫描prior强度/阈值，不延长本已关闭配置，不把条件化指标替代主指标。
这份诊断增加0fit、0forward、0新VALID/TEST、0COS，Windows仅接收计数与39类指标。

[完整结果、源码与固定协议](../evidence/rmea/20261125-prior-adjusted-native-training/summary.json)。
