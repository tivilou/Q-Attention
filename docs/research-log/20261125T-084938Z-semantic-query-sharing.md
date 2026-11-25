# 关系语义共享查询：固定四路小规模筛选

## 问题与实际干预

上一轮原生模型诊断显示，Q在8192条训练窗口比D拟合更好，但在固定4096条评估上
更差，稀有关系贡献了较大Macro差距，部分关系在训练集也弱。这没有隔离统计方差、
query、相干或读出的原因。本轮检验一个已知共享结构是否有帮助的假设，不把它包装
成新理论：公开关系名的固定低秩编码，将40×12自由查询改成40×8固定码乘8×12系数。
null独占第1列，正类使用其余7维；原草稿的“第8坐标”措辞已澄清，编码字节未改。

Q/D再使用原pi*tanh与3qubit circuit；Gaussian仍使用原12维各向异性距离。
key/value/decoder/bias与原始初始化及训练方法一致；W由各原seed13自由query的
最小二乘投影初始化，不声称初始路由输出与自由query模型相同。原query仅为buffer。
三个结构臂同样获得公开关系名信息，均80644训练参数，原81028；registered count
相同不证明有效容量相同。Qpermuted按固定public-name hash循环移位全部39正类行，
null不动，code.T@code谱与秩保持，语义对应被打乱；不声称同标签两两距离保持。
精确code b7622c7b固定，无任务文本、训练targets、支持数或表现参与编码构造。

## 全部八路结果

| 条件 | 全40类Macro-F1 | CE | 正关系→null | null→正关系 |
|---|---:|---:|---:|---:|
| new/quantum_shared | 0.468022177009 | 0.750726461411 | 433 | 316 |
| new/dephased_shared | 0.492804148663 | 0.735639274120 | 426 | 350 |
| new/gaussian_shared | 0.500862151478 | 0.699699640274 | 372 | 351 |
| new/quantum_permuted | 0.430947686422 | 0.760348081589 | 480 | 303 |
| reused/quantum | 0.389300602574 | 0.750126183033 | 450 | 333 |
| reused/dephased | 0.472501085293 | 0.707974433899 | 400 | 362 |
| reused/gaussian_base | 0.484228946800 | 0.705439209938 | 385 | 360 |
| reused/gaussian_covariance | 0.512054050292 | 0.695091843605 | 372 | 336 |

- Qshared_minus_Qnative: +7.872157个百分点。
- Qshared_minus_Qpermuted: +3.707449个百分点。
- Qshared_minus_Dshared: -2.478197个百分点。
- Qshared_minus_Gshared: -3.283997个百分点。
- Qshared_minus_Gcov_practical: -4.403187个百分点。
- Dshared_minus_Dnative: +2.030306个百分点。
- Gshared_minus_Gnative: +1.663320个百分点。

固定七门2/7通过。七门未全部通过，暂缓这个确切rank7/name-token/初始化/1024配置，不进入全量/多seed。
结果分析必须同时保留三个新对照和全部四个旧参照；不改成正类条件排序、低支持组
或平均CE作为成功判据。Gaussian covariance是更强函数类的实用参照，不是同参数
共享消融。仅胜原nativeQ不能证明量子独立贡献；小筛选不确认L3或T。
不扫rank、分词、阈值、LR、seed、训练预算、cohort或删除对照救援；本确切配置
未晋级也不证明所有语义共享/量子方案不可能。所有40类CM/F1/support均保留。

## 执行与边界

同一原TRAIN8192fit/4096assessment、sentence-group无重叠、全部40类、seed13，
1024updates/b128/micro4/FP32-complex64/TF32off，原AdamW参数不变；四路fit
全部完成后统一最终评估。四旧参照cohort/order/初始化方法/预算相同，无重复baseline。
239.232秒/4096实际更新，双2080，gen96已可信归还。
稳定screen50ce858f/host238f7e68字节未改；只做新code/buffer、最小二乘、查询梯度
链式/有限差分、原生公式替换/Gaussian NumPy、padding和克隆actual8更新检查。
Astra6.2/high首轮BLOCK保留：修正继承的总更新数字和训练顺序union检查；完整runtime
design原本含四reused rows，审阅精简packet遗漏已澄清，第二轮确切语义PASS。
该静态审核不是实际fixtures或独立理论审核。实际fixtures结果另存fixtures.json。
八路40类CM独立重算最大误差1.11e-16。
无串行/暂停/多卡/COS恢复矩阵，无新VALID/TEST或COS写入，Windows无模型/特征/
逐例数组。短筛选只记录Git代码/配置/聚合结果，不自动上传checkpoint。

已知语义参数共享与低秩编码不是本项目新T，没有证明量子优势、泛化界、方差因果、
硬件加速或实质近邻区别。新L3/L4-T仍NOT_ACCEPTED，目标ACTIVE，原限定L4及
所有历史结果保留。下一步依据全部结果选择，不选择性追加seed或升级算力。


## 全类支持数分组的补充观察

按上一轮已用的fit支持数/classID排序每13正类一组，事后描述低/中/高组：
Qshared Macro=.295022/.497896/.580555，Dshared=.341689/.545800/.562687；
Qnative=.203452/.400804/.527465。Q三个组均改善，但对D仍是低/中组弱、高组强。
最低组assessment只有59条，不能挑高支持组、构造新cohort或主张确定的规模规律。
这些描述不改变七门失败的结论，也不构成新的独立确认。

下一步已冻结共享query与原固定covariance读出的五臂交互筛选设计：两个模块曾各自
带来Q局部改善，但都未胜公平经典；组合互补仍是待检验假设，并非已经查明原因。
保留语义置换、mean-product消融以及同样组合的D/G。不是重开旧exact profile或
调rank/seed/预算；没有理论创新/量子独立效用保证，需先经新增组合检查和小筛选。

[完整结果、源码与固定协议](../evidence/rmea/20261125-semantic-query-sharing/summary.json)。
