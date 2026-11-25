# 固定下一小筛选：共享关系查询与中心化证据统计的交互

## 问题与假设

gen96的语义共享把Q主Macro从.389301提高到.468022，也高于打乱对应的.430948，
但仍低于同样共享的D.492804/G.500862。gen91原生covariance读出Q.428488，
高于其mean-product .402680，但仍低于Dcov .473626/Gcov .512054。
两个确切配置均未晋级，保持关闭；本轮不调它们的rank、分词、seed或训练预算。

待检验的新联合假设：固定共享查询与固定中心化value统计可能互补，联合改动可能
在公平D/G对照下保留独立实用收益。没有证据已证明交互、互补原因或量子优势。
已知低秩共享和二阶pooling的组合不是新理论；本轮只筛选效用，T仍需独立建立。

## 固定机制与五臂

沿用gen96 code b7622c7b与原LS初始化、key/value/decoder/bias、Q/D原3qubit
circuit和Gaussian原各向异性12维距离。null是首列独立坐标，正类使用后7列。
在同一个relation attention alpha下，沿用gen91的64维value相邻通道配对：

    mu = sum_t alpha_t value_t
    covariance_j = sum_t alpha_t value_t[2j] value_t[2j+1]
                   - mu[2j] mu[2j+1]
    mean_product_j = mu[2j] mu[2j+1]
    z_r = original_base_r + dot(moment_decoder_r, statistic_r)

moment_decoder为40×32、全0初始化。其余参数使用gen96原初始化；初始输出
必须等于对应gen96共享query端点，而不是gen91自由query端点。所有参数联合训练。
没有并行经典分类器、新文本输入、采样、loss、冻结阶段或后处理。

五臂：quantum_shared_covariance、dephased_shared_covariance、
gaussian_shared_covariance、quantum_shared_mean_product、quantum_permuted_covariance。
三种router同样得到公开label-name结构及同样的covariance读出；Qpermuted沿用
原39行固定置换，mean-product是固定使用正号均值乘积的同参数量替代对照。
每臂预计80644+1280=81924 registered trainables，不声称effective capacity相等。
pair/通道/scale固定，不扫描或从结果中选择。

## 固定投入和晋级

同原TRAIN8192fit/4096assessment、全部40类、sentence-group不重叠、seed13，
每臂1024updates、b128/micro4、FP32-complex64、TF32off、原AdamW：
lr.001/wd.01/betas(.9,.999)/eps1e-8/clip1。5fit全部结束后统一最终评估。
总5120物理updates，双GPU动态readyqueue，900sec/arm/3600sec总上限。

复用完全同cohort/order/初始化方法/预算的四个固定参照：gen96 Qshared、
gen91 Qcov、gen91 Gcov、gen96 Gshared。保留所有其他历史结果，既不重跑baseline，
也不改变对照的身份或数值。主指标仍全40类Macro-F1，无任何正类/稀有类替代。

QsharedCov严格胜4个新对照及上述4个已有参照，八门全部通过，才允许设计
完整单seed协议，不自动全量或多seed。失败暂缓这个确切联合配置；不调rank、
分词、pair、通道、scale、threshold、LR、seed、cohort、预算或删除控制救援。
小筛选不能证明整个联合方法族不可能，成功也不是独立L3或L4-T。

## 执行边界

复用stable screen50ce858f与host238f7e68。直接复用已接受的query模块和固定
二阶公式，只查实际新增组合：0系数共享端点、alpha和值的对齐、中心化/mean-product
独立公式与梯度、非0 moment下query梯度、原模型未改和克隆actual8更新；一轮
确切科学语义审查，无核心/续训/COS生命周期矩阵。fresh共享调度后才运行。
短可重启screen0新COS，Windows0权重/特征/逐例数组；Git保存代码/配置/聚合报告。
无新VALID/TEST或独立确认数据，原限定L4/全部历史保留，L3/L4-T NOT_ACCEPTED。

当前只冻结设计，尚未实现、审核、分配或运行。
