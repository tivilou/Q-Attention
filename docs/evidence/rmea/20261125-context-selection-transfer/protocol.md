# 冻结的上下文选择交换诊断

先于新结果冻结。全部58465条TRAIN，全部40类，既有VALID-selected fock1、Gaussian3、nativeReal12三个完整checkpoint。同一已锁定词特征/role/mask。没有训练、调参、官方VALID/TEST、新checkpoint、COS或Windows逐例数组。

每个关系的context为valid且role2=role3=0。接收者H保留每个实体词的原attention，保留整行context总质量cH；context内改为cH*alphaDonor/cDonor，使用H自己的完整value/decoder/bias重算40个logit。空context恒等；单个context词亦恒等。正EPS使非空context的cDonor严格正。四个有向交换一次全部报告，不选接收者/类别/样本救援。

问题：Fock是否包含两个强经典接收者都能利用的“如何在context位置之间分配质量”的信息？这不是语言语义、单光子零点、相干因果或新方法效用假设。

固定四门：F->G3-G3、F->Real-Real、F-F<-G3、F-F<-Real的全40类macro F1差都严格为正，才继续以context选择为问题方向。训练过的固定状态交换不是独立确认或共享可学习性。CE、全类CM、修复/损害、entity/context质量误差、empty/singleton计数同时报告，不替换主门。FP32原路径与把冻结FP32中间记录提升至FP64的读出/交换复算须给出相同四个正向判断，任何反号/零/歧义关闭此确切诊断方向，不能通过改种子、子集、阈值或指标救援。

原公式与实际model.forward首3真实TRAIN样本核对；独立NumPy标量参考核对首3记录的四次交换和全部logit。随机夹具覆盖padding/empty/singleton及自捐赠，另对全TRAIN自捐赠/实体逐词质量/context总量检查。只验证新算式；宿主runner/调度/续训/COS能力不重测。完整source、cache、selected checkpoint及state身份固定，运行后state不变。两个既有GPU在一次分配内平分TRAIN只读计算，同一科学人群；批128，timeout600秒。输出只允许summary.json、fixtures.json、RUN_COMPLETE，逐例logits只在进程内。

旧DPP建议已因与Nov14已知成对分支重合而不采用；本诊断不实现DPP，也不重新开放刚关闭的光学全量配置。无论结果怎样都不自动准入L3或L4-T。
