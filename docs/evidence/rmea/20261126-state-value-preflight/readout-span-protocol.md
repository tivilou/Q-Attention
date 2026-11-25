# 下一项诊断：固定量子词态是否保留独立 signed value 信息

这是 D 组件的一项必要条件检查，不是新任务方法的选择或训练晋级。只检查固定表示；它的失败不证明联合重训所有表示均失败，成功也不证明收益或新颖性。原 gen98 停止门保持，不更改旧试验。

## 固定对象与输入核验

唯一 teacher 为既有全量原生 quantum seed13 的固定 selected checkpoint：`8ec179f4605a901ea1aeeeaefaacc20d78a0b202f236a2d41c1a37b06220cf3c`，state SHA `c8f29a9d6694fc1fc4c21a65ecb2cb4e237b820def31eb6d5fda92e6248ed7a7`，epoch17。这是既定模型身份，不重新选择 checkpoint。定位和缓存哈希来自旧 shared-query-pair-feasibility 的 design.json；当前尚未验证服务器可用性。

使用唯一既有 TRAIN cohort SHA `8fb8393bbe2241929748283f2bbaa219682696409857ec15678585bcd0b88ff0`：8192 FIT/4096 assessment，句子组不交叉。cohort.py 的索引基于缓存 metadata 的 id_hash，而非 canonical JSONL 行号。必须先验证 cohort、metadata SHA `9e6eac9ad582a4ef4e1a8a12190bfef3b8f0a18c5efbd9072f91f31563656c0e` 和 targets-file SHA `6ce409b586a421995f7d70906f062a7e91fe9d35a04be5359a2f58a3485ee604`，不得重新建 cohort 或猜行映射。targets只校验文件身份，不用于拟合或决定门槛。

同样验证 h/roles/offsets 三个缓存哈希和 native代码哈希，详情见 next-direction.json。完整 native teacher 曾用全TRAIN训练并经原来VALID选择；本诊断两组都属于开发暴露资料。两组隔离只适用于当前线性解，不构成独立任务确认。没有新读 official VALID/TEST、token文本或 feature重建。

## 一次线性跨度检查

冻结 teacher 的全部参数。以 float64、eval/no-grad，取原生 ψ、u_ct 及 α_ct，不变动原 EPS。将 ρ=ψψ† 展开为64个实 Hermitian 坐标 x；对照使用其8个对角坐标。它们共享同一个 teacher u 和同一原生 α，不把两套已分别训练的 value 混在归因比较中。

仅对 FIT 按每句总权重1、每词权重1/T_i，累加 G=Σwxx^T、B=Σwxu^T，使用 Moore–Penrose 解 A=G^+B，固定相对特征值截断10^-10，不扫ridge/截断/seed。40类同时保留。Hermitian 系数不施加谱范数约束，这是更宽的可表示性松弛；须报告每类实际算符范数。通过它不证明单位范数或任何较窄物理约束可行。

锁定 A 后，只在同一固定4096 assessment 输入上核 u_hat=xA，以及固定 α 下的 pooled logits。统一报告 FIT/assessment 的40类 MSE，按 FIT teacher value方差+10^-12归一化后的全部40类均值、最大值，以及 teacher argmax 保持率。常数 teacher FIT均值是低端参照，原生64-channel classical value→decoder 是精确 teacher 参照；它不是新经典成果，也不作为量子替代主线。不读取 gold labels 计算新F1。

必要信息门冻结为：assessment 的全40类平均 normalized MSE≤.01、每类≤.04、teacher argmax保持率≥.99。它表达“近似保留既有值函数”，不是效用改善标准；所有类均计入，没有类别/句长/子集筛选。若失败则关闭这个固定表示的 value替换假设，不改阈值、谱范数、rank、cohort或checkpoint补救。若通过，仍须另有任务机制假设、量子value近邻原文与公平Q/D/强经典控制，才考虑独立预声明训练。

## 资源与执行边界

本轮只有数学检查，未执行此任务诊断。下一轮先验证来源、现有文件和共享调度；仅在既有 gpushare-i2-38434-2080tix2 分配至多一张卡，不购容量、不碰他人工作。最多两遍 FIT和两遍assessment特征访问、每遍按64句及必要microbatch执行；无 optimizer/backprop/新seed，GPU wall上限600秒，不重启失败尝试。所有数组、系数和checkpoint留服务器，Windows/public只收脚本、哈希与汇总。缺少固定输入或一致映射则停止这项诊断并记录缺口，不选其他模型凑齐。
