# Q-ECPA：Re-TACRED 正式 seed 13

默认入口（已按下文同步分支并配置环境后）：

```bash
bash scripts/run_retacred_q_ecpa_formal_single_seed.sh --gpu auto
```

该命令依次完成数据及分支预检、一次共享 baseline、七路 selector、完整指标、私有案例张量上传、审计报告导出、报告提交与推送。完整实验由合作者执行。CPU canary 不能替代 GPU 检查，尚未发布的开发分支也不能直接用于正式实验。

## 同步及环境

```bash
git fetch origin --prune
git switch 1.1
git merge --no-edit origin/1.1
git merge --no-edit origin/main
git status --short --branch
python -c 'import torch,numpy; print(torch.__version__, torch.cuda.is_available())'
nvidia-smi
bash scripts/check_retacred_q_ecpa_formal_single_seed.sh --gpu auto
```

需要 Linux、Python 3.10+、NumPy、CUDA 版 PyTorch（推荐已验证的 2.7.1）、可用 Git 身份及 push 权限。环境内可通过 `PYTHON_BIN` 指定 Python，入口不含服务器路径。数据在 `data/relation/retacred/{train,valid,test}.jsonl`；可用 `--data-dir` 指向现有数据。严格校验 58465/19584/13418 条，词表及标签映射只用训练集，不截断句子或丢弃样本。预先设置项目提供的 `PROJECT_EXCHANGE_URL`、`PROJECT_EXCHANGE_TOKEN`；token 不得写进仓库或日志。

首次正式运行前，在当前 GPU 环境完成小规模真实启动检查：

```bash
bash scripts/run_retacred_q_ecpa_formal_single_seed.sh --canary --gpu auto
```

canary 用公开合成样本、每任务最多 6 次更新，通过真实父子进程执行 baseline 和全部七路 selector；验证初始/最佳/最终案例与任意坐标读取，保留 GPU 实测记录。它不会上传、提交或启动完整数据训练。失败时停止，反馈对应 `runs/.../logs/`，不要更改协议规避错误。

## 冻结实验

seed 13；原有 32 维、2 层、4 头、FF64、dropout=0 骨干。先训练 baseline 8 epoch，逻辑 batch 16、Adam lr .001；按 valid Macro-F1 选择并冻结骨干，再训练可学习 selector 12 epoch，逻辑 batch 256、Adam lr .001。test 只用于事后评价；记录 accuracy、loss、Macro-F1、逐类及混淆矩阵、全类 micro P/R/F1、排除 no_relation 的 micro P/R/F1。

七路包含 Disabled、Q-ECPA、同 Hamiltonian 的逐步退相干、匹配经典路径、冻结随机相位、仅交换主客体条件的量子消融，以及两层原始 QK。Q-ECPA 只插入第一层；原始 QK 保留历史两层设置，因此它是历史参考，并非等容量控制。random_phase 无可训练参数，不虚构更新次数；主客体交换只改变条件输入，不改变池化对象、正确关系及样本集合。

固定主客体交换后重新训练，可能被条件器交换权重两半吸收；这一组只作为角色顺序敏感性的有限诊断，不能证明正确实体条件是增益来源。量子与退相干/经典对照的归因也必须依据实际结果单独分析。

机制：实体有序对生成反对称边相位，构造 Hermitian token 图；用三步 unitary 的最终模平方形成注意力先验，中心化 log 后作为有界分数残差。退相干控制使用同一单步 unitary 的概率矩阵逐步传播。普通前向只算当前分支；选中案例才额外计算对照诊断。

## 调度、恢复与显存

`--gpu auto` 选择可见且至少空闲 8 GiB 的卡；baseline 完成后，七任务进入全局队列，每张卡一个进程。只有一张时自动顺序执行。TTY 界面固定 GPU 行、pending/completed/failed 行并原位刷新；重定向时定时写可读快照，详细信息在每任务日志。

先使用完整物理 batch。仅实际 CUDA OOM 才线性减少 micro：256、248、240……16、15……1，最后再启用精确激活重算；不切 token 轴、不降低逻辑 batch、不改变 epoch。各任务单独记录 tier。OOM 从最后持久化完整优化步恢复，重新执行未保存步骤，绝不累加半个失败 batch 的梯度。仅保留 latest、best 两个 checkpoint；latest 原子保存模型、优化器、RNG、数据游标、best 和验证完成游标，每 50 更新或 120 秒及 epoch/安全暂停边界保存。突然断电可能重算最近未落盘步骤。

Ctrl+C 等待安全暂停后，用真实目录恢复；换单/多 GPU 无需修改科学配置：

```bash
bash scripts/run_retacred_q_ecpa_formal_single_seed.sh --gpu auto \
  --resume runs/retacred_q_ecpa_formal_single_seed/<时间戳>_seed13
```

## 案例证据及导出重试

预先固定各 split 前三个样本，在 initial/best/final 记录句子、token、两个实体及位置、正确/预测关系、实际模型哈希和优化步。逐层保存 embedding、QKV、相位/图/H/U/每步复振幅、退相干/经典概率路径、score、attention、context、残差及 FFN hidden、分类器状态、logits、概率、loss、完整可训练梯度向量。原始 QK 额外保存每头电路角度和实际复状态。第一条训练样本还记录真实首个优化步的前向与整逻辑 batch 梯度；replay 梯度明确是该样本的梯度。

所有数值保存 lossless NPY，包含哈希、轴含义和任意坐标读取入口；同时执行通用 sample-trace.v2 strict complete 验证和嵌入相加、attention×V 等数值链路检查。固定控制的训练状态明确不适用。没有实测证据不得标 complete。私有原值阅读器见 `research-web/apps/ecpa-case-reader/README.md`；需先按回执验证并解包 exchange 原始文件，只监听本机，不能把归档或原值放入公共静态目录。合成 canary 的实际 HTTPS 回读与页面切片已通过 `docs/validation/q_ecpa_evidence_preflight.md` 所述验收，正式案例仍需在返回后单独审计。

训练期间不上传。所有任务完成并写 RUN_COMPLETE 后，私有 trace 归档并按 64 MiB 分片上传；三次有界重试，校验 size/SHA256 回执，保留原始数值。网络失败保留训练成果，重试只做导出：

```bash
bash scripts/run_retacred_q_ecpa_formal_single_seed.sh \
  --export-only --resume runs/retacred_q_ecpa_formal_single_seed/<时间戳>_seed13
```

报告默认 `reports/retacred_q_ecpa_formal_single_seed/<同一时间戳>_seed13/`，只含汇总、指标、config、代码/数据指纹、计数、私有证据回执和审计索引，不含句子、原始数据、NPY、权重或日志。导出采用校验后原子发布，失败会清理临时报告目录。自动 push 目标是 1.1，失败后修复 Git 网络/权限再用上述 export-only 重试；不允许强推。

检查自动提交结果，必要时仅补推同一报告：

```bash
git status --short --branch
git diff --cached --name-only
git log --oneline origin/1.1..HEAD
git -c http.version=HTTP/1.1 push origin HEAD:1.1
```

不要直接 `git add .`；如需手动补提交，先 `git add -- reports/retacred_q_ecpa_formal_single_seed/<同一时间戳>_seed13`，确认 staged 只有该目录，再 `git commit -m "reports: Q-ECPA seed13"`。有其他修改、分支不符、计数/哈希/证据缺失时停下反馈。

完整单 seed 主指标相对 Disabled 任何正增益可按项目规则候选 L1；经典对应相对增益严格大于 1% 才是 QI 候选。量子归因另行判断，不能由 toy 或单 seed 宣称量子优势。多 seed 由项目方审计后另行授权，不自动启动。
