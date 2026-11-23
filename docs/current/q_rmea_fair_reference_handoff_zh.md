# 原版 Q-RMEA：共同缓存五路公平对照

本轮交给师弟执行固定单种子全量对照。训练核心与启动、收尾、回传已通过定向检查；实际服务器还须具备我们安装的精确分配、准入和 COS 发布器绑定。本文不表示实验已启动或新 L3/L4 已成立。

本轮先回答：用同一份固定特征缓存、完整原预算和选模方式，原 Q 是否分别优于 disabled/uniform、去相干和两个经典对照。单种子通过也不自动获得新 L3/L4，更不证明量子独有性。

## 固定实验

TRAIN 58,465、VALID 19,584、TEST 13,418；五路 uniform、quantum、dephased、softmax12、real12。固定 seed13、20 epochs、batch128、FP32、AdamW(lr=.001, weight_decay=.01, betas=.9/.999, eps=1e-8)、clip1。每轮按 Torch(seed13+epoch0) 排序，完整 VALID all40 Macro 严格提高才更新 best，相同值保留最早 checkpoint；最终 VALID/TEST 指标来自该 selected checkpoint。

共同重建缓存 manifest：dfb347aef35657c5463486e1043aca0617ccc11d3aca1eea851a59af15ec5f49。旧四轮 native 只记录输入来源，不作为本轮对照；旧 fresh5 另表保留。非 uniform 注册参数均为81,028，不意味着有效容量相等。

主门槛：Q 的 TEST all40 Macro 分别严格大于其余四路。任何一项差值不正则关闭该固定配置的晋升；不补种子、改指标、加轮数或挑 checkpoint。完整报告全部正负差值。后续独立确认必须先审查结果、另行冻结计划；本脚本不会自动运行多种子。

## 环境与准备

由我们先经 research-infrastructure 分配现有2080 Ti节点，准备私有 execution binding、精确代码/准入、固定缓存和已核实存活的 owner COS 发布器。师弟执行正式训练；我们不代启动新全量。当前 owner 输入路径为 `/hy-tmp/Q-Attention/inputs/native-baseline/discovery`，不得据此假设师弟自己的服务器已有缓存。

私有绑定默认位于 `/etc/q-attention/owner-handoffs/fair-reference.json`；本轮常驻启动固定使用该文件；不得改成另一个绑定文件名。文件及其祖先须由可信 root 账号控制，禁止组/其他用户写入及符号链接；沿用现有节点权限模型，具有该节点 root 权限的操作者属于可信管理方。其中只有路径、哈希、分配和 owner 发布器核验记录，不能包含云密钥；绑定文件和运行输出均不提交 Git。该准备记录只绑定发布器身份，不声称它当前存活。首次训练更新前，既有恢复通道必须收到该运行的 COS preparation 真实确认；缺失时等待并保留现场，不开始优化更新。师弟不修改绑定或准入文件。

启动前激活已准备的 Python 环境；2080 节点的解释器为 `/hy-tmp/Q-Attention/env/bin/python`，须使用既定环境变量：`export PYTHON_BIN=/hy-tmp/Q-Attention/env/bin/python CUBLAS_WORKSPACE_CONFIG=:4096:8`。脚本按 `PYTHON_BIN`、`python`、`python3` 顺序选择，不自行安装或替换依赖。

auto 仅使用分配中的显存可用量至少8 GiB的GPU，按最弱入选卡记录 low-memory/balanced/high 调度档；五个独立条件在这些卡上排队，不改 seed、数据、batch 或轮数，不调用未分配的空闲卡。

当前双2080节点已配置仅此仓库的写入部署密钥，私钥留在节点，GitHub主机密钥经官方API固定校验。入口允许同一仓库的精确HTTPS或 `git@github.com:tivilou/Q-Attention.git` 地址，fetch与唯一push地址须一致；SSH地址别名、重写和多目标配置均拒绝。师弟无需在节点输入个人token。

## 发布完成后执行

在 Q-Attention 项目根目录同步，先合入远端1.1，再合入 main：

```bash
git fetch origin --prune
git switch 1.1
git merge --no-edit origin/1.1
git merge --no-edit origin/main
```

普通启动只用一条命令；确认常驻监督器启动后即返回 STARTED 与进程/控制目录，关闭该终端不影响实验。监督器负责训练、进度与收尾，完成后自动回传报告；不要因命令已返回而重复启动：

```bash
bash scripts/run_q_rmea_fair_reference.sh
```

仅检查绑定、分支、环境与GPU，不训练：

```bash
bash scripts/run_q_rmea_fair_reference.sh check
```

本地 best/latest 持续保存；固定10%进度节点封口最小恢复状态，由既有 owner 发布器直接写入 COS、固定 VersionId、校验与登记后确认。发布器缺失或失败时不要绕过等待、关闭 COS 或重建实验目录。更换节点由我们从固定恢复索引处理，师弟不自行拼接 checkpoint。

续训需要我们先确认旧进程已停止并准备新的分配/常驻启动记录，不能复用已消耗的启动记录。准备完成后可先用 `check --resume`（恢复 COS 时再加 `--restore-pin`）核对同一续训合同，再按我们指示使用：

```bash
bash scripts/run_q_rmea_fair_reference.sh run --resume
```

从已安装的固定 COS 恢复记录继续时，再按我们指示添加 `--restore-pin`；它必须与 `run --resume` 同用。

## 自动回传及失败处理

训练结束后自动验证并导出七文件，提交并普通推送到1.1：summary.json、manifest.json、analysis-readiness.json、receipt.json、run_summary.json、data.sha256、data_counts.txt。目录为 `reports/q_rmea_fair_reference/<启动时间>_seed13/`。不提交缓存、模型、预测、私有绑定、完整原始日志或云索引。

提交前检查分支/远端/祖先、完整报告、训练提交对应源码、暂存的精确路径和字节；推送固定提交，禁止带出本地标签或无关提交。报告提交或推送失败后，保存私有运行目录，只重试：

```bash
bash scripts/run_q_rmea_fair_reference.sh report
```

此命令不训练，也不依赖原GPU分配仍然有效。远端1.1前进时先普通合并 origin/1.1，再重试 report；有冲突、无关修改/提交、未知模型身份、缓存不符、准入失败、OOM或发布器问题时保留现场并反馈，禁止 force push、删除重开、改预算或改科学配置。推送失败不重新运行训练。
