# Q-ECPA：补齐 seed 29、53 的三 seed 复现

本轮复用已审计的 seed 13 报告，只新训练 seed 29、53。主指标仍为 test Macro-F1；模型、完整数据、训练预算和七路条件保持不变。本轮目标是检验可重复性，不把量子版超过经典版设为普通任务效用的必要条件，也不因均值提升而自动宣称量子机制优势。

seed 13 锁定结果提交 `6eed9f458d34815b260acebcdbdd20105309e686` 中的 `reports/retacred_q_ecpa_formal_single_seed/20261018T020502Z_seed13`，由 Git 对象按文件哈希自动导入；无需旧服务器或旧 raw run。请勿手工复制、编辑报告。

## 更新与环境

在项目根目录、已配置好的 conda 环境运行。先保存自己的改动；有冲突时停止并反馈，不要 reset 或强制覆盖。

```bash
git fetch origin --prune
git switch 1.1
git merge --no-edit origin/1.1
git merge --no-edit origin/main
git status --short --branch
python -c 'import torch,numpy; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
nvidia-smi
```

需要项目已有依赖、完整 Re-TACRED 数据和可写磁盘，以及项目私有交换服务的 `PROJECT_EXCHANGE_URL`、`PROJECT_EXCHANGE_TOKEN` 环境变量。不要将 token 写入 Git 或反馈截图。runner 不使用我们服务器的绝对 Python 路径，依次选择 `PYTHON_BIN`、当前 `python`、`python3`。

## 检查与一键运行

```bash
bash scripts/check_retacred_q_ecpa_replication.sh --gpu auto
bash scripts/run_retacred_q_ecpa_formal_multi_seed.sh --gpu auto
```

不传 seed 参数：29、53 和历史 13 已冻结。检查会验证 seed 13 的固定 Git 对象、源代码合同、完整数据数量与哈希及可见 GPU。缺对象时重新 `git fetch origin`，不要改为另一个结果。

`auto` 使用当前可见且空闲显存至少 8 GiB 的 GPU；一张卡也能跑，多张卡共享全局就绪队列，每卡同时一个 worker。两个 seed 先各自训练 baseline；某个 baseline 完成后，该 seed 的七路条件即可调度，不等待另一个 seed。三张 A2000 可以并行独立任务；这不是单模型 DDP。

七路条件为 disabled、quantum、dephased、classical、random_phase、entity_permuted、original_qk。原 QK 是历史参考，层数/容量并非严格匹配。每个新 seed 使用自己训练的 baseline 权重；不同 seed 不共享训练权重。

终端在 TTY 中固定 GPU 行，刷新进度、速率、ETA、显存、档位及 checkpoint 游标，下方展示 pending/completed/failed 队列。文件重定向时输出周期快照，具体 worker 日志保存在本次 run 的各 seed `logs/` 下。

## 停止和续跑

按一次 Ctrl+C，等待安全暂停后退出。运行目录会打印出来，形如 `runs/retacred_q_ecpa_formal_multi_seed/<启动时间>`。

```bash
bash scripts/run_retacred_q_ecpa_formal_multi_seed.sh \
  --gpu auto \
  --resume runs/retacred_q_ecpa_formal_multi_seed/<原启动时间>
```

支持同一 run 在单卡/多卡间切换；已完成的 `(seed, 条件)` 跳过，正在训练的任务从 latest 已落盘的完整 optimizer update 恢复，可能重算尚未落盘的更新。latest 包含模型、优化器、RNG、游标和最佳状态；保留 latest/best。每任务的 OOM 档位独立持久化，续跑不会重新从最高档试起。

保持逻辑 batch 和预算不变；只有发生实际 OOM 才线性减小物理 micro-batch，最终才启用重计算，降档不会混用不同 seed 或改变科学超参数。不能修改源代码/合同后继续旧 run。配置中的 `formal_seed13` 是冻结单 seed 配方的历史标识；新任务实际 `config.seed` 为 29 或 53，不能把该 profile 名称误改为新 seed，否则会改变案例来源标签。

## 完整案例、汇总、自动提交

训练和回放继续使用已验证的全数值案例记录：固定 train/valid/test 样本，initial/best/final 状态与实际训练样本，原句、token、实体及位置、目标、预测、embedding、注意力、路径/相位、残差、logits、梯度等既有完整张量及语义链路。完整值保存为无损 NPY，导出时逐一验证形状、数量、哈希和中间计算关系；缺值即拒绝报告。

原始数据、样本、张量、checkpoint 不进 Git。新 seed 的私有证据使用唯一 run ID 自动上传 18084；公开报告保留三 seed 的完整指标/逐类结果、mean ± 样本标准差、各 seed 差值和配对 t 的 95% 区间、数据哈希、来源和私有证据索引。旧 seed 13 保留已审计的完整私有档案索引。

三 seed 的量子任务效用 L2 门槛：量子减 disabled 的平均 Macro-F1 差值为正，且配对 95% CI 下界大于零。仅三 seed 的区间不确定性较大；经典对照相对 baseline 均值提升严格超过 1% 单独记 QI 候选。accuracy、loss 和 relation micro-F1 等结果同样保留，不能替换主指标。

完成后自动审计、导出到 `reports/retacred_q_ecpa_formal_multi_seed/<启动时间>/`，并自动 commit/push 到 `1.1`。不需要另敲导出和 push 命令。若上传、导出或 Git 推送失败，保留已完成 run；修复网络/分支问题后只重试：

```bash
bash scripts/run_retacred_q_ecpa_formal_multi_seed.sh \
  --export-only \
  --resume runs/retacred_q_ecpa_formal_multi_seed/<原启动时间>
```

该命令不请求 GPU、不重新训练，并复用已验证上传回执。不要删除 run、篡改白名单、忽略案例不完整错误或强推 Git。

仅在自动推送失败且报告已经完整生成时，可以人工检查同一个报告目录再提交；不要 `git add .`：

```bash
git status --short --branch
git add -- reports/retacred_q_ecpa_formal_multi_seed/<原启动时间>
git diff --cached --stat
git diff --cached --name-only
# 确认只有上述报告白名单文件，没有数据、日志、checkpoint 或 NPY；有异常则停止。
git commit -m 'reports: Q-ECPA paired seeds13/29/53'
git push origin HEAD:1.1
```

请反馈最终报告路径与 Git 提交号。正式完整数据训练交由合作者执行；开发阶段的 synthetic canary 不构成效果提升证据。
