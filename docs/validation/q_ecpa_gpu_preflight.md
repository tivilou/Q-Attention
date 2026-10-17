# Q-ECPA 正式运行包：有界 GPU 验收

日期：2026-10-17。这里验证实际 CUDA 运行和恢复，不是 Re-TACRED 正式实验结果。

## 冻结版本和环境

- 源提交：`6544708fcb2bd9de568a8afa6ab2b392c200c7ee`；没有修改已审查的实现、配置或测试。
- 运行源集合 SHA-256：`7e40d4f728c624bee2c4a9be929acb64a57385a9977eb765bdb39da0dd267f08`。
- 15 文件审查清单 SHA-256：`4d954d2a65dd260c1db142698383f769cf5e0b5d08713a139606f230a3024e7f`。
- 原审查：`q_ecpa_formal_review.json`；本次执行范围补充：`q_ecpa_gpu_review.json`，PASS_WITH_RISK，无本范围 must-fix。复用同一冻结代码的原审查，没有虚构一次新的外部咨询。
- 自有、已绑定的 GPU 服务器 `gpushare-i1-54996-titan5`；实际可见两张 GTX TITAN X，各约 12 GiB，计算能力 5.2。
- 使用现有环境：Python 3.10.22、PyTorch 2.7.1+cu118、NumPy 1.26.4；未安装或修改依赖。隔离目录 `/tmp/qattn-ecpa-gpu-6544708` 由权威开发 worktree 的 Git bundle 克隆，代码和审查哈希一致。
- 仅使用预先定义的短句合成 canary；逻辑 batch 4，每个可学习任务最多 6 次更新，固定控制无优化更新。没有真实数据 GPU 训练。

## 实际入口与恢复

均使用正式 shell 入口 `scripts/run_retacred_q_ecpa_formal_single_seed.sh`，没有另写训练路径。`RUN=runs/ecpa_gpu_canary/20261017_titan_seed13`。

| 检查 | 入口参数 | CUDA_VISIBLE_DEVICES | 退出码 | 实测耗时 |
| --- | --- | --- | ---: | ---: |
| 两 GPU 启动、注入 OOM、安全暂停 | `--canary --gpu auto --output-dir RUN --inject-oom-once --stop-after 2` | `0,1` | 85 | 38.643 秒 |
| 同一 run 单 GPU 恢复完成 | `--canary --gpu auto --resume RUN` | `0` | 0 | 123.031 秒 |
| 完成后换回两 GPU | `--canary --gpu auto --resume RUN` | `0,1` | 0 | 3.517 秒 |

父进程完成 baseline 依赖后，将独立 selector 分配到两张卡。15 次子进程观察覆盖两卡，每个 worker 的 `CUDA_VISIBLE_DEVICES` 等于所分配的物理 GPU；这不是单任务 DDP。Quantum 注入 OOM 后持久化 tier 1、物理 micro 3，逻辑 batch 仍为 4；在完整更新边界暂停后，单卡恢复没有重新从最大 micro 试起。再次恢复已完成的目录没有启动重训或改写任何 latest checkpoint。

baseline 和七路 selector 全部通过；六个可学习任务各 6 次优化更新，disabled/random_phase 各 0 次。全部 checkpoint 模型有限，采样到的 CUDA 内存峰值为正。短句 canary 中最大的采样 PyTorch tensor 峰值约 0.01752 GiB；这不是驱动统计的总占用，更不能用来估算真实长句或全量吞吐。

## 完整案例数值

8 个任务共 222 条 `sample-trace.v2`、10,554 份完整数值文件，Quantum 为 28 条、1,428 份。逐份解码、内容哈希、任意张量坐标读取以及语义数值链路校验通过；包含固定 train/valid/test 案例的 initial/best/final 重放，可学习任务另有首个真实训练更新。固定控制没有虚构训练更新。

这证明私有 run 中存在完整值，尚不代表门户已成功展示。原始 trace、NPY、checkpoint 和运行日志留在私有目录；本次仅将聚合验收证据纳入 Git。

## 真实 CUDA OOM 的独立检查

另外运行一个临时进程，仅尝试一次大于单卡物理总显存的分配：设备总量 12,799,967,232 字节，请求 12,801,015,808 字节。实际捕获 `torch.cuda.OutOfMemoryError`；本进程错误前后已分配 tensor 字节均为 0，随后的小型 CUDA 计算和同步成功。探针退出后无其 GPU context 遗留，其他进程未被操作。

必须区分：父子训练恢复用的是显式 OOM 故障注入；独立探针证明真实分配 OOM 能被分类及该进程可恢复。二者不能合并声称“真实训练步骤 OOM 已通过整个恢复路径”。

## 当前边界

- 已通过：冻结代码在实际 CUDA 上运行；两卡独立任务队列；注入 OOM 档位持久化；安全暂停；单卡同 run 恢复；完成后拓扑变更不重训；完整案例值读取；独立真实分配 OOM；无本次遗留子进程。
- 尚未验收：真实训练步骤硬件 OOM 的完整恢复路径、完整真实数据 GPU 显存与吞吐、生产 HTTPS 上传、门户完整值导入与交互切片。
- 本轮没有执行全量正式训练、没有推送 GitHub，不产生 L1/L2 或量子优势结论。完整 seed 13 仍交由合作者执行。

机器可读证据：`q_ecpa_gpu_preflight.json`。硬件执行前的范围审查：`q_ecpa_gpu_review.json`。此前 CPU 和数据兼容证据：`q_ecpa_formal_preflight.{json,md}`。
