# 线性 OOM 降档与当前实验续跑

适用于 `a90af7d` 发布版创建的 `runs/retacred_reference_complete/<原时间戳>`。无需重跑已经保存的训练更新，也无需转换或重写大 checkpoint。

## 改动

- 大档固定减量：起点为每卡256时，256→248→240→…→16；小档每次减1：16→15→…→2→1。实际起点决定大档步长 `ceil(start/32)`，不按当前档位除2。最后仍 OOM 才启用激活重计算。
- 当前 auto 任务升级时只向上探测1个 micro；例如旧版 m1 试 m2，实际 OOM 后回 m1。显式指定 micro、H3 的单句语义以及已经启用的重计算档保持。一次探测记入持久状态，反复重启不会重复探测。
- 只有真正的 CUDA OOM 才降档。逻辑 batch、完整候选集合、损失权重、每逻辑 batch 一次优化器更新、学习率、训练预算和评估口径保持。
- 旧科学契约和 checkpoint 哈希保持；新执行源码与升级记录分别绑定。旧运行的数据、模型、运行环境、GPU 数量和原始 rank 顺序仍须通过严格检查。升级记录也随最终 report 导出。
- `VRAM` 为最近一次更新结束时的 PyTorch allocated；`peak` 是该次训练更新的 allocated 峰值及设备总容量，宽终端另显示 peak reserved。不是 `nvidia-smi` 整卡实时占用，也不包含其他进程；更新尚未结束时显示仍是上一次值。

## 师弟操作

先在正在跑的总脚本终端按一次 Ctrl+C，等待脚本退出后更新。保留当前 run 目录和已下载的 assets：

```bash
git -c http.version=HTTP/1.1 fetch origin
git merge --no-edit origin/main

bash scripts/run_retacred_all_references.sh \
  --gpu auto --models all --seeds 13 \
  --resume runs/retacred_reference_complete/<这次正在跑的时间戳>
```

原命令若显式指定 `--micro-batch`、`--checkpoint-seconds` 或 `--ctl-gpus-per-task`，续跑时须使用原值。使用相同 GPU 选择与原顺序、现有 Python/CUDA 环境和 `PROJECT_EXCHANGE_URL/TOKEN`。

恢复从最后完整落盘的优化器更新开始；突然终止可能重放落盘后的少量更新。只兼容这次明确绑定的历史源码，不支持任意旧版或更换 DDP GPU 拓扑。分割方式改变可能改变 dropout 随机数的消费和浮点舍入，不能承诺与原分割逐位相同。

## 验证

48个聚焦测试通过；RTX3060上实际运行旧源码保存和新版恢复的两步 canary：相同物理分割下，模型、优化器、调度器和 RNG 与连续运行完全一致，旧契约哈希未变。另有实际 CUDA OOM 的父子进程恢复测试通过。没有运行正式全量实验；本次 Titan 多GPU主机无法连接，未重测其硬件吞吐。
