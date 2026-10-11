# CTL-DRP 单 GPU 诊断边界

这是两条合成句子的小规模检查，不是 Re-TACRED 全量实验。

- `--tiny`：随机两层、宽度32的模型；结果仅证明当前 GPU 路径可执行。
- `--weight FILE --variant sup|unsup`：离线核对官方候选的固定 revision、完整 SHA256、大小、配置与参数后加载。不自动下载，不回退到随机模型。
- 保留原始50265个 embedding 行，只增加4个实体标记。原词表固定输入在移除 pooler/增加标记前后必须完全一致。
- 共享 scalar head + 每句候选 softmax/NLL；不经过 SimCSE pooler 或 L2归一化。
- CUDA FP32、非 fused/foreach AdamW、固定 dropout 与确定性配置。保存一次 post-update checkpoint，重放下一次更新，比较模型、优化器、CPU/CUDA/Python/NumPy RNG 的真实字节。
- checkpoint 只用于此单可见 GPU 诊断，不代表 AMP、多 GPU 拓扑变化或全量数据 cursor 已实现。
- 训练/测试诊断分别保存完整实际数值 `.pt`（embedding、层输入输出、QKV、attention、读出、分数、概率；训练包括参数梯度），安全重载逐 tensor 验证。shape/摘要不能代替数值。
- 必须使用新的私有输出目录。缺权重时保留 pending，不以 tiny 结果填充真实预训练证据。

从任意项目位置调用项目根目录的 `scripts/check_retacred_ctl_drp_cuda.sh`；脚本从自身路径定位源码，不写死服务器目录。Python 由 `PYTHON_BIN` 或当前环境解析。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/check_retacred_ctl_drp_cuda.sh \
  --cache <已锁定的小文件缓存> --paper-source <已锁定的原论文tex> \
  --output <新私有目录> --weight <校验通过的pytorch_model.bin> --variant sup
```

权重身份仍是我们冻结的官方候选解释，不宣称作者使用的 exact checkpoint；历史 SimCSE 审计锁保持不变。完整 adapter、训练配方、数据身份、报告与张量交付、独立审查通过前仍不加入默认正式实验队列。
