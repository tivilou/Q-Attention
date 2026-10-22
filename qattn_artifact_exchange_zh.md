# Q-Attention 文件交换

2026-10-22：原服务器已回收，项目 HTTPS 交换服务已迁移。旧端点不再作为运行依据。地址和 token 由负责人通过私密渠道提供；新的公开信任证书随 `main` 更新，不要继续使用旧配置或关闭 TLS 校验。

本服务只接收实验入口明确声明的文件，所有私有证据按 `q-attention/<experiment>/<run-id>/` 隔离。公开 GitHub 报告只保留白名单摘要和哈希收据，原始数据、checkpoint、预测、日志和完整张量不得提交 Git。

当前 Q-CFRA 更新步骤和运行命令见 [新服务器恢复指南](docs/current/q_cfra_server_recovery_zh.md)，完整实验契约见 [正式 seed13 文档](docs/current/retacred_q_cfra_formal_single_seed_zh.md)。合作者保持在 `1.1`，按顺序整合 `origin/1.1` 和 `origin/main`。

安全收取配置文件并放到仓库外后，可在仓库根目录执行：

```bash
chmod 600 "$HOME/.config/q-attention/exchange.env"
source "$HOME/.config/q-attention/exchange.env"
"${PYTHON_BIN:-python}" scripts/check_qattn_exchange.py
```

不要打印或提交环境变量值。检查会验证仓库中的 CA、HTTPS 和项目鉴权；成功不代表历史上传内容已经恢复。已有 Q-CFRA 运行还应带 `--run-dir` 检查缓存收据的分片是否存在于当前服务；缺失时保留原始证据和收据，停止导出并反馈负责人。

上传只有在大小和 SHA256 与服务端回执一致时才算成功。同内容重试幂等，已有路径的不同内容会被拒绝，不通过覆盖旧文件解决故障。
