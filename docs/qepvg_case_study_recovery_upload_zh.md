# Q-EPVG Case Study 恢复资料一键上传

这条命令只上传恢复 `context` 阶段所需的审计资料，不上传数据集、权重、checkpoint、预测文件或完整日志。

## 师弟执行

在仓库根目录、已经同步最新 `main` 的 `1.1` 分支执行：

```bash
git fetch origin --prune
git checkout 1.1
git pull --ff-only origin 1.1
git merge origin/main

export PROJECT_EXCHANGE_URL="https://117.50.198.37:18084"
export PROJECT_EXCHANGE_TOKEN="从项目方获得的token"
bash scripts/upload_qepvg_case_study_recovery.sh
```

如果自动选择的目录不是本次多 seed 运行，可显式指定：

```bash
bash scripts/upload_qepvg_case_study_recovery.sh \
  --group-dir runs/retacred_q_epvg_formal_multi_seed/<时间戳> \
  --selector q_epvg_zz_value_only_quantum
```

脚本会自动检查成功的 `multi_seed_run_summary.json`、每个 seed 的 `case_study.json` 与 `sample_trace.json`，并依据每个 case 的 manifest 收集以下七类同样本、同 checkpoint 张量：

`q_epvg_query`、`q_epvg_key`、`q_epvg_query_update`、`q_epvg_score_adjustment`、`q_epvg_attention`、`steered_attention_scores`、`q_epvg_routed_values`。

每个文件上传前计算 SHA-256，服务器回执的路径、字节数和 SHA-256 不一致时命令以非零状态退出，并保留 pending manifest 供重试。上传完成后把终端中的目标目录和 manifest 路径发回项目方即可。
