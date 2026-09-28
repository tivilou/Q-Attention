# Q-Attention 文件交换

本项目使用独立的 HTTPS 文件交换服务，地址为：

```text
https://117.50.198.37:18084
```

服务只接收明确声明的文件。当前 Q-EPVG 诊断上传命令会从指定 multi-seed 运行组中收集 `q_epvg_zz_value_only_quantum` 的每个 seed 下的 `case_study.json` 和 `sample_trace.json`，不会上传完整 `runs/`、checkpoint、日志或数据集。

## 师弟执行

先同步 `main`，再在仓库根目录执行：

```bash
git fetch origin main
git checkout 1.1
git merge --ff-only origin/main
export PROJECT_EXCHANGE_URL='https://117.50.198.37:18084'
export PROJECT_EXCHANGE_TOKEN='<通过私密渠道取得的 Q-Attention token>'
python scripts/upload_qepvg_case_study_diagnostics.py \
  --group-dir runs/retacred_q_epvg_formal_multi_seed/20260927T071350Z \
  --selector q_epvg_zz_value_only_quantum
```

如果不设置 `PROJECT_EXCHANGE_TOKEN`，脚本会在终端安全地提示输入。仓库中的 `scripts/collab/certs/qattn-exchange-ca.crt` 用于校验服务证书，不要关闭 HTTPS 校验，也不要把 token 写入脚本、日志或 Git。

上传成功后，脚本会打印服务端回执和本地 pending manifest 路径。服务端目录为：

```text
q-attention/qepvg-case-study-recovery/20260927T071350Z/
```
