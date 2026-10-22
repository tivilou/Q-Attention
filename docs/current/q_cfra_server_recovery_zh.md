# Q-CFRA 新服务器恢复与合作者更新

2026-10-22：原项目服务器已回收，代码开发环境、固定数据和私有 HTTPS 交换服务已在新 3080Ti 恢复。正式 seed13 仍由合作者在自己的服务器执行。这里只完成测试、合成 GPU canary 和上传验证，没有启动完整数据训练。

合作者仍使用原入口：

```bash
bash scripts/run_retacred_q_cfra_formal_single_seed.sh --gpu auto
```

首次启动前必须先完成下面的分支、证书和私密配置更新。完整预算、暂停恢复和报告回传规则见 [正式单 seed 文档](retacred_q_cfra_formal_single_seed_zh.md)。

## 同步与私密配置

确认 `git status --short` 为空，先整合 `origin/1.1`，再合入 `origin/main`；有脏文件或冲突就停止反馈。

```bash
git status --short
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
git status --short
```

负责人会通过私密渠道提供新的 `client.env`。它同时设置 `PROJECT_EXCHANGE_URL` 和 `PROJECT_EXCHANGE_TOKEN`，放在仓库外并以 `chmod 600` 限制权限。新服务使用标准 HTTPS 443；旧地址和旧 token 不再用于本项目。不要将文件、token 或环境变量展开结果发到聊天、截图、日志或 Git。

将文件安全保存为 `$HOME/.config/q-attention/exchange.env` 后执行：

```bash
chmod 600 "$HOME/.config/q-attention/exchange.env"
source "$HOME/.config/q-attention/exchange.env"
sha256sum scripts/collab/certs/qattn-exchange-ca.crt
```

证书文件 SHA256 必须为：

```text
dbe9611905fdfb37eec630bfe275576e408b94c1672c428ba3a5befa10d84e4b
```

证书从 `main` 同步；不要关闭 TLS 校验。保持现有兼容环境：Linux、Python3.10+、PyTorch2.7.1、NumPy1.26.4 和可用 CUDA。`PYTHON_BIN` 可指向合作者自己的解释器；开发服务器的环境路径不是合作者的要求。

## 检查后启动

```bash
"${PYTHON_BIN:-python}" scripts/check_qattn_exchange.py
bash scripts/check_retacred_q_cfra_formal_single_seed.sh --gpu auto
bash scripts/run_retacred_q_cfra_formal_single_seed.sh --gpu auto
```

交换检查只读验证 HTTPS 和项目鉴权，不打印地址或 token。随后原预检验证完整数据数量和哈希；正式入口仍会在合作者 GPU 上自动跑合成 canary，通过后才进入完整数据训练。本次换服务器没有修改运行源码、seed、数据、预算、控制组或报告白名单。

## 已有运行目录

保留全部原目录，不重开、不清理、不重训。先检查当前服务是否确实持有旧上传收据对应的分片：

```bash
"${PYTHON_BIN:-python}" scripts/check_qattn_exchange.py \
  --run-dir runs/retacred_q_cfra_formal_single_seed/原启动时间_seed13
```

检查通过后，未完成训练使用原 `--resume`；完成训练但尚未交付使用原 `--export-only --resume`。检查若发现旧收据对应文件缺失，立即停止导出并返回非敏感错误，由负责人审计恢复；不要删除收据来绕过检查，也不要把历史收据当作新服务已有文件的证明。历史私有上传内容不能仅凭服务重部署认定为恢复。

## 交付核验

默认 runner 完成后自动审计、上传并提交推送至 `1.1`。核对自动生成的报告提交与路径：

```bash
git status --short
git diff --cached --check
git diff --cached --name-only
git log -1 --oneline
git show --stat --oneline HEAD
git ls-remote origin refs/heads/1.1
```

如果按正式文档使用 `--no-push`，只添加那一次运行的 `reports/` 目录，检查暂存文件后再提交并推送。返回最终 commit、报告目录以及任何未完成项。仅 seed13；多 seed 等负责人审计决定。

验证证据：[新服务器恢复检查](../validation/q_cfra_server_recovery_20261022.json)。服务器端已通过43项测试、8作业 GPU 生命周期 canary、外网8MiB上传和完整回读，以及真实 Q-CFRA 合成案例导出上传和回读。它们验证部署可用性，不构成正式任务性能结果。
