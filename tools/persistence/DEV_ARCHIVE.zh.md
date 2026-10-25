# 开发测试材料自动归档

本工具用于本方的单元测试、小样本检查、诊断和预检材料。Q-Attention 的完整正式实验仍由合作者执行，代码和审核报告继续通过 GitHub 交换。

它在指定测试结束后自动打包选定材料；控制端发布时自动完成 COS 上传、固定版本回读、独立恢复和私有 Git 索引提交。控制端入口可用一条命令串起两阶段；没有后台监听器或任务调度服务，也不会接管任意一次直接运行的 pytest/unittest。

## 控制端：一条命令

dev_archive_owner.py 通过 serverctl 解析指定项目的 development 绑定，只连接 profile 明确指定且已验证密钥指纹的服务器；严格 SSH 执行 capture，拉取包后才在控制端加载 COS 凭据并发布。

```bash
python /PATH/TO/tools/persistence/dev_archive_owner.py --profile /PRIVATE/owner-dev-archive.json
# 上传或索引失败后，使用已封存的本机目录重试；不再连接服务器或运行测试：
python /PATH/TO/tools/persistence/dev_archive_owner.py --profile /PRIVATE/owner-dev-archive.json --retry /LOCAL/SEALED_ATTEMPT
```

profile 只含本机操作信息和凭据路径，存放在 Git 与快照之外。例如：

```json
{
  "project_root": "/LOCAL/PROJECT",
  "server_id": "EXPLICIT_SERVERCTL_ID",
  "repo": "/REMOTE/CLEAN_CHECKOUT",
  "python": "/REMOTE/PYTHON",
  "recipe": "tools/persistence/dev-test-example.json",
  "remote_root": "/REMOTE/OWNER_DEV_ARCHIVES",
  "local_root": "/LOCAL/OWNER_DEV_ARCHIVES",
  "index": "/LOCAL/PRIVATE_INDEX_CHECKOUT",
  "credential_file": "/PROTECTED/COS_CREDENTIAL_REFERENCE.json"
}
```

credential_file 使用既有受保护的 SecretId/SecretKey JSON，可含临时 Token；不得把值写入 profile。控制端须已有正常 Git 作者身份、私有仓库访问和所选服务器 SSH host key。连接中断后的远端执行状态可能未知，此时保留 transport.json 并检查原 attempt；不要直接启动第二次测试。网络代理如需要，应按命令设置 Git 的临时配置，不修改全局设置。

## 工作端：测试后封存

先提交代码，保持 checkout 干净。准备明确的 CPU 测试 recipe；示例是本目录的 dev-test-example.json。timeout_seconds 为 1–900 秒；command 是参数数组，第一个参数为 `{python}`，不经 shell。`{output}` 和 RESEARCH_OUTPUT_DIR 指向本次专用输出目录。

```bash
python tools/persistence/dev_archive.py capture \
  --repo /PATH/TO/CLEAN_CHECKOUT \
  --recipe tools/persistence/dev-test-example.json \
  --out /PATH/OUTSIDE_CHECKOUT/NEW_ATTEMPT
```

recipe.include 只能列出明确的、已跟踪的普通文件；必须包含 environment_file。代码材料保存的是该 commit 的精确 Git blob。outputs 列出本次输出目录中的文件；只归档列出的输出，不扫描整个项目或历史 runs。日志、recipe、Python/包版本清单、测试状态会自动纳入。

capture 限 Linux，使用进程组和超时终止普通子进程，隐藏 CUDA，并从子进程环境移除常见凭据变量。仅运行已审查的可信同步测试；这是操作约定，不是针对恶意程序的安全沙箱。测试不得启动脱离进程组的后台服务、修改源码或读取凭据。

成功封存会生成 SEALED.json 和 package/。测试通过退出 0；测试失败或超时但材料封存完整时退出 2。失败材料也可保存，测试状态始终保留为 failed/timed_out。启动、输入或打包失败退出 1，不能发布未封存目录。

## 控制端：自动发布

只把 SEALED.json 和 package/ 复制到控制端新目录，工作端无需 COS 密钥。控制端使用 Python 3.10+、requirements-cos.txt 中固定版本的 SDK，以及项目已有的受保护凭据加载方式；不要在命令行或 Git 中填写密钥值。

使用已确认 PRIVATE 的共享恢复索引的专用 main checkout。项目 deployment.json 的 project_id、storage_namespace、cos_bucket/cos_region 必须与共享 storage.json 一致。该项目的 COS 前缀权限必须已经独立授权。

```bash
python /PATH/TO/tools/persistence/dev_archive.py publish \
  --attempt /PATH/TO/COPIED_ATTEMPT \
  --index /PATH/TO/PRIVATE_INDEX_CHECKOUT
```

publish 将材料写入 `projects/<project-id>/dev-tests/`，复用现有固定 VersionId 的快照协议。在全文件独立恢复成功后，将紧凑回执写入共享私有仓库的 `projects/<project-id>/snapshots/dev-<attempt-id>.json`，提交并推送 main。Git 只包含回执，不包含代码包、日志、数据或凭据。ARCHIVED.json 仅在远端索引包含提交后生成；测试成功与归档成功是两个状态。

## 中断和重试

原样保留 SEALED.json、package/ 和控制端回执，再运行同一条 publish。不要重跑测试来重试上传，也不要换 snapshot ID 或覆盖历史索引。相同数据块会被验证后复用；已有可信 COS 回执会使用原固定版本。索引 push 失败时，本地单个待发布提交可继续推送；暂存后 commit 失败也可在排除原因后重试。

并发发布使用工作副本及 attempt 的操作系统锁；进程退出会释放锁。共享索引仅允许当前记录的待发布更改。若其他人已更新远端，而本地有待推送提交，工具停止，保留材料供人工合并；不自动 rebase 或强制推送。归档冲突、下载损坏、错误项目或缺失版本均不能产生完成标记。

## 恢复与范围

从可信私有索引记录取得 publish_receipt，按 qpersist.py restore 的 COS 参数恢复；prefix 使用 `projects/<project-id>/dev-tests/`，snapshot、manifest_sha256 和每个对象的 VersionId 以回执为准。

本归档覆盖 recipe 明确选择的开发测试材料。完整源码通过对应 Git commit 的源代码仓库取得；这里的选定代码文件不等于全项目 Git 历史。environment_file 是声明的环境定义，另存的包版本清单是执行环境观察；二者不自动证明环境完全一致或依赖长期可下载。没有保存真实数据、完整历史实验、活动检查点或训练续跑状态。恢复验收不授权释放任何其他服务器。
