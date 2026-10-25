# 项目持久化与恢复（第一版）

本工具保存**已经关闭或确认不可变的文件集合**。它不启动实验，不调整既有 checkpoint 恢复规则，不自动删除云端历史文件。当前账号尚未申请 OpenAPI 应用；真实阿里云盘备份和全新服务器部署尚待验收。不要据此释放原服务器。

GitHub 公开仓库存工具、代码和环境定义；私有 `tivilou/Q-Attention-persistence` 存恢复索引、清单哈希和脱敏回执；阿里云盘存原始数据、检查点、完整 traces 和代码 bundle。私有 GitHub 也不存这些大文件或凭据。Windows 本机保留额外副本。

## 文件选择与一致性

先停止写入，或选取 runner 明确提交且不会再清理的检查点代次。活动目录的文件复制不能保证逻辑一致性。`--closed` 是操作者对此的声明，不是工具对运行状态的证明。第一版不自动截取运行中的 checkpoint，也不调度无人值守备份。

`--include` 必须显式指定相对文件或目录，不能选择 `.`。目录递归包含所有普通文件；不要选整个 home、仓库根或秘密目录。拒绝符号链接、常见密钥路径、路径穿越和跨平台大小写冲突。这不是任意文件内容的秘密检测器，操作者仍须审查选择范围。

私有 `metadata.json` 必须仅含以下四项，使用实际值：

```json
{"kind":"closed_run","code_commit":"完整40位代码commit","environment_sha256":"环境定义文件的64位SHA256","dataset_sha256":{"train":"64位SHA256","valid":"64位SHA256","test":"64位SHA256"}}
```

可用 kind：dataset、closed_run、code、environment、canary。代码 bundle 和环境定义应作为单独快照保存或明确加入文件集合。清单记录的代码 commit 必须能从 GitHub 或已经验证的 bundle 恢复。

metadata 中的代码 commit、环境 hash 以及未包含在快照中的依赖数据 hash 是**操作者声明的来源信息**，本工具不据此认证训练来源。原 runner 必须再次校验。`dataset_sha256` 的键使用数据文件相对路径；若该文件包含在快照中，工具会强制比对声明与实际 hash。`kind=dataset` 时，声明不能为空且所有声明的数据文件必须包含在快照中。

```bash
python tools/persistence/qpersist.py inventory --source /private/closed-artifacts \
  --include completed-run --metadata /private/metadata.json \
  --out /private/spec.json --closed
python tools/persistence/qpersist.py pack --source /private/closed-artifacts \
  --spec /private/spec.json --out /private/package
```

64 MiB 块采用串接字节流，清单记录每个文件长度、SHA256、执行权限。不会保存 uid/gid、ACL、软链接或扩展属性；不要用它克隆操作系统或直接搬运整个 conda 环境。空文件受支持。打包需要额外约一份所选文件大小的磁盘空间。

## 上传和恢复

先用独立磁盘验证工具；同一 GPU 服务器内的另一个目录只算测试，不算持久化：

```bash
python tools/persistence/qpersist.py publish --backend local --store /independent-storage/objects \
  --package /private/package --receipt /private/receipt.json
```

阿里云盘应用获批并完成用户授权后，在本机安全地注入短期 `ALIPAN_ACCESS_TOKEN`，不在聊天、Git、命令参数或文档中填写它。指定用户授权范围内的独立项目目录，拒绝直接使用根目录。

```bash
python tools/persistence/qpersist.py publish --backend alipan --drive-id DRIVE_ID --folder-id PROJECT_FOLDER_ID \
  --package /private/package --receipt /private/receipt.json
python tools/persistence/qpersist.py status --backend alipan --drive-id DRIVE_ID --folder-id PROJECT_FOLDER_ID \
  --snapshot SNAPSHOT_ID --manifest-sha256 TRUSTED_MANIFEST_SHA256
python tools/persistence/qpersist.py restore --backend alipan --drive-id DRIVE_ID --folder-id PROJECT_FOLDER_ID \
  --snapshot SNAPSHOT_ID --manifest-sha256 TRUSTED_MANIFEST_SHA256 --out /private/restored-run
```

上传逐块回读核验后才上传 manifest；manifest 是提交点，回执写在最后。失败不产生成功回执。已完成同名块可核验复用；同名不同内容和重复云端名称会停止。中断在 `create/PUT/complete` 之间可能留下未完成对象：第一版失败关闭，需检查未完成对象后重试，不自动删除。没有已保存的上传会话时，不宣称能续传未完成的单块。已完成块的重试不会重传完整快照。

restore 要求来自私有 GitHub 索引的可信 manifest SHA256；仅知道云端快照 ID 不足以恢复。全部内容校验后才生成最终目录；目标必须不存在。失败清理本次临时恢复目录。`status` 只验证提交清单，不重读所有大文件。回执 `remote_readback_verified` 与独立机器恢复演练是不同状态。

第一版仅支持单写入者、受信任的本机目录，不能在多个机器同时发布同一 package。不要手改远端对象。Alipan API 的写权限本身包含删除权；本工具不调用删除接口，并不等于平台级不可变存储。

本地后端要求支持原子同文件系统 rename 和硬链接的文件系统（Linux 本地磁盘或 NTFS）；NFS/CIFS 等网络文件系统未认证。断电/内核崩溃后的目录项持久性尚未实测，不能以进程级故障测试替代电源故障验证。

Linux 会同步文件和相关目录项；Windows 标准库不能等价同步目录句柄，所以 Windows 副本不具备已认证的断电持久性。恶意进程并发替换路径不在单写入者/受信任目录模型内，路径检查不能抵御这种 TOCTOU 攻击。磁盘同步在最终目录发布后失败时，目录可能已经可见但命令失败；需重新校验，不能强行覆盖。

API 使用官方 `openapi.alipan.com`，没有非官方 WebAPI 或第三方 token 中转。鉴权失效立即停止；不存储、不共享、不自动刷新 refresh token。长期自动化需后续实现独立于 GPU 服务器的串行令牌刷新与安全存储。当前受控人工授权适用于首轮验收。

## 新服务器安装

准备 Linux x86_64、Python 3.10（原环境 3.10.22）、git 与可用 NVIDIA 驱动。从固定工具 commit 获取本目录，检查私有索引后执行：

```bash
EXPECTED_ENV_SHA256=TRUSTED_LOCK_SHA256 PYTHON_BIN=python3.10 \
  bash tools/persistence/bootstrap.sh FULL_CODE_COMMIT /new/project
# 若 commit 尚未发布到公开仓库，先从已验证快照恢复 Git bundle，再传第三个参数：
EXPECTED_ENV_SHA256=TRUSTED_LOCK_SHA256 EXPECTED_BUNDLE_SHA256=TRUSTED_BUNDLE_SHA256 \
  PYTHON_BIN=python3.10 bash tools/persistence/bootstrap.sh FULL_CODE_COMMIT /new/project /absolute/verified.bundle
```

脚本只安装环境和指定代码；失败保留 `DEPLOYMENT_INCOMPLETE`，成功生成 `ENVIRONMENT_INSTALLED`。不会启动训练。版本清单来自现有环境，尚未在全新服务器验证安装；没有包制品哈希或离线 wheel 镜像，不承诺长期第三方软件源可用性。

脚本强制比对私有索引中可信的环境定义 hash；使用 bundle 时也必须提供独立可信的 bundle hash。`git bundle verify` 只校验 bundle 结构/依赖，不验证作者签名；第一版不提供 GPG 签名认证，信任入口是受控私有索引和固定代码 commit。

随后恢复所需数据/实验快照，运行原实验 preflight。CPU→CUDA、不同 GPU/运行库等变化仍需原 runner 判定；不得通过修改指纹续训。环境安装成功不代表 CUDA 可用，也不代表 checkpoint 可续训。

## 阿里云盘账号待办

在 [开放平台](https://www.alipan.com/developer) 登录个人账号，按[接入流程](https://www.yuque.com/aliyundrive/zpfszx/btw0tw)申请符合个人研究备份用途的应用，并确认应用类型、回调地址与文件读写权限。平台审批不是脚本能够替代的。

获批后提供非秘密信息：client_id、应用类型、允许的回调 URI、是否允许文件夹授权。client_secret 和 token 应在本机专用秘密存储录入，不发聊天。先授权专用项目目录，用合成小文件做上传→断网/失败→重试→另一目录下载验收，再备份真实数据。恢复速度、容量和第三方下载权益以实际账号实测为准。

接口依据：[OAuth](https://www.yuque.com/aliyundrive/zpfszx/efabcs)、[上传](https://www.yuque.com/aliyundrive/zpfszx/ezlzok)、[下载](https://www.yuque.com/aliyundrive/zpfszx/gogo34oi2gy98w5d)。

## 开发验收

```bash
python -m unittest discover -s tools/persistence -p 'test_*.py' -v
# 生命周期审查通过后才运行真实 CLI 故障/恢复用例：
QPERSIST_LIFECYCLE_CANARY=1 python -m unittest discover -s tools/persistence -p 'test_*.py' -v
```

本地后端测试不代表官方 API 实测成功。真实云端验收、私有索引记录及独立恢复演练完成之前，系统状态是未完成。
