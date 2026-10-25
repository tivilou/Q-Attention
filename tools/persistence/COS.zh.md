# COS 快照与恢复

本后端使用启用版本控制的私有 COS 桶；依赖 `requirements-cos.txt`。
单 AZ/多 AZ、地域和命名由配置决定。初始上传使用 STANDARD 存储和 SSE-COS。

通过环境变量注入 `TENCENTCLOUD_SECRET_ID`、`TENCENTCLOUD_SECRET_KEY`，
临时凭据另需 `TENCENTCLOUD_SESSION_TOKEN`。不要将密钥放到命令参数、Git、
环境锁文件或快照目录。程序用户只需项目目录的 PutObject/GetObject/HeadObject。
具体策略由运维配置保存，不在代码里固定账号或桶名。

```bash
python -m pip install -r tools/persistence/requirements-cos.txt
python tools/persistence/qpersist.py publish --backend cos \
  --cos-bucket BUCKET --cos-region REGION \
  --cos-prefix projects/PROJECT/snapshots/ \
  --package PACKAGE --receipt publish-receipt.json

python tools/persistence/qpersist.py restore --backend cos \
  --cos-bucket BUCKET --cos-region REGION \
  --cos-prefix projects/PROJECT/snapshots/ \
  --cos-receipt publish-receipt.json \
  --snapshot SNAPSHOT_ID --manifest-sha256 TRUSTED_SHA256 \
  --out NEW_RESTORE_DIRECTORY
```

`inventory`、`pack` 沿用 README.zh.md 的显式输入选择和关闭写入确认。
恢复时必须保存并提供发布回执：回执记录每个数据块和清单的 COS VersionId。
程序固定读取这些版本，不会在旧版本不存在时退回最新对象。
独立可信索引中的清单 SHA-256 用于核对内容；COS ETag 不充当 SHA-256。

数据块回读通过后才发布清单。失败不产生成功回执；同一快照重试会核对并复用
已存在的相同内容，拒绝不同内容。并发发布可能留下多个版本，每份成功回执固定
其自己的版本集合，不保证 exactly-once。不要并发写同一逻辑快照。

版本控制必须保持开启；返回缺失或 null VersionId 会失败，可能留下未提交对象。
禁止为受保护快照配置删除旧版本的生命周期规则；删除历史版本会使已有回执无法恢复。
程序用户无删除对象、修改桶版本控制或生命周期权限。运维核验桶配置和权限，
工作进程不为此获得桶管理权限。

本后端不自动刷新临时凭据、不调度备份、不快照正在写入的训练检查点。
下载/回读产生实际流量；历史版本占用容量。通过小型合成快照不等于真实训练续跑通过。
