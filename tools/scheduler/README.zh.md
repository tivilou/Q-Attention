# Research 共享调度

所有 research 项目使用同一控制端和 `researchctl` 命令。控制端运行在所有者电脑，队列 SQLite/WAL 保存在该电脑的私有持久目录；执行节点仅通过严格验证的 SSH RPC 接收任务，无开放公网接口。CPU/GPU 池共享一个资源账本，按项目轮转、项目内 FIFO，每项目最多一个活动任务。

```powershell
researchctl status
researchctl health
researchctl submit --project . --recipe infra-smoke --key my-unique-test --wait
```

命令需要先注册项目、代码完整 commit、可信 Python argv 配方、超时、CPU/RAM/磁盘/GPU 请求、输出白名单和有限重试策略。配方和项目政策在所有者控制端与节点两侧匹配；项目的 serverctl development 授权在派发和查询时重新检查。关联 research 不自动授予服务器或 COS 权限。无配方/代码身份时拒绝提交；无授权节点或资源不足时留在队列。资源账本管理本系统任务，不能限制外部用户或对抗恶意子进程。

## 生命周期

控制端先在事务中预留全部资源和 GPU 组，再创建独立 attempt/token，最后请求节点启动。节点先持久化启动身份，再启动独立监督进程；相同 attempt 请求只能查询，不能重复启动。监督进程在独立 POSIX 进程组运行真实 argv，CPU 任务屏蔽全部 GPU，GPU 任务仅暴露所分配 UUID。RAM 以系统可用与 cgroup 剩余额度的较小值准入，另留32MiB；节点启动时再次检查资源。CPU/RAM/磁盘请求为可信预算和准入规则，不是内核硬限额。

`queued -> launching -> running -> complete/failed/timed_out/cancelled`；无法确认的执行变为 `unknown`，继续占用资源。控制端重启后先核对旧 attempt，不重新派发未知执行。独立监督进程须持续收到租约续期；失去续期会停止整个子进程组，确认进程组退出后才发终态证明。只有确认租约终止或节点重启、且配方明确 replay_safe、尚未超过1至3次上限，才可重新排队。科学训练续跑不由此实现；未认证检查点恢复的任务不能藉重试重新训练。

输出必须全部存在、是输出目录内常规文件，并带字节数和 SHA256。源码 commit 和干净状态必须保持不变，进程退出码0本身不算成功。旧 token、错误节点、缺失输出或不可信终态保留占用并停止推进。排队任务可取消；运行取消先请求节点停止，再凭终止证明释放资源。

```powershell
researchctl cancel TASK_ID
researchctl backup --output NEW_PRIVATE_BACKUP_PATH
```

备份使用 SQLite backup API 并检查完整性。此阶段队列在所有者电脑持久保存，尚未认证 COS 控制面恢复或24小时控制主机。实验输出的 COS 持久化继续由既有、显式调用的归档流程负责；调度任务完成不等于已归档。没有云 CPU 自动扩容、GPU 购买/释放或远端项目代码部署。

## 运维边界

运行控制端：`researchctl controller`，停止：`researchctl stop`。单控制端文件锁防止重复领导；停止后未结束任务会按节点租约到期停止，恢复控制端后核对状态。不要手改数据库释放 unknown，不要删启动记录或强行重排；先检查原节点。所有者电脑睡眠或离线时不会继续派发。

Q-Attention 仍由我们完成 idea、代码和有界测试，通过 GitHub 把正式实验交给合作者。其现有正式/真实数据筛选脚本不因此迁入所有者队列。本阶段只注册基础设施合成诊断配方。未授权项目可以查看共享队列，但不能借其他项目的节点执行。

验证范围记录在 validation.json。GPU 分配的单元测试可使用模拟遥测；没有物理 GPU canary 时必须保留硬件验证缺口。日志、队列 DB、节点配置、凭据和任务产物均不进入公共 Git。

## 服务器健康探测

控制端配置 `health.enabled=true` 后，后台约每60秒调用 serverctl 的全清单维护探测，最多并发3条连接。探测与活动任务的续租分开执行；慢连接不阻塞租约。只读检查已登记端口、严格 SSH 认证、CPU/内存/临时盘和 GPU 遥测，不启动实验，不改远端环境。已回收或认证未配置的记录跳过；无现有具体项目绑定的记录不进行 SSH 资源查询。

```powershell
researchctl health
researchctl health --refresh
```

默认读取私有缓存；显式刷新检查到期条目。底层 `serverctl health --refresh --fleet --force` 可立即重查全部条目，`--no-save` 只返回结果。缓存独立于 inventory 和项目绑定，记录检查时间、最近成功、失败原因、连续失败数、下次检查时间及资源；不存密钥或原始 SSH 日志。一次失败即阻止新派发，连续3次失败标为 offline，重试间隔从60秒逐步退避至15分钟。默认超过180秒的成功信息失效；修改端点、信任或绑定后旧缓存失效。

`researchctl status` 同时显示全清单健康、已接入节点的 worker RPC 状态与资源。健康探测不会自动接入机器、增加项目权限或调整配方/容量。OS 可见 CPU/内存不是容器配额，GPU 有设备也不代表空闲；派发仍须满足已注册节点、项目授权、队列预留与 worker 的即时资源检查。健康失败只影响新任务准入；旧任务继续独立续租/核对，失联保留 unknown 和资源预留。
