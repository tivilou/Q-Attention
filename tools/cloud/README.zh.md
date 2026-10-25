# 腾讯云单台 CPU 测试控制

这是有费用和时长限制的测试工具，当前支持单台 2 核、4 GiB、50 GiB 系统盘的按量实例。尚未实现跨项目队列、CPU 自动扩容或后台资源池。

在操作机安装 requirements.txt。专用程序身份使用 cam-policy.json 中列出的 CVM/VPC 操作；实际按量购买还需要 CVM 后付费权限。实测需附加官方 QcloudCVMFinanceAccess，其 finance:* 仅作用于 qcs::cvm:::*，无需账号级财务管理员权限。DryRun 通过也可能仍缺实际支付权限。CVM/VPC 操作资源范围为全部资源，实际操作范围由本机可信账本中的区域、唯一名称、实例 ID 和专用网络约束；这不是服务端资源隔离。身份不授予 CAM 或 COS 权限。密钥、SSH 私钥、cloud-init 和账本放在 Git 外的受保护目录；不要把程序密钥发到工作节点。

按顺序运行 cpu_control.py 的 init、quote、prepare、create、status。所有云操作都明确提供 --region、--credential、--ledger。先查机器库存和镜像，再提供经过 InquiryPriceRunInstances 核实的 spec。询价包含系统盘和网络规格；小时费用与按 GB 计费的出流量分别检查。默认上限为每小时 0.50 元、出流量每 GB 0.80 元、5 Mbps、最多 4 小时，仅允许创建一台。价格不是长期报价，每次创建前重新询价。

prepare_host.py 只在受保护目录生成专用 RSA2048 登录密钥、ed25519 主机密钥及 cloud-init。ImportKeyPair 使用 OpenSSH RSA、ProjectId=0，名称为至多 25 字符的字母数字下划线。通过经过认证的 API 配置主机公钥所对应的私钥后，先在 serverctl 登记并绑定项目角色，再固定已知主机公钥并严格 SSH；不要跳过主机验证。

create 在调用前持久化 ClientToken、请求和 UserData 的哈希，成功后核对定时销毁任务。定时任务查询可能短暂为空；只对空结果最多查询 10 次、间隔 2 秒，冲突任务立即拒绝。未能确认任务时保留实例 ID 并要求终止，不重新创建。再次执行 create 会拒绝。若响应不明确，先 reconcile 查询；显式 retry 会先查询已存在的实例，仅在没有匹配实例时重放完全相同的 token、请求、UserData 和原绝对销毁时间。它不延长时限或生成新实例标识。辅助资源创建不明确时必须先人工核对唯一名称和资源 ID。

安全组每次请求只处理一个方向；协议 ALL 不传 Port；API 返回协议名可能为小写。确认过的相同规则可重用，不同规则拒绝继续。现代 VPC API 的权限名称可能仍是旧接口名称：DescribeSecurityGroupPolicies 对应 cvm:DescribeSecurityGroupPolicys，CreateSecurityGroupPolicies 对应 cvm:CreateSecurityGroupPolicy。以实际拒绝信息和操作验证为准。

只有在节点上完成所需验证并把日志、结果和恢复凭据持久化后，才测试 stop、start、reboot。每次操作后用 status 等待期望状态。关机测试使用 KEEP_CHARGING，关机本身不会结束计费。最后 terminate，确认实例已终止或消失，再 cleanup。工具只删除账本中确认属于本次测试的网络、安全组和密钥；删除请求返回成功也要核对资源消失。

测试失败时保留不完整目录，不写成功凭据。恢复数据、代码与环境的操作由独立持久化工具执行。本工具不会启动训练、采集实时检查点或证明历史训练可以跨目录续跑。

运行 `python -m pytest -q tools/cloud/test_cpu_control.py` 检查费用、幂等重放、部分规则失败重试、资源归属和拒绝路径。Windows 桌面版可能将 AppData 重定向到其他卷，账本原子替换前必须解析实际路径。

官方接口：[RunInstances](https://cloud.tencent.com/document/api/213/15730)、[VPC 官方 Python SDK 源码](https://github.com/TencentCloud/tencentcloud-sdk-python/blob/master/tencentcloud/vpc/v20170312/vpc_client.py)、[cloud-init SSH 配置](https://docs.cloud-init.io/en/latest/reference/modules.html#ssh)。
