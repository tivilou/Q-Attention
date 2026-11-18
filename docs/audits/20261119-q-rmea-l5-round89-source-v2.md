# Q-RMEA R89：公共报告路径问题的修订

第一版 exact BOTH 审核返回 BLOCK：public compact-report 的 parent_child.command 会包含私有检查点等启动路径。首次审核、源码、控制、契约及失败测试的精确字节保留；该版没有获得执行资格，也没有运行诊断。

修订将完整 child argv 存到私有 scientific-work/child-command.json，公共报告只包含该记录哈希、同步/退出状态与日志哈希。只读验收器在私有环境核对 argv，不把其内容回传到公共报告。同时增加条件专属的固定 tensor inventory，逐个 NPZ 对照，避免仅由已产生字段的交集定义“完整”。

新的源码、运行契约、失败测试和验收器需要按新哈希重新审核。本修订记录不声称重审或科学运行已经通过。四模型、972/866训练来源留出、34/136案例、单次CPU诊断、不训练/不读COS的科学边界不变，L5 NOT_ACCEPTED。

- [修订设计](../evidence/rmea/20261119-l5-round89/training-case-design-v2.json)
