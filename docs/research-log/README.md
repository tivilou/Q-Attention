# 研究日志

## 当前研究总览

最新检查点：[同前向训练梯度三卡回放](20261121T-date-only-backward-credit.md)。三路各256更新、完整validation联合门槛未过；冻结词图条件数值较高但区间跨零，不能据此选新方法。保留原L4，L5未达标。下一步冻结cotangent配对局部方向诊断。

## 时间线

- [2026-11-21：同前向训练梯度三卡回放](20261121T-date-only-backward-credit.md)（短256更新；三卡实际同时训练；不支持联合归因门槛）。

- [2026-11-21：查询/上下文匹配三卡诊断](20261121T-date-only-query-alignment.md)（三卡真实重叠；全量validation门槛未过，partial trace）。

- [2026-11-21：词质量/方向配对诊断，停止质量补偿](20261121T-date-only-mass-direction.md)（仅日期命名；三项短诊断完成，partial trace 边界明确）。

- [2026-11-21：三阶输运全量结果与线性路径线索](20261121T-034306Z-token-transport-full-seed13.md)（原始运行开始UTC已记录；不扩大历史证据等级）。

- [2026-10-27：Q-RMEA完整seed13结果](20261027T-075438Z-q-rmea-seed13.md)（运行启动UTC已记录；L1候选，待独立确认）。

- [2026-10-26：Q-RMEA 实现验收](20261026T-162616Z-rmea-implementation.md)（实际关闭 UTC 收据；技术验收，不是正式效用）。
- [2026-10-26：前置检查](20261026-date-only-rmea-readiness.md)（条目时间精度仅日期；完整机制 trace 缺失）。
- [历史条目 20261026T-022952Z-q-srm-bounded-screen.md](20261026T-022952Z-q-srm-bounded-screen.md)（时间及结论以原文为准）。
- [历史条目 20261020T-073818Z-q-cfra-toy.md](20261020T-073818Z-q-cfra-toy.md)（时间及结论以原文为准）。
