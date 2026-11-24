# 数相干光学实验：中断后的续训绑定

当前状态：**已准备、离线绑定检查通过、Astra6.2/high生命周期复审通过；未部署、未续训。** 本记录不接受新L3/L4-T，也不改变原限定L4、科学代码、四个条件、seed13、9140更新/条件或十三个任务门。

原双2080节点重新连接后，resident616736已不在，一张GPU UUID已更换，因此旧分配/permit不适用于当前两卡。原数据和结果保留。四个本地worker的最新完整状态分别为9140、9140、485、457，总19222；进度面板旧值19194已停止。COS050的18280状态是已经核对request/ACK、sealed文件哈希、固定版本HEAD和远端索引的回退点；这次核对没有重新下载模型或大型缓存。

唯一代码差异是两个派生控制绑定：主控从fresh-only改为当前父实验的明确local `--resume`，要求新attempt/代次、固定父launch/source以及精确argv/输出路径；发布器采用同一新控制目录、拒绝旧attempt/代次和另建index lineage。原host、publisher、日志和训练源码保留。逻辑run_id、COS对象前缀和已认证000..050历史不变；完整终态仍交付原七文件及既有delivery/resource-return记录。

外部v1指出输出路径和发布器代次两个实质问题；两项修正后v2为PASS。检查了正常续训绑定、18种错误绑定拒绝及派生副本回滚；三条命令均exit0。Windows夹具使用POSIX路径语义，未把路径夹具失败归于Linux运行时。

可复核文件：

- 派生代码：[主控](modified/full_host_run.py)、[发布器](modified/allocation_publisher.py)。
- [差异](binding.diff)、[命令/原样输出/退出状态](verification.json)、[复审](review-v2.json)、[哈希接受记录](review-acceptance.json)。
- [可运行回滚](rollback.py)将`before/`的原字节恢复到派生目录；已在独立`rollback-check`目录执行，不接触服务器。

启动前仍缺：用户已被询问的关机/资源调整事实、关联旧writer终止/隔离与资源归还、最新全fleet观察及实际两卡新分配；随后在新attempt中独立复制关闭的本地worker/身份/历史文件，只对不可变缓存复用硬链接，并通过既有core核对state/config/source/cache/completed-worker。实际故障的有界恢复检查也尚未执行。不得直接运行旧start.py、用旧permit附着publisher、重训已完成条件、删除父目录，或把本次离线复审称为恢复成功。
