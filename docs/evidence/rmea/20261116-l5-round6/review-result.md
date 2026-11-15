# Round6 机制与生命周期精确审查

**PASS_WITH_RISK，must_fix=[]，scope=both。** 仅允许当前冻结包的一次有界CPU单元预检及四条件开发筛查，不接受任务效用、新颖性、正式实验准备、旧L4继承或L5。

实体projector池化、Born索引、commutator正号及disk参数化一致。独立partial-swap参考的两个方向混合恰对应实现的`+i lambda[A,B]/2`；trace1 PSD效应是各关系独立二元测量，不是40类联合POVM。global路径与原pinned实现一致，unit包含同初始化原模型的完整前向及参数梯度对照。单元测试未在本轮执行。

四条件共享准备/value/decoder/bias初始化和批次顺序，固定seed307、4epochs、2361train/1024valid。数据是已暴露开发集，验证集按预定Macro选epoch。mixture的tanh与fusion的disk映射、参数数目及优化几何不同，因此任何差值都不足以单独归因phase；固定selected模型的lambda=0仅为局部消融。阳性也只保留探索线索，阴性只是固定预算HOLD。

初稿确有审核继承漏洞：source身份本轮仅覆盖source/，旧gate未把helpers的freeze绑定到review。已修正为deploy和queue激活均校验`review.freeze_sha256 == SHA256(raw freeze.json)`。新旧helper仅增加这两行；静态内存篡改模型确认改helper-map会使该gate失败。旧包保存在pre-review-v1，科学源码身份未变。当前12文件逐项与packet/磁盘/冻结hash一致，原3模块已用round3精确副本核对。

调度限定CPU1/RAM1536MiB/GPU0/1800s、单attempt。paired CAS保留其他项目，未知状态不恢复/不重试，终态静止后恢复原配置字节。collector仅下载紧凑JSON及日志，14文件私有闭包在worker核验；精确逆补丁回滚核parent tree。执行、资源适配、单元及闭包仍需实际观察，任何失败不得自动扩预算。

## 精确身份

- source SHA256: `8458d724c09deeb151bb60b8e4f6f65d4c493d7a6eb7c43022dbde13c84cf7b5`
- contract SHA256: `f4d32b7b0b4b2a837e9da1c9cd9a592c50444a82c296b882c4ab9a76f47901fa`
- freeze SHA256: `2c405bc76727bbd8018dfd555b7f77e282932e7fb5d646ac47ce5ed1b1397fa1`
- packet SHA256: `43d1b83bf5a0162f47da73988854fc74da0c985b5383ef6e01402a5478d0006e`

本轮source身份规则是source/路径到hash映射的compact sorted JSON；完整freeze raw SHA另外绑定helpers及依赖，未沿用此前轮次的聚合规则。完整细节、逐文件哈希、风险及允许动作见review-result.json。审查只解析与读取本地文件，未导入Torch或候选、未跑科学代码、未访问远程/COS，也未改源码。
