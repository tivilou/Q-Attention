# Round8 独立静态审核

**PASS_WITH_RISK；must_fix=[]。** 结论只允许这份冻结的 forward-free 描述性审计与其调度生命周期，不表示已运行成功、接受新理论或达到L5。

| 身份 | SHA256 |
|---|---|
| source_sha256 | `fffbaa3b99432243940cd9ff7a21bf7d05328e541b6f9e83578a53110ad5c454` |
| contract_sha256 | `59ff29de88b727bf1b6d992c4857631b063d478b336909433e20eb776875ff03` |
| freeze_sha256 | `f6c5a25ca38641425c5c4957d418a081cc6d98e145999590ff01488f8d524c40` |
| input_packet_sha256 | `635a8d166d71351e4b23e8936a9760b31d558c352ac7cdb9e0ffc30f053c9682` |

9份冻结文件、1份本地运维依赖、6份原报告依赖哈希全部匹配；packet中的9份完整文本逐字一致。7个Python模块及6段内嵌脚本AST通过，实际recipe与contract的资源、命令、900秒、单attempt及5份输出完全一致。源码身份使用仅 source/* 的紧凑排序JSON；完整 raw freeze 另行绑定全部helpers与依赖。

指标与原runner等价：同float64、40类全量Macro/CM、相同CE运算；加法操作数顺序变化没有引入新的结合次序。对三份原指标字典完全相等是运行时硬门，本审核没有读取私有logits或声称已通过该门。配对CE是a-b，负数支持a；margin是正确类减其余最大logit，正差支持a。四格正确性与预测改变分别记录，错误类别互换不会冒充纠错。

覆盖1024例和所有40类；主体/客体中点方向、绝对间隔32及合并训练支持16均是预定边际分层，没有结果筛样或交叉搜索。零支持类和空分层保留。角色列2/3、偏移、标签及ID顺序来自冻结原cache；313条补充训练targets由固定overlay闭包校验。原数组、IDs、特征和权重不输出。

两个核心比较明确分开：同所选模型关闭lambda项，以及该所选输出相对于另行训练受限control。内部phase_free只关闭lambda-commutator；Born与复态制备仍可带相位。已补入最终literature-synthesis的措辞限定，未改科学代码。旧L4不继承到本审计，新颖性或优化加速不由已知先例推出。

部署检查完整review/freeze，并逐项绑定owner与worker文件及Git blob。共享调度沿用停止、leader锁、双配置字节CAS、同锁补偿、未知状态保留、终态后精确恢复；wait超时不触发新attempt。collector必须验单个complete/exit0/quiescent身份与完整闭包；无私有输出，仅4份JSON加标记。逆向补丁恢复要求clean精确HEAD并核parent完整树，禁止覆盖其他工作。

风险：这是反复暴露开发集上的描述性观察，margin受尺度影响，CE/accuracy/Macro可能异向；小分层和空分层不是独立确认。执行结果、真实节点状态、私有输入存在、精确指标相等、闭包与binding恢复仍待主代理按门验收。不得把一次有利分层或指标改述为总体优越性、相位必要性、新理论或L5。

本审核仅本机读取、hash及AST解析；没有导入Torch、执行科学runner、连接远端、访问COS、修改受审文件或再委派。
