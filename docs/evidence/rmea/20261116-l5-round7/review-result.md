# Round7 精确机制与生命周期审查

**PASS_WITH_RISK，must_fix=[]，scope=both。** 仅允许一个新控制：原FusionHead与disk映射不变，训练/评价始终phase_off=True，seed307/4epochs。旧四条件及其选择结果只读复用，没有重跑。

初始化公共参数哈希、完整输入ID与字节身份、NP batch顺序、优化器/预算/选模规则均与旧fusion匹配。g2=0时kappa对g2的一阶导为0，lambda路径关闭，零初始化与AdamW及每步断言维持g2和其梯度严格为0。40坐标仅在本次约束轨迹不活动；g2在完整函数的kappa分母仍出现，80628存储参数不意味着同等表达能力。此措辞已在最终contract和综述中明确，科学audit代码与helpers未改。

atanh(kappa)ordinary-mixture twin复制全部共享状态，函数对应正确；unit与完整1024例回放是数值硬门，非训练或优化等价。旧fusion−新控制的Macro正号最多保留探索训练干预线索，不能独立归因全部相位收益，更不产生L5/正式实验准备或旧L4继承。

完整freeze绑定deploy与queue，源/contract/依赖精确核验；paired-CAS、未知状态保留、单attempt和终态后配置恢复保持。收集只核worker私有闭包（7数据文件，4MiB内），本机只取compact；逆补丁回滚须恢复精确parent tree。此处均为静态审核，尚未运行Torch、unit、模型或远程动作。

## 精确身份

- source SHA256: `edb88c05d9d79f29fc70eef190e97de79d8259eb5d04bca846bdf7e103ddbc91`
- contract SHA256: `e1644bf3c79f17827515a681e59bb95ab41af84cf73f1518c1a8d6b1d89688e1`
- freeze SHA256: `4e04db90e8e817ab66777508f130dfafa46bdec3f119a9f89ae0e938deb166b0`
- packet SHA256: `69ee9e0bd46ce850d394bdbf9d0f160d36f43f28e4f32105693be58ab72388d1`

Source身份是source/*路径哈希的compact sorted JSON；完整raw freeze另绑定全部helpers与依赖。所有已读输入在落盘前后字节不变。具体约束、风险和允许动作见review-result.json。
