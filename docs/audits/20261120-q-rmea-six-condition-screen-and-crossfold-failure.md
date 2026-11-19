# 六条件测试与交叉分折检查：已获得正向线索，尚未晋级

六个匹配条件已通过共享调度器执行并审计：centered_full Macro-F1为0.06453672849105976，full为0.05901847394515695，simplex为0.059127704734673936。centered_full相对full约提升9.35%，但其CE为1.1955781302152042，劣于centered_native的1.171003775925422。预设Macro门槛通过、CE门槛失败，维持CLASS_CENTERED_SCREEN_INCONCLUSIVE_NO_PROMOTION；不追加seed、epoch或系数，不自动交付正式实验。

训练为固定原train前缀内部fit3124/evaluation972、一个共同初始化、四epoch、六条件共2352步，科学运行525.7695秒。原native/full/simplex全部指标与R96逐值一致，六个最终权重首batch重载完全相同。退出0、进程静默、精确源码及输出哈希均已核对；无官方valid/test、全量正式运行、COS请求或独立确认。新模型不继承旧限定L4，L5 NOT_ACCEPTED。

交叉分折八项检查的旧任务一次退出1且已静默，第三组的聚合平移等式出现4.376535511774193e-08误差，没有报告或完成标记。修复仅把平移mask在乘小数前显式转为u.dtype，并增加dtype元数据；保持原2e-9容差、八项检查、seed、机制和预算。单次gpt-6.1-sol/medium BOTH审查PASS_WITH_RISK、must_fix=0；审查确认精度解释与源码相容，但不能代替实际数值验证。修正版仍未执行，必须通过新的冻结合同单次准入；原失败与日志保留。

调度适配v003已用连续SQLite写排他消除idle检查到controller停止之间的claim竞态；六个实际SQLite竞态夹具及六个作用域夹具通过，其他项目队列完整保留。未改调度核心或node基础配置。本次训练和失败canary均只有一次attempt，禁止重启旧任务。

[六条件完整聚合结果](../evidence/rmea/20261120-l5-matched-screen/training-result.json) · [解释与研究优先级](../evidence/rmea/20261120-l5-matched-screen/screen-interpretation.md) · [失败记录](../evidence/rmea/20261120-l5-matched-screen/cross-fold-failure.json) · [修复审查](../evidence/rmea/20261120-l5-matched-screen/fixture-repair-review.json) · [准入记录](../evidence/rmea/20261120-l5-matched-screen/admission-record.json)
