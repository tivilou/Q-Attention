# 从均值读出到 value 条件下的 Born 重分配：暂定机制

状态：探索假设，L5 NOT_ACCEPTED；尚未实现或执行。不是把现有L4重新评级。理论工具的首创性未建立，下面的实验预测也尚未验证。

## 依据与因果变化

已执行的全量seed13中，linear的test Macro39为.4628824678、native为.4539221915。固定linear模型的句子旋转匹配诊断及短程梯度门槛未支持强归因。这些只是选择新问题的依据，不是新机制效果。R84的路由/readout交叉回放来自另一个较小cohort，不能与这个全量模型混作同一机制证明。

独立Astra6.2high咨询（未收到下面候选）建议检查平均value之外的分歧，明确指出加权方差池化是已知工具、仅宜诊断。当前改动的因果链是：由同一relation的value效用形成一个量子可观测量，再反馈到relation的测量状态，最后仍通过Born注意力与原value平均产生logit。没有并列的经典残差主分支。

## 精确接口

固定一句的纯词态psi_t、一个关系态phi（范数1）及实效用u_t。令eps=1e-6，F_t=|psi_t><psi_t|+eps I_8，C=sum_t F_t>0，A=sum_t u_t F_t，Z=phi† C phi。

原注意力p_t=phi†F_t phi/Z，原无bias分数m=sum_t p_t u_t=phi†A phi/Z。

令 E_t=C^(-1/2) F_t C^(-1/2)，z=C^(1/2)phi/sqrt(Z)，O=C^(-1/2) A C^(-1/2)。有sum_t E_t=I，||z||=1，p_t=z†E_tz，m=z†Oz。这是每个relation独立的token-POVM表示，不是把40个relation称为一个联合POVM。frame归一化是已知工具，不作为新颖性。

对linear checkpoint，phi必须先变为该句的已验证8维query旋转作用后的phi_eff；不把原始phi与输运后的attention混用。eps加在每个F_t内，不能只在分母临时加eps。

## 要检验的重分配机制

定义一个value反馈滤波

    z_eta = exp(eta O/2) z / ||exp(eta O/2) z||,
    p_t(eta) = z_eta† E_t z_eta,
    m(eta) = sum_t p_t(eta)u_t.

这是正、归一化的Born路由。可将phi_eta正比于C^(-1/2)z_eta，返回原score接口；分数m(eta)+bias，没有加性variance残差主方法。此滤波一般不是unitary；任何真实量子实现需说明滤波/制备及成功率，不作硬件加速或免费state-preparation声称。eta=0精确回到原方法。

标准导数为 dm(eta)/deta = Var_{z_eta}(O)>=0。每个类别分数都增加并不保证分类CE变好，故这是可否证的任务假设，不是效果保证。经典matched反馈是p_t^C(eta)正比于p_t exp(eta u_t)，在eta0的导数为V_C=sum_t p_tu_t²-m²。

量子响应在eta0为

    V_Q = z†O²z-m² = (Aphi)† C^(-1)(Aphi)/Z-m².

不需要对64×64矩阵或所有token构造大矩阵；只有8维Hermitian solve。原routing训练参数不变。

## 分歧的精确区别

V_C-V_Q=z†N z，

    N=sum_t u_t² E_t-O²
     =sum_t (u_t I-O) E_t (u_t I-O) >=0.

故0<=V_Q<=V_C。这个正性是已知的POVM内禀噪声/算子方差关系，不能声称本项目首创。它表明经典结果方差含有一部分不进入量子observable响应的分歧；它没有证明这部分对语言任务是噪声或无用。

另一个精确预测：固定F_t时，若效用改变delta_u满足sum_t delta_u_t F_t=0，则A、O、m、V_Q乃至上述整个eta滤波均不变；V_C可改变。例如两个F完全相同的token分别得到u+delta和u-delta，原平均及量子方差不变，经典方差增加相应正量。真实语料是否有这种可忽略扰动尚未证明，不把合成可辨性当成真实任务证据。

## 最小任务诊断（主模型尚不实施）

冻结现有linear epoch4 checkpoint，不更新原参数，只用完整train58465条分别拟合一个共享signed scalar gamma：原logit+gamma*V_Q、原logit+gamma*V_C、原logit+gamma*V_U，其中V_U是同一句同一relation效用的均匀方差。gamma0包含原头。每个标量仅从train的40类CE拟合，不能从valid/test选符号、范围或条件。均值原模型为第四路。

固定validation19584条一次评估；原Macro39和CE必须复现旧报告。主门槛：Q路相对于原模型、C路、U路的配对CE差，其三个单侧98.333333% bootstrap上界都<0（Bonferroni家族alpha.05）；报告全部四路Macro39。若任一失败，不提名该统计量的独立任务优势，不换checkpoint、样本、bootstrap规则或gamma搜索策略。即使通过，这仍是方差读出诊断，不能声称有限eta的滤波已验证、量子优势、独立确认或L5。

校准的CE是gamma的凸函数；预定收敛/有界可达范围应在代码前固定并记录命中边界，不用valid选择。允许signed gamma便于诊断校准作用；若仅负gamma有效，不支持正eta反馈的任务机制。实际反馈fullrun只在正向相容证据后另冻结：原linear、量子filter、经典exponentialtilt共享训练预算，single seed先行；不能把已有反复看过的test当新独立确认。

## 已知碰撞与下一步

量子滤波/imaginary-time、frame-POVM、方差分解、凸校准与累计量导数都是已有工具。候选有待验证的区别是：value反馈通过有限维算子O响应，恰好忽略测量不可分辨的效用分歧；是否比经典结果方差/反馈更有任务价值。文献新颖性仍待限定查重，当前不是L5提名。先做数学与对照审查；通过后实现一次有界校准诊断，不增加正式训练或COS副本。


## 审查后冻结的诊断规则

数学/设计咨询返回PASS_WITH_SCOPE，但有两项必须补足；不把该返回当成实现放行。实现须逐样本逐关系验证eta0重建的概率与logit，atol1e-9；每split各GPU首batch另与原token-space路由核对2e-12；完整valid原CE和Macro复现atol1e-10。标量固定[-1,1]，训练CE导数单调root用brentq(xtol=rtol=1e-12,maxiter100)，边界/零点规则固定且记录命中边界。bootstrap固定10000次、seed20261122，三个同一重采样的Q-minus-comparator均值差上界使用分位数1-.05/3；推进正eta机制还必须gamma_Q>0。不得把负gamma的校准收益算作正反馈证据。实现另需精确源码both作用域审查通过且must_fix为空，再经既有调度器测试/执行。
