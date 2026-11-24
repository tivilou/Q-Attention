# 文献与本轮理论边界

Tsai等，Transformer Dissection: A Unified Understanding of Transformer’s Attention via the Lens of Kernel，arXiv:1908.11775v4，11 Nov2019。实际获取官方PDF249008字节/10页，SHA256 37665e8038b6e6229c8d673bdca5e9afc5c0094d8368f190520fb888f6d70eda；读取第1–3页、2.2节Definition1归一化核加权。原PDF/全文/receipt本机保留，不将论文全文上传Git。

该定义把注意力写成非负核的归一化加权和，并允许广义非对称核。它不证明本轮词先验有任务优势，也不支持量子专属性或新理论。RBF展开和辅助位条件概率是已知数学，属于本轮机制校验，不是L4-T创新。未声称完成全文综述或完整查新。

本轮可区分预测：若新输入词先验有Q独立效用，Qprior须在固定预算Macro上同时超过原Q、Dprior、两个经典prior、原real/soft与RBF及RBF-noenergy。RBF相对noenergy的差只涉及共享key energy，但学习后的差异不能当作冻结模型的因果归因；所有筛选结果均为已暴露TRAIN，不是独立确认。
