# 已有输入包含实体对标记

本次完成短实验后，只读核对当前注册科学源，未执行新模型或训练。

- `baselines/retacred/rmea_confirmation_run.py` SHA256 `9f6de9013354dd90122b839a12e4a8cfc8b6946a2e2169e41fd248b472511e19`：extract第41–52行先调用marked_words，随后RoBERTa编码，对原词子词求均值，写入h。
- `scripts/audit_rmea_tokenization.py` SHA256 `5f7860e47a7a0979976e5acc435ed34acede4c96bfac799d79d3018366b6aef8`：marked_words将主语与宾语分别用`@`和`#`包围；标签、类型、metadata不进入此构造；标记本身不参与证据池化，但上下文化原词保留其影响。
- `baselines/retacred/rmea.py` SHA256 `9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0`：第55行量子态prepare接收h和四维roles；第64行values接收h；第65行关系特定权重作用于utilities。

因此，“values未直接拼接roles”不意味着“values从未收到实体对信息”。h已由带主宾标记的输入产生。它也不保证关系语义可被当前固定取值映射可靠区分。下一步需检验信息的可用性及其被读出利用的方式，避免将缺失输入当成未经验证的根因。本记录为新源码路径观察，不改写已发布的固定实验结果。
