# 词框架候选的已知近邻与当前边界

已读Barnum/Knill quant-ph/0004088v1标题及pp2–3相关段落：p3式12直接给出PGM算子rho_out^-1/2 rho_j rho_out^-1/2。我们的wordPOVM是这一已知构造，EPS正则和按词求和不能当作新理论。其状态区分近最优界并不证明带独立signed utilities及40类argmax的Macro-F1最优。

已读Murray/Perronnin《Generalized Max Pooling》CVPR2014 pp1、3、4。它针对频繁描述子淹没稀有描述子的聚合问题，p3式4用同图像未中心化相关矩阵的伪逆，式6加入ridge，p4式10给出kernel逆形式权重。不能把“使用当前样本的Gram矩阵抑制频繁特征”当新原理。GMP给出的是聚合向量/可有符号权重；当前框架给出固定classquery下的非负词POVM权重。该差别还不是实质创新清关。

Crossref找到ICPR2022《Whitening Transformation inspired Self-Attention for Powerline Element Detection》，DOI10.1109/ICPR56361.2022.9956307。目前仅元数据；普通HTTP与普通headless Edge均为IEEE418，没有读到方法，不能声称与它区别成立。自己创建的headless会话已关闭；没有挑战绕过。其他Crossref不相关命中不作为证据。

附2维已知线性代数示例：两个正交词key、等幅query，重复第一个词后，原生Born逐词归一化把该模式总质量变成2/3；框架权重约保持两模式各1/2。原生固定key的两词odds不依赖其他词，框架一般依赖。实数向量可精确重现这个例子，所以它不是量子独占、语义解释或任务效益证明；同算法real12框架对照不可省略。

当前八路固定筛选仍在训练，不因上述已知性质预先宣布有效。新L3/L4-T未通过。即使筛选通过，也只支持设计完整比较，不自动通过L3或T。
