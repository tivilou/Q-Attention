# Quantum embeddings：源结果与当前任务的差别

原文：https://arxiv.org/pdf/2001.03622 。实际PDF首页题名Quantum embeddings for machine learning；作者Seth Lloyd、Maria Schuld、Aroosa Ijaz、Josh Izaac、Nathan Killoran。首页边栏标记arXiv:2001.03622v2 [quant-ph] 10 Feb2020，正文另标Dated: July3,2022；两个日期均保留，不推定更改历史。11页、4951839bytes、SHA256 c7c5d66f2b34210856c35900561a89d6744d76f5bdf3cfa0ce384f0019e3aebc。正式出版版本未核。身份、获取、解析凭证在fulltexts-v2/0e9215626571b763/receipt.json；首个请求因helper把说明性版本字符串当严格值而未发请求，失败保留于fulltexts，修正为any后单次官方PDF获取成功。

实际阅读：PDF页1摘要/引言、页2全部§I/§II开头、页3全部§II/§III、页4§III及§IV、页5图4和§IV收尾（§V只读开头，未作为论据）；另补读页7附录A、页8附录A/B、页9附录B收尾/C全部/D开头。页3另渲染核对正文公式与图2说明。未读页6正文/参考文献、页7参考文献、页10–11、完整附录D/后续附录、外部代码；解析完11页不表示读完。没有将原文全文/图片提交公开Git。

## 已核源事实

- 页2–3把二分类数据各自嵌入密度矩阵rho/sigma，比较f(x)=tr(|x><x|M)；所用论证明确选线性损失，分类器可重复测量，与单次最小误差态判别的设置区分。
- 页3：f_fid(x)=<x|(rho-sigma)|x>，其该线性经验风险等于负Hilbert–Schmidt距离；Helstrom读出用rho-sigma正负谱投影，相应风险联系迹距离。由此主张训练embedding来分离类别。这里不是对任意损失、任意下游decoder或任意attention汇聚的普遍最优性声明。
- 页3–4的可训练量子embedding与Hilbert–Schmidt/MMD联系已存在；不能把这些联系、Born重叠、类别距离最大化或可训练编码单独列为本项目新理论。
- 页5作者明确说演示旨在说明embedding可训练，并没有回答量子embedding是否优于经典方法；不把该演示转借成本项目公平经典优势证据。
- 页8附录B再次明确，多次运行后估计确定性期望时，单次最小误差判别不再等同于泛化目标；“最佳测量”依赖选定embedding和loss。页9附录C的fidelity classifier是对两个训练类集合的重叠差，仍不是本项目按词归一化、随后读signed utility的路由。附录D的完整成本/可实现性论证未核，不作为资源结论使用。

## 对当前方法的解释与下一步边界

当前Q-RMEA不是该文的二类线性风险分类器。它先逐关系/逐词取Born分数，再按词归一化、加权另一条带正负号的value/decoder路径，最后用40类CE训练。共享条件U改变的是关系态相对词态的方向；保持query之间Gram不等于按标签把样本密度分离。直接照搬Helstrom最优性或“最大化量子距离即改善当前Macro-F1”缺少假设对应。

因此后续若需要理论，必须分析当前归一化与signed value实际保留/丢失了什么，以及哪条可检验结论超出已知态判别/metric learning；不能换成更容易证明但没有在L3模型中实现的目标。本文提供一条有内容的近邻限制，没有使当前候选获得L3/T，也没有预先判定仍在运行的三路结果。
