# Round5 定向勘误复核

**PASS，must_fix=[]。** 本轮只核对三份勘误相关公开文档及原冻结proof/runner/contract/numeric/summary。结论不是重跑验收或新的L5评定。

64维value池化表示的正确关系是固定可逆符号变换 `D=diag(-1,1,…,1)`：第一坐标变号，其余63维为0；预定decoder变号抵消该变换。保存的alpha机械核算得到A第一坐标约`[1,-1]`、B约`[-1,1]`，符号关系残差`7.771561172376096e-16`，低于原1e-12容差。注意力移植后交换的是两类活跃logits，不能据此把64维池化value称为坐标置换。1024维原h池化的首两坐标置换是另一个边界的陈述。前次审查没有明确指出这一边界歧义，本轮予以纠正。

公开proof现在有显著notice，notice之后的原正文与冻结proof逐字节相同。报告只增加一个勘误段落；删除它就精确恢复旧报告SHA256 `860d74756cb0749de134c5d4ced5d0bce51d6c79556354d71dffae66b0063d87`。五份冻结证据的SHA均与前轮一致。没有改源码、阈值或数值；own`[2,2]`、swap`[0,0]`、redecode`[2,2]`、CE增量2和最大误差`8.881784197001252e-16`均保持。共同适配反例的窄结论不变，L5未接受，原限定L4保持，新commutator线索没有得到本轮科学审核。

## 精确哈希

- `docs/audits/20261116-q-rmea-l5-round5.md`: `6821d7d7d2617093df9fe35eb0d4a4d1950baae5443985681d93cd6c292d88dd`
- `docs/evidence/rmea/20261116-l5-round5/proof.md`: `8b6b03c5a2b9faef2c9af5b8374c8863951aa8db665f17636d1147b4a1e6453e`
- `docs/evidence/rmea/20261116-l5-round5/proof-wording-erratum.md`: `08b5ded56bc3a8ab7ef1af7b1182f94a47700bb87150f7993dd6d88ad4ee3f24`
- `.codex-artifacts/qattn-l5-round5-20261116/proof.md`: `35238d9de433217fc64707a949d448814f71feeb052870bad4e0dd7331875122`
- `.codex-artifacts/qattn-l5-round5-20261116/source/runner.py`: `106180057542dd1aeb1e37962c1821e399ba5764e4ca582a917154bc706bc519`
- `.codex-artifacts/qattn-l5-round5-20261116/source/contract.json`: `36ec11aaa4de0b3d9fb740a2e657e9a1906e31785b8a543d3b9e7b709ce24280`
- `.codex-artifacts/qattn-l5-round5-20261116/numeric-fixture.json`: `b49e0a13a439b8e9683d9be41243126e2aecb07dd76acdf5eaf2648222e022f4`
- `.codex-artifacts/qattn-l5-round5-20261116/summary.json`: `976dcfb845fda4fb50ffdc88499523ea6fb1aa573dd9bc2a80e8b2eba2d07452`

聚合source：`08b3ba5183c92c28d45bc39bc300cd28dd63292e49fd921de586a12917a5875f`。本轮仅重哈希上述冻结文件，沿用先前精确source审查的聚合身份。所有八份输入在写审查件前后字节保持不变。
