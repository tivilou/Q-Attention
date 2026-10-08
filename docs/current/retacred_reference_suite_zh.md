# Re-TACRED 参考模型复现：统一入口

当前默认配置包含四个官方实体标记模型和普通 PTR，**不是十五种方法已全部复现**：
typed_marker_punct、typed_marker、entity_marker_punct、entity_marker、ptr。
完整候选及阻塞原因在 `baselines/retacred/cohort.json`。普通 PTR 保留官方五项 prompt
CE 加关系 CE、双 optimizer/scheduler 和 FP32，已通过小模型 CPU 行为门槛；
完整权重 CUDA 门槛仍待完成。PTR reversed 是部分关系反向后的**变更任务设置**，
作者明确说明不应与常规设置直接比较；论文91.9保留为参考，不列为本任务同协议 SOTA。
原始完整反转列表、逆关系 verbalizer 与转换入口仍未恢复，不用普通 PTR 冒充。
Sequence Routing 的 `Abstraction.py` 已从官方历史恢复，H3/Mix 的输出维度512已确认，
但分类器41类与 canonical40类的边界、relation 文件的生成前读取问题、论文64batch/6epoch
与源码24batch/5epoch的差异仍待厘清。公开源码的全标签 micro-F1包含 no_relation，
等于 accuracy，并非本项目非空关系 micro-F1；论文92.2的评估绑定尚未认证，不作为同口径排名。
CTL-DRP 官方仓库仅 README；MetaEntail-RE 论文源码
链接返回 404；LUKE 对照变体仍缺可核实的完整配置。
+Curriculum 论文的91.4来自实体编码＋依存图 GAT＋双线性分类器＋跨模型课程学习，
+其自身 baseline 为91.2，增益为0.2个百分点。reviewer分别只用自身独立子集训练，
+不是常规 OOF 或按单模型 loss 排序。候选源码尚未绑定作者身份与该论文，GAT细节、
+reviewer数量、课程调度和评估器仍未核实；不使用普通 marker 冒充。
+审计契约见 `baselines/retacred/curriculum-contract.md`，离线审计
+`python -m baselines.retacred.curriculum_audit`预期 exit2 / not_verifiable / 零训练任务。

LUKE/CFIE 的作者仓库已找到并锁定：默认分支为NER，关系抽取在RE分支。
+该源码使用NYT24上的GCN/BERT，尚未绑定CoRE论文中的LUKE/Re-TACRED移植配置。
+CoRE发布脚本是TACRED checkpoint＋按gold标签筛选挑战子集＋macro-F1，
+与论文全量Re-TACRED micro-F1不是同一运行路径。论文同时报告macro和micro，
+聚合方式是五次训练的中位数，不是单seed或均值。四个LUKE候选继续阻塞，
+默认五个任务不变；审计见 `baselines/retacred/luke-variants-contract.md`。
+
+## 默认运行指令

完成下方同步、环境与 token 配置后，在项目根目录执行：

```bash
bash scripts/run_retacred_reference_suite.sh --gpu auto --checkpointing
```

代码由 `main` 提供，实验在师弟的 `1.1` 执行。当前工作仅在开发分支，**尚未发布**，
不要在拉取到正式发布提交前启动全量实验。项目侧只做 bounded 测试，不跑全量训练。

## 同步与环境

```bash
git fetch origin --prune
git switch 1.1
git merge --ff-only origin/1.1
git merge origin/main
git status --short --branch
# 激活已有的兼容 CUDA/PyTorch conda 环境，然后安装轻量依赖；不重装 CUDA/PyTorch。
python -m pip install -r baselines/retacred/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
bash scripts/check_retacred_reference_suite.sh --gpu auto
```

canonical 数据保持 `data/relation/retacred/{train,valid,test}.jsonl`，包含 tokens、
end-exclusive 两实体 span、label 和 metadata 中的实体类型。预检查验证 40 类和
58465/19584/13418 三 split。数据不从公开上游样例替换，不上传到 GitHub。

18084 环境变量沿用项目配置，token 不写入指令文件或仓库：

```bash
export PROJECT_EXCHANGE_URL="https://117.50.198.37:18084"
export PROJECT_EXCHANGE_TOKEN="项目方私下提供的 token"
```

程序自动下载该服务中已锁定的 RoBERTa-large 预训练文件，校验每个分片及合并文件
SHA-256。首次约 1.4 GB，缓存后复用。项目附带 CA，绝不跳过 HTTPS 证书验证。

## 科学口径

保持官方 marker、编码器、实体位置读出、MLP 和 loss。来源固定到官方 commit。
每个配置先执行真实权重的固定输入 forward/loss/gradient/AdamW 更新与 trace
透明性检查；通过后才训练。训练采用单独声明的共有预算（5 epoch、logical batch64、
micro batch4），而不是宣称原论文训练循环完全等价。选择 final epoch，不用 test
选模型。benchmark micro P/R/F1 排除 no_relation；同时记录全标签 accuracy、
micro、macro、逐类和混淆矩阵及 loss。论文报告值不填成实际复现结果。

完整权重的 fidelity 检查至少要求 **24 GiB 可用系统内存**，不是显存；同时保留两份
模型、梯度和 Adam 状态的需求远高于单份推理。低于此值会
在模型加载前拒绝，避免 OOM 影响其他服务。项目服务器目前只能做小模型 CPU 机制
和行为测试，尚未证明完整 RoBERTa/GPU gate。Attention 明确固定 eager 实现，以
记录实际进入 softmax 的 attention scores 和输出概率，而非事后重建。

`baselines/retacred/manifests/` 是逐模型来源与验证清单；机制和允许改动见同目录
`mechanism-contract.md`、`adaptation-diff.md`。目前整体证书仍是 `not_verifiable`，
不能把小模型测试或 CPU 固定输入检查当成完整权重 CUDA/AMP 认证。本批代码
尚不满足正式交付门槛；发布前须补齐相关门槛证据。本协议训练预算也不同于
此前冻结编码器的核心 baseline，不能据此直接宣称某注意力机制的因果优势。

PTR 的原始优化细节不强行统一成 marker 的参数：encoder LR3e-5、extra LR1e-5、
双 scheduler warmup500、encoder decay.01（bias/LayerNorm 除外），FP32。其 flags
为0的 extra-token 分支保持原样，不为了让其“起作用”擅自激活。loss 记录包含
五项 prompt CE 的均值**加**关系 CE。按标签名称对齐指标，PTR 数字标签顺序不同。
PTR 记录五个 mask 的真实 hidden、verbalizer 词向量投影、各项 logits、关系输出和
梯度，并在 train/test 中分别保存完整数字张量。双优化器及 scheduler 全部进入续跑文件。

`--checkpointing` 开启激活重算，适合 12GB 显卡；显存充足可在**新运行开始前**省略。
OOM 保留 checkpoint 并报错，不偷偷改变物理 batch/dropout 分区。下一次运行如改变
micro batch 或 checkpointing 必须创建新 run，不能把它称为严格同轨迹续跑。
此次不复制量子 selector 的 chunk 机制，Transformer 不属于相同 chunkable 路径。

## 进度、暂停与恢复

父进程每十秒打印任务总进度、各 GPU 任务和 batch/s；子日志单独保存在
`runs/retacred_reference_suite/<UTC时间戳>/logs/`。每 GPU 默认一个重任务，先空闲的
GPU 从队列取下一个 `(model,seed)`。每十个 update 保存 batch checkpoint，阶段尾
和 Ctrl-C 安全暂停必保存。恢复包含模型、optimizer、scheduler、scaler、RNG、
epoch permutation 和 next-update cursor。单 GPU 与多 GPU 切换仅改变队列设备：

```bash
bash scripts/run_retacred_reference_suite.sh --gpu auto --checkpointing \
  --resume runs/retacred_reference_suite/<原UTC时间戳>
```

同一组 models/seeds、batch 和 checkpointing 配置须保持。完成的任务验证后跳过。
每次恢复最坏回放最后十个尚未持久化的更新，不从 epoch 头跳 batch。

## 完整 case study 和一键导出推送

真实训练样本记录全部叶模块输入输出及 embedding、各层状态、Q/K/V、attention、
分类结果、反向激活梯度和分类器参数梯度；测试样本单独记录实际最终前向。JSON
包含原句、tokens、实体及位置、正确/预测关系和张量映射。完整 numeric `.pt` 与
私人 trace JSON 自动上传 18084；GitHub 只上传带哈希的紧凑引用和完整指标报告。
必需张量上传失败时，流程不宣称交付完成，保留 pending manifest。

所有训练完成后自动 export、只暂存白名单报告、commit，并 push `1.1`。
导出或网络推送失败**不用重训**，用同一配置：

```bash
bash scripts/run_retacred_reference_suite.sh --checkpointing \
  --resume runs/retacred_reference_suite/<原UTC时间戳> --export-only
```

`--no-push` 只禁用最后 push，不关闭证据上传或报告 commit。检查最终状态：

```bash
git status --short --branch
git diff --cached --name-only
git log --oneline origin/1.1..HEAD
```

如果仍 ahead，先检查网络/权限；不要 force-push，不要 `git add .`，不要推送
`runs/`、原始预测、数据、权重或 token。RUN_COMPLETE 仅代表训练任务完成，
`publication.json` 的 pushed 才代表结果上传 GitHub 成功。

只读检查计划：`bash scripts/run_retacred_reference_suite.sh --dry-run`。
离线审计 Sequence Routing：`python -m baselines.retacred.sequence_audit`，预期打印
`not_verifiable` 并退出2，不启动模型；它不是训练失败或“已经复现”。单独恢复历史文件用
`python -m baselines.retacred.sequence_resources`，验证缓存加 `--verify-only`，不会覆盖官方源码。
只读查看所有候选与缺口：`bash scripts/check_retacred_reference_suite.sh --inventory`。
`--models all --dry-run` 会指出缺源码/未接入方法并退出，不会把不完整 cohort 算全完成。

PTR reversed 离线审计：`python -m baselines.retacred.ptr_reversed_audit`，预期退出2、
`not_verifiable`、零任务启动。它不是训练脚本；证据与边界见
`baselines/retacred/ptr-reversed-contract.md`。若未来复现变更任务，应单独建数据协议与结果表。

## LUKE 基础适配器进度（2026-10-08，尚未加入正式任务队列）

新增 `baselines/retacred/luke.py`，直接执行锁定的作者解析器和分类器。
保留两实体最终表示拼接、实体感知注意力、带偏置40类头与交叉熵；
实体位置包含两侧标记，并按 batch 的最大实体长度补 -1，不截成30。
已锁定作者源码、预训练 luke-large 的 revision 与八个资源的身份。
这一步完成时尚未下载2.246GB权重；后续资源镜像已经完成。
主干/梯度/更新/恢复/数值张量记录/AMP门槛尚待验证。
这里没有新增 selector，也不是 CoRE/CFIE 等变体的复现。

当前仍运行默认5个配置。LUKE 四个变体继续在启动前拦截；完整单seed
仍交给合作者，暂不追加任何 LUKE 正式实验命令。

## LUKE 资源镜像（2026-10-08）

已准备锁定版本的预训练 luke-large 权重与分词资源，共8个文件、12个
不超过512MiB的分片。原始文件、上传回执、下载分片及重组文件均校验
尺寸和哈希。新镜像独立于现有 RoBERTa 资源，不修改默认任务队列。

将来发布这项资源获取支持后，在项目根目录执行：

```bash
bash scripts/fetch_retacred_luke_assets.sh
```

缺失资源时，从环境读取 `PROJECT_EXCHANGE_URL` 与
`PROJECT_EXCHANGE_TOKEN`，使用项目 CA 验证 HTTPS；两者均不写入 Git。
已校验的完整缓存可直接复用，无须令牌。项目内默认缓存为
`.external/assets/luke/<锁定revision>/model/`，可用 `--cache-dir` 指定
缓存根目录。脚本从 `PYTHON_BIN` 或当前 `python`/`python3` 解析环境，
不依赖我们服务器上的绝对路径。它只获取资源，不启动训练。

这里只证明资源传输完整；真实主干、梯度、优化器、断点恢复、数值
case study 与 CUDA/AMP 仍需独立验证。源代码的准备期 manifest 不会
因资源镜像完成而被升级为模型保真证书。代码回滚保留镜像，服务端
对象保持不可覆盖/不可删除的既有交换协议。

## LUKE 真实分词接口验证（独立于模型验证）

使用真实本地 `LukeTokenizer` 与锁定的作者实体位置方法，验证原始
token到BPE子词、token id、两实体位置及batch补齐的链路。只读取7个
分词资源文件，不读取模型权重、不导入Torch、不启动训练或网络请求。
在独立进程中执行：

```bash
python -m baselines.retacred.luke_tokenizer_fidelity \
  --output .external/retacred/luke-tokenizer-evidence.json
```

可用 `--source-cache` 与 `--checkpoint` 显式指定锁定源码缓存和本地
资源目录。输出记录合成样本的完整句子、BPE token、token id、实体
位置和各项校验结果；已有证据文件会保留，重跑时选择新的输出路径。
覆盖实体前后顺序、相邻实体、标点与Unicode、子词展开、超过30个
位置的实体、512/513-token边界、batch补齐和标签不进入模型输入。

此处直接执行作者的实体位置方法，但其AllenNLP字段载体用列表替代，
尚未验证实际AllenNLP分词包装器、indexer和张量化路径。结果只能标为
分词/接口证据，不能升级完整模型保真等级或解锁LUKE四个正式变体。
使用现有环境，不安装AllenNLP或修改容器内存限制。
