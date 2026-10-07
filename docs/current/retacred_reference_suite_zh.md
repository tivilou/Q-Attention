# Re-TACRED 参考模型复现：统一入口

当前默认只执行已经接入的四个官方实体标记配置，**不是十五种方法已全部复现**：
typed_marker_punct、typed_marker、entity_marker_punct、entity_marker。
完整候选及阻塞原因在 `baselines/retacred/cohort.json`。PTR 的 verbalizer/40 类映射
已从官方历史提交恢复并校验，但五目标 prompt、双 optimizer 适配尚未完成；
不能把它算作已可运行。Sequence Routing
缺 `Abstraction.py` 和 Re-TACRED relation 文件，且源码配置 41 类而非本项目的 40 类。
CTL-DRP 官方仓库仅 README；MetaEntail-RE 论文源码
链接返回 404；Curriculum 与 LUKE 对照变体仍缺可核实的完整配置。不会静默替换。

## 默认运行指令

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
透明性检查；通过后才训练。训练采用单独声明的统一协议（5 epoch、logical batch64、
micro batch4），而不是宣称原论文训练循环完全等价。选择 final epoch，不用 test
选模型。benchmark micro P/R/F1 排除 no_relation；同时记录全标签 accuracy、
micro、macro、逐类和混淆矩阵及 loss。论文报告值不填成实际复现结果。

完整权重的 fidelity 检查至少要求 **12 GiB 可用系统内存**，不是显存；低于此值会
在模型加载前拒绝，避免 OOM 影响其他服务。项目服务器目前只能做小模型 CPU 机制
和行为测试，尚未证明完整 RoBERTa/GPU gate。Attention 明确固定 eager 实现，以
记录实际进入 softmax 的 attention scores 和输出概率，而非事后重建。

`baselines/retacred/manifests/` 是逐模型来源与验证清单；机制和允许改动见同目录
`mechanism-contract.md`、`adaptation-diff.md`。目前整体证书仍是 `not_verifiable`，
不能把小模型测试或 CPU 固定输入检查当成完整权重 CUDA/AMP 认证。本批代码
尚不满足正式交付门槛；发布前须补齐相关门槛证据。本协议训练预算也不同于
此前冻结编码器的核心 baseline，不能据此直接宣称某注意力机制的因果优势。

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
只读查看所有候选与缺口：`bash scripts/check_retacred_reference_suite.sh --inventory`。
`--models all --dry-run` 会指出缺源码/未接入方法并退出，不会把不完整 cohort 算全完成。
