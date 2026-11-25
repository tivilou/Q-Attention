from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

r = Path(__file__).resolve().parent


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(n):
    return json.loads((r / n).read_bytes())


def write(n, x):
    (r / n).write_bytes((json.dumps(x, ensure_ascii=False, indent=2) + '\n').encode())


now = datetime.now(timezone.utc).isoformat()
conference = read('fulltexts/8fcb56c452e49c5e/receipt.json')
arxiv = read('arxiv-fulltexts/61034008b4c371d1/receipt.json')
supplement = read('supplement-acquisition.json')
crossref = read('knyazev-crossref.json')['message']
authors = read('fastweights-arxiv-request.json')['papers'][0]['authors']
assert conference['sha256'] == '028b1b47b1d86d0e8e2340cbc07a290f9e477c0a5b1ed439437de6a4a1cecc63'
assert arxiv['sha256'] == '981110f0948e734528f0624230c0999cd6bb41a1f478d133bad9b6e1bc3d8503'
assert crossref['DOI'].lower() == '10.1137/s1064827500366124'
assert authors == ['Jimmy Ba', 'Geoffrey Hinton', 'Volodymyr Mnih', 'Joel Z. Leibo', 'Catalin Ionescu']

attempts = []
for n in ['fulltexts/03331f9f4bc03c3f/receipt.json',
          'author-copy-fulltexts/03331f9f4bc03c3f/receipt.json',
          'institutional-copy-fulltexts/03331f9f4bc03c3f/receipt.json',
          'repository-copy-fulltexts/03331f9f4bc03c3f/receipt.json',
          'institutional-www-copy-fulltexts/03331f9f4bc03c3f/receipt.json']:
    x = read(n)
    assert x['acquisition'] == 'unresolved'
    attempts.append(dict(receipt=n, acquisition=x['acquisition'], reading=x['reading'],
                         receipt_sha256=sha((r / n).read_bytes()), attempts=x['attempts']))

write('source-receipts.json', dict(
    status='PARTIAL_ORIGINAL_SOURCE_AUDIT', recorded_at=now,
    knyazev=dict(title=crossref['title'][0], author='Andrew V. Knyazev',
                 DOI=crossref['DOI'], journal=crossref['container-title'][0],
                 year=2001, volume='23', issue='2', pages='517-541',
                 identity='VERIFIED_OFFICIAL_METADATA', fulltext='UNRESOLVED',
                 reading='NOT_STARTED', algorithm_claims_from_this_paper=[], attempts=attempts),
    fastweights=dict(title=conference['title'], authors=authors,
                     conference=dict(receipt='fulltexts/8fcb56c452e49c5e/receipt.json',
                                     sha256=conference['sha256'], pages=9,
                                     version=conference['requested_version'],
                                     URL=conference['attempts'][0]['source']),
                     arxiv=dict(receipt='arxiv-fulltexts/61034008b4c371d1/receipt.json',
                                sha256=arxiv['sha256'], pages=11, version='1610.06258v1',
                                URL=arxiv['attempts'][0]['source']),
                     version_warning='Conference Eq1 A(t)=lambda A(t-1)+eta h(t)h(t)^T; v1 prints A(t+1)=lambda A(t)+eta h(t)h(t)^T. Versions retained separately; no whole-document identity claim.'),
    official_supplement=dict(URL=supplement['url'], archive_sha256=supplement['sha256'],
                             PDF_sha256=supplement['PDFs'][0]['sha256'], PDF_pages=23,
                             identity='MISMATCH_QUARANTINED',
                             actually_inspected=[1],
                             reason='First PDF page is an appendix on non-smooth finite-sum optimization with proximal oracles, Eq13 and Theorems9/10, rather than fast-weight model details. Other-work exact identity unresolved.',
                             scientific_use=False),
    collection=dict(ordinary_requests=18, public_browser_navigations=1,
                    stopped_origins=['epubs.siam.org:403', 'Google:429/challenge'],
                    usable_Google_results=0, usable_Bing_exact_title_results=0,
                    Google_HTTP200_semantic_status='JAVASCRIPT_SHELL_NO_RESULTS',
                    global_web_tool='POST /v1/alpha/search HTTP404; no source content',
                    browser_session='qattn-neighbor-sources CLOSED',
                    unrelated_browser_session='UNTOUCHED',
                    no_more_requests_in_this_audit=True),
    resources=dict(task_rows=0, feature_reads=0, checkpoint_reads=0, native_head_forwards=0,
                   circuit_forwards=0, optimizer_updates=0, GPU_allocations=0,
                   remote_server_writes=0, COS=0, model_API_inference=0, package_installs=0)))

visuals = ['fastweights-p3.png', 'fastweights-p5.png', 'fastweights-p6.png',
           'fastweights-p7.png', 'fastweights-arxiv-p10.png', 'mismatched-supplement-p1.png']
write('reading-record.json', dict(
    schema='researcher-reading-attestation.v1', recorded_at=now,
    conference=dict(source_sha256=conference['sha256'], identity='title/authors/year and official record matched',
                    text_pages_read=list(range(1, 10)), reference_page_9='reference list inspected; cited originals not acquired/read',
                    scope='Mechanism Eq1-5 and experimental controls/results; no code reproduction.',
                    visual_pages_inspected=[3, 5, 6, 7],
                    page8='Text read; plotted curves not digitized; no exact RL effect or uncertainty claim.'),
    arxiv_v1=dict(source_sha256=arxiv['sha256'],
                  identity='same title/five authors, official exact-ID API and PDF first-page arXiv:1610.06258v1 [stat.ML] 20 Oct2016',
                  fully_read_pages=[10, 11],
                  partial_pages={'1':'title/authors/version identity', '3':'Eq1 and iterative update window',
                                 '4':'Eq5/default eta/lambda and validation selection windows'},
                  visual_pages_inspected=[10],
                  not_read_as_full_document=True),
    mismatched_supplement=dict(source_sha256=supplement['PDFs'][0]['sha256'],
                               text_and_visual_pages_inspected=[1],
                               remaining_22_pages='parsed but not read; excluded from claims'),
    knyazev=dict(fulltext_pages_read=[], metadata_only=True),
    visual_sha256={n:sha((r / 'visual' / n).read_bytes()) for n in visuals},
    terminal_correction='One local extraction stopped on GBK UnicodeEncodeError; rerun with PYTHONUTF8=1 exited0. No redownload.'))

write('comparison.json', dict(
    status='HOLD_NO_NEIGHBOR_CLEARANCE',
    source_facts=[
        dict(id='FW-MEMORY', source='conference', pages=[3],
             fact='Decayed outer-product hidden-state memory; matrix-vector application is dot-product attention over prior hidden states (Eq1-4).'),
        dict(id='FW-UPDATE', source='conference', pages=[3,4],
             fact='Repeated query update has sustained slow/input boundary, nonlinearity and layer normalization (Eq2/5).'),
        dict(id='FW-HYPER', source='conference', pages=[4],
             fact='Default fast learning rate .5 and decay .95 unless stated; validation grid search; LN used for remaining experiments.'),
        dict(id='FW-APPENDIX', source='arxiv_v1', pages=[10,11],
             fact='Associative retrieval uses eta .5/lambda .9, ReLU and one fast update per time step; RL averages top5 random-search models. Sustained boundary and identity-memory variants are not exactly equivalent for nonzero memory.')],
    own_algebra=dict(
        definitions='Bt=rho_t+EPS I; B=sumBt; A=sumuBt; z0=(phi* A phi)/(phi* B phi); C=A-z0B fixed',
        decomposition='C=sum_t (u_t-z0)rho_t + EPS sum_t(u_t-z0) I',
        normalized_exponential_identity='exp(sC)phi/||exp(sC)phi|| = exp(s sum_t(u_t-z0)rho_t)phi/||exp(s sum_t(u_t-z0)rho_t)phi|| for finite real s',
        boundary='Scalar-identity factor cancels only in normalized query. EPS still determines z0 and final original readout. No EPS or operator change.',
        interpretation='class-dependent signed outer-product operator and spectral filter; known building blocks, not accepted new theory',
        conditional_linear_iteration='For hypothetical f=identity, no LN, h0=b, and h_{k+1}=b+Mh_k: hS=sum_{k=0}^S M^k b. This is NOT the actual nonlinear Ba Eq5 or a proof of model inequivalence.'),
    distinctions=[
        dict(field='memory coefficients', candidate='class utility minus fixed original class score; can be signed',
             audited_Ba_model='positive fast learning/decay coefficients over temporal hidden-state history',
             limit='No claim that signed/class-conditioned memories are novel in the literature.'),
        dict(field='query update', candidate='normalized fixed-operator exponential',
             audited_Ba_model='LN/nonlinearity with sustained boundary in a recurrent inner loop',
             limit='These formulas differ; no expressivity separation or absent-predictor claim for general fast-weight models.'),
        dict(field='prior matched-alpha/value contrast', candidate='0 versus .9640256520 on general pure-state fixture',
             audited_Ba_model='context memory may also retain geometry beyond scalar attention',
             limit='Prior contrast excludes only its pinned scalar/diagonal/old formulas; actual FastWeights model not excluded.'),
        dict(field='ordering', candidate='sums are permutation-invariant conditional on fixed states/utilities',
             audited_Ba_model='temporal decay and recurrent state history',
             limit='Whole Q-RMEA NLP model is not permutation-invariant: contextual features/roles can encode order.')],
    unresolved=['Knyazev original residual/preconditioner/trial-subspace update and fixed-exponential relation',
                'Substantive distinctness beyond known residual sharpening and fast associative memory',
                'Native ansatz joint construction of the matched phase fixture',
                'Natural-task gold-versus-rival derivative alignment and protected-correct effects',
                'Fair learned real/operator/scalar/diagonal/strong-Gaussian controls and independent task confirmation'],
    admission=dict(method_selected=False, new_theory=False, task_diagnostic=False, training=False,
                   newL3='NOT_ACCEPTED', newL4T='NOT_ACCEPTED', goal='ACTIVE')))

write('reported-tables.json', dict(
    status='TRANSCRIBED_SOURCE_RESULTS_NOT_REPRODUCED', source_sha256=conference['sha256'],
    table1=dict(page=5, metric='test error percent', columns_recurrent_units=[20,50,100],
                IRNN=[62.11,60.23,.34], LSTM=[60.81,1.85,0], A_LSTM=[60.13,1.62,0], FastWeights=[1.81,0,0]),
    table2=dict(page=6, metric='MNIST test error percent', columns_features=[50,100,200],
                IRNN=[12.95,1.95,1.42], LSTM=[12,1.55,1.10], ConvNet=[1.81,1.00,.9], FastWeights=[7.21,1.30,.85]),
    table3=dict(page=7, metric='facial expression test accuracy percent',
                IRNN=81.11,LSTM=81.32,ConvNet=88.23,FastWeights=86.34),
    fairness_limits=['Same hidden-size columns do not certify equal total temporary memory/compute/parameter budget.',
                     'Glimpse/store policies predetermined; no learned-glimpse-policy comparison.',
                     'arXiv RL top5 hyperparameter-search averages are not independent-seed confirmation.',
                     'Different tasks/metrics from Q-RMEA Macro; no numerical pooling or benefit transfer.']))

native = Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea.py')
assert sha(native.read_bytes()) == '9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
write('next-direction.json', dict(
    decision='HOLD_CANDIDATE; ALLOW_ONE_INDEPENDENT_SYNTHETIC_CPU_IMPLEMENTATION_CHECK',
    action='Freeze then execute a constructive native three-qubit/two-layer phase contrast, independent of unresolved source novelty.',
    hypothesis='The existing matched-alpha/all64value/population contrast is jointly realizable by the actual native preparation, query and value maps with explicit synthetic inputs/parameters.',
    source=dict(path=str(native),sha256=sha(native.read_bytes()), inspected_lines='14-33 circuit;36-68 head'),
    minimum=dict(contexts=2, words_per_context=2, classes=40, value_channels=64,
                 device='CPU', dtype='float64/complex128',
                 parameters='One explicit construction, no optimizer/random search. prepare raw angle0 selects h0; only first-layer wire0 RY nonzero, all RZ/other RY0; all effect angles0; value channels0/1 select h1/h2; class0 decoder selects channel0; other classes/bias0.',
                 inputs='h0=+atanh(.5) for both words in A; second word negated in B; h1=(-1,+1),h2=(2,-3), others0; role0 and both valid.',
                 prospective_states='(|0>+|2>)/sqrt2 and (|0>-|2>)/sqrt2 after native double CNOT ring; query|0>. Verify using independent matrix/basis construction.',
                 finite_illustration='s=1 solely repeats the earlier constructed algebra value; not a task hyperparameter selection.',
                 budget='at most8 native head forwards/16 circuit calls and10 CPU seconds measured inside computation; no task/checkpoint features, model inference or GPU. Freeze exact call accounting before execution.'),
    success_gate='Native/independent states agree to1e-12 up to stated global phase, unit norms, both contexts native attention/all64values/populations match, native logits0, guided class0 A0/B tanh2/(1+2e-6) within1e-12; pinned source unchanged.',
    kill_gate='Any mismatch stops the constructive claim, preserves trace and corrected interpretation; no strength/norm/EPS/rank/seed rescue or automatic task admission.',
    uncertainty='Synthetic reachability only, not trained-checkpoint/natural-task frequency, utility, quantum necessity or substantive novelty. Knyazev fulltext remains prerequisite for neighbor clearance before task-method admission.',
    authorization='Existing sustained research goal covers reversible bounded local synthetic verification; no additional confirmation required.',
    prohibited=['new task rows/features/checkpoint reads','optimizer updates','GPU/server/COS writes',
                'new model consultations','new original-source network requests this audit','training admission'],
    preserved=['original limited L4 and all positive/negative history','gen98 STOP','gen102 STOP','original quantum target'],
    goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED'))

report = '''# 两近邻来源审计：Fast Weights 已核，Knyazev 全文仍缺

[已测事实] 本次是原文与证据边界检查，不是新实验结果。两篇文献身份已核；Fast Weights 的会议原文及同题 arXiv v1 附录已按范围阅读。Knyazev 只有官方元数据，原文审计未完成。候选继续 HOLD，新 L3/L4-T 未接受，原量子目标 ACTIVE。

## 项目实际工作流

项目用冻结文本表示 h、四维角色和有效词 mask，分别形成三量子比特词态与40个类 query；Born 概率加原 EPS=1e-6 后归一，池化64维词 value，再由各类 decoder 与 bias 输出关系分类分数。原协议训练只用 TRAIN，按冻结的验证与评估规则报告 Macro 指标；公平比较包括原 quantum、dephased 和强 Gaussian/经典控制。训练、独立确认、关键消融与独立审核共同决定证据等级。本轮不读取任务 feature/checkpoint，不训练、不评估任务 Macro，也不申请共享 GPU。历史 gen98 十三门10PASS 后 STOP、gen102 固定 teacher value 替换三门0PASS 后 STOP 均保持。

上一检查点把 signed utility 构成的 C=A-z0B 固定，再归一 exp(sC)phi；证明高于起点而不保证随 s 单调，更不保证分类改善。其960行代数、12个精确例、分类反例和独立静态审核均为既有证据。本次只补近邻来源检查，不把旧结果重计成新实验。

| 比较项 | 上一检查点 | 本次新增 |
| --- | --- | --- |
| 算符行为 | 特定构造能区分原 alpha/value 相同的两个 context | 对照实际 fast-memory 文献，收窄排除范围 |
| 最近邻依据 | reviewer 给出两篇题名 | 准确身份、版本、阅读页与获取失败记录 |
| 证据等级 | 无任务准入、无新 T | 仍 HOLD；未提升 |

## 原文核验结果

| 文献/版本 | 取得与实际阅读 | 可支持的范围 |
| --- | --- | --- |
| Ba 等，NIPS2016会议版，9页 | 全文1–9页；3/5/6/7页图像复核 | Eq1–5、原表1–3、控制与限制 |
| 同题 arXiv:1610.06258v1，11页 | 1页身份；3/4页指定公式窗口；10/11页全文，10页图像 | 单独的固定版本实验附录，不声称全篇与会议版相同 |
| Knyazev，2001，SIAM JSC23(2):517–541，DOI10.1137/s1064827500366124 | Crossref/索引身份核验；全文未取得、未读 | 不归因任何算法公式或固定指数更新有无 |
| 官网链接 Supplemental.zip | 23页PDF仅检查第1页文本与图像 | 与 fast weights 附录内容不符，隔离；其真实论文身份未核 |

来源：[会议官方页](https://papers.nips.cc/paper_files/paper/2016/hash/9f44e956e3a2b7b5598c625fcc802c36-Abstract.html)、[固定 arXiv v1](https://arxiv.org/abs/1610.06258v1)、[Knyazev DOI](https://doi.org/10.1137/s1064827500366124)。本地 hash、页码及状态见 source-receipts.json 和 reading-record.json。

[原文事实] Ba 的 Eq1–4 用随时间衰减的隐藏态外积存储序列历史；Eq5 的 query 内循环含持续输入边界、非线性及 layer normalization。默认 eta=.5、lambda=.95；v1附录的关联检索 lambda=.9 单独保留。v1附录还说明 RL 对随机超参数搜索的前5个模型取平均，不能解释成独立 seed 复现。会议版 Eq1 与 v1 的时间下标印法也分别保存。

[机制解释] 自有代数 C=sum_t(u_t-z0)rho_t + EPS sum_t(u_t-z0)I 中，标量 I 因子在归一指数 query 时抵消；EPS 仍影响 z0 和最终原 readout，未改 EPS。这里可解释为类相关 signed 外积算符与谱滤波。它与已读 Ba 模型的具体更新公式不同，但这些已知构件及其组合尚未构成实质近邻清关或新理论。

[边界更正] 先前的双 context 构造只排除了所列固定公式。实际 fast-memory 模型也可保留原 scalar attention 以外的上下文几何，因此该例没有排除实际 Fast Weights 模型；这补充了上一报告“有功能区别、未清关”的限制。不能把假设线性、去 LN 后的多项式内循环当作 Ba 实际模型，也不能由某公式不同推出模型表达力分离。给定词态与 utility 的求和不含顺序，不等于整个使用上下文 h/role 的 NLP 模型顺序不敏感。

## 结果与公平性

[原文结果、未复现] 会议表1是 test error%，列为20/50/100个 recurrent units：FastWeights 1.81/0/0，LSTM60.81/1.85/0，存在平局。表2是 MNIST error%，特征50/100/200：FastWeights7.21/1.30/.85，对 ConvNet1.81/1.00/.9 为两负一胜。表3 facial accuracy%：FastWeights86.34，ConvNet88.23。完整紧凑转录见 reported-tables.json；不同任务和指标不合并为 Q-RMEA 的 Macro。

[限制] 相同隐藏单元数不认证总临时记忆或计算公平；视觉 glimpse/store 策略预先指定。RL 曲线未数字化，不报告额外精确数值或置信区间。这些结果不替代本项目的强经典控制、独立确认及消融。

本轮18个普通元数据/下载请求与1次公开浏览器导航有预算记录。SIAM403后停止该源；Google200为 JS shell、无可用结果，单次普通浏览器导航429后关闭专用会话，未触碰挑战或其他浏览器会话。Bing返回无关词典页，不能据此声称论文不存在。web 工具后端404未提供任何源内容。最后 www.math.ucdenver 传输错误，未取得PDF。获取失败及补充包错配不代表科学反证，也不由解析页数推断已阅读。

## 案例与证据上限

本轮未执行模型训练、推理或量子仿真；sample-trace.v2 任务案例和批量因果轨迹不适用。不是任务诊断的已完成条目，Macro/置信区间/新 split 覆盖均未提供。取得与阅读证据独立保存：两份正式论文PDF按 hash 私有留存，公开包只含小型元数据、阅读记录与自有分析，不含全文、解析文本、浏览器 profile 或搜索日志。

[已测事实] 外积记忆已经是原文明确机制；官网补充包的来源链接与实际内容需分别验证。

[尚未证明] Knyazev 的实际算法关系、原生线路对构造的联合可达、自然任务中 gold 相对 rival 的导数对齐、强经典公平比较、实质新颖性和独立量子贡献。既有“每类分数不降”与固定 teacher 的431例障碍均不放宽。

## 唯一推荐下一步

先冻结并完成一个小型 CPU 原生线路构造核验：用现有 prepare/effect/value/decoder 的显式参数和两组人工 h，检验此前相位构造能否由实际三比特两层线路同时生成。只测合成可达性，不读取任务或 checkpoint；独立基向量/CNOT 实现做对照。成功要求1e-12精度的态、mask、全64 value、原 attention 匹配及0/.9640256520的构造 readout；任一失败停止该可达性主张。预算最多8次 native-head/16次 circuit、内部计算10 CPU秒，具体调用数执行前冻结。不得改 strength/norm/EPS/rank/seed 救援。

该检查由持续科研目标覆盖，独立于文献缺口；即便通过，也不准入任务方法或训练。Knyazev 原文仍是后续近邻清关条件。自然任务导数诊断排在其后，当前未准入；重新训练或 GPU 筛选不推荐。详见 next-direction.json。

原限定 L4、全部正负历史、gen98与gen102 STOP保持，原量子研究主线 ACTIVE。证据索引：source-receipts.json、reading-record.json、comparison.json、reported-tables.json、verification.json、next-direction.json。
'''
(r / 'analysis.md').write_bytes(report.encode())
write('report-build.json', dict(status='BUILT_PARTIAL_SOURCE_REPORT', recorded_at=now,
                                code_sha256=sha(Path(__file__).read_bytes()),
                                files={n:sha((r/n).read_bytes()) for n in
                                       ['source-receipts.json','reading-record.json','comparison.json',
                                        'reported-tables.json','next-direction.json','analysis.md']}))
print(json.dumps(dict(status='BUILT_PARTIAL_SOURCE_REPORT', files=6)))
