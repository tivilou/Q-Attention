"""Freeze the primary-selected next hypothesis, preserving every failed comparator."""
from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
save=lambda p,v:p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
s=json.loads((r/'summary.json').read_bytes())
advice=json.loads((r/'theory-advice.json').read_bytes())
assert s['gate']=='DEFER_EXACT_SEMANTIC_CONNECTED_SCREEN'
assert advice['response']['recommendation']=='ONE'
n=r.parent/'distinct-token-evidence-20261125'
assert not n.exists();n.mkdir()
references={'previous/'+k:v for k,v in s['results'].items()}
references.update({'earlier/'+k:v for k,v in s['reused_results'].items()})
arms=['quantum_distinct','dephased_distinct','gaussian_distinct','quantum_permuted_distinct','quantum_collision_mass']
selection=dict(candidate='distinct-token-evidence',status='DESIGN_ONLY_NOT_IMPLEMENTED_NOT_RUN',
    selection_owner='primary_agent',advice_provenance='gpt-6.2-astra/high, bounded advice; no selection or execution authority',
    hypothesis='Deleting same-position self-products may improve relation discrimination beyond a pure attention-collision-mass multiplier.',
    mechanism='mu=E_alpha[v]; M=mu_even*mu_odd; S=E_alpha_squared[v_even*v_odd]; U=M-S. Keep U unnormalized.',
    primary_added_control='quantum_collision_mass uses (1-sum(alpha**2))*M, unlike the proposed U. This prevents treating a mere attention-concentration effect as distinct evidence binding.',
    arms=arms,expected_trainables=81924,
    budget=dict(fit=8192,assessment=4096,seed=13,updates=1024,physical_updates=5120,batch_size=128,microbatch_size=4),
    initialization='Accepted gen96 shared query state and zero40x32 moment decoder, original key/value/base decoder/bias; no initialization search',
    frozen_pairs='Same32 adjacent value channels, no selected token subset, position thresholds or pair scan',
    promotion='Quantum distinct strictly beats four new controls and ALL nine retained previous rows on all40 Macro; thirteen gates. All fits before assessment. Pass permits full single-seed protocol design only.',
    stop='Any failure defers exact fixed profile; no normalization, concentration, pair, rank, tokenization, scale, LR, seed, budget, cohort or comparator rescue.',
    known_identity_not_new_theory=True,quantum_advantage_established=False,full_or_multiseed_started=False,
    reused_rows=references)
save(n/'selection.json',selection)
source=dict(previous_summary_sha256=hashlib.sha256((r/'summary.json').read_bytes()).hexdigest(),
    cohort_sha256=json.loads((r/'design.json').read_bytes())['cohort_sha256'],
    shared_query_sha256=hashlib.sha256((r/'shared_query.py').read_bytes()).hexdigest(),
    label_code_sha256=hashlib.sha256((r/'label-codes.json').read_bytes()).hexdigest(),
    stable_core_sha256=hashlib.sha256((r/'screen.py').read_bytes()).hexdigest(),
    advice_packet_sha256=advice['packet_sha256'])
save(n/'source-identities.json',source)
protocol='''# 固定下一小筛选：删除同token二次自乘

上一轮Qmeanproduct .504045671350好于Qcov .478786491123，但未胜最强Gcov
.512054050292，且CE/正确数更差；这不构成成功，也未隔离原因。新的假设只删除
均值乘积里相同token位置的自乘，检验不同位置之间的配对证据是否更有用。
这是新开发假设，不重新解释旧八门失败。原先所有确切配置及历史结果保持原状。

固定mu=sum(alpha*v)，M=mu_even*mu_odd，S=sum(alpha**2*v_even*v_odd)，
U=M-S=sum_{t!=u}alpha_t*alpha_u*v_t,even*v_u,odd。U不除以
1-sum(alpha**2)，不加温度或缩放。仍用原32相邻通道及40x32零初始化decoder，
其他部分原样复用gen96共享query初始状态，各81924 trainables。

五臂固定：Qdistinct、Ddistinct、Gdistinct、QpermutedDistinct、QcollisionMass。
后者只用(1-sum(alpha**2))*M。它由主模型加入，用于区分真正的去自乘差异和
单纯attention集中度调制；外部建议没有替代这项主模型判断。

Qdistinct必须在全40类Macro严格胜四个新对照和全部九个已存参照，共十三门。
所有比较身份及值在selection.json固定，包含上一轮全部五臂和四个更早参照，
尤其保留.504045671350的Qmeanproduct和.512054050292的GnativeCov。
不删对照、转成只胜自身基线、只看稀有类、CE或特定关系来宣布成功。

沿用同TRAIN8192/4096、句子组隔离、全部40类、seed13、1024updates/b128/micro4、
原AdamW/FP32/TF32off；五fit结束才统一评估，总5120物理updates。相同预算
900sec/arm/3600sec总上限，调度现有双GPU。旧参照直接复用，无baseline重跑。
八门旧规则和失败状态不改；这次十三门全过也只允许设计完整单seed协议。
任何门失败即暂缓本确切profile，不扫归一化、集中度、通道、pair、rank、token、
scale、LR、seed、budget、cohort或对照救援，不推断整个方法族不可能。

只测试新增U及mass-control独立双token公式/梯度、单token U=0、全同value时
U=(1-sum(alpha**2))*M、padding、零moment共享端点和clone actual8。复用query/
circuit/稳定core及原生命周期覆盖；不重跑串行/暂停/多卡/COS恢复矩阵。

U是已知非对角二次统计身份，适用于任何归一attention，并非新T或量子排他性质。
它对token位置对称；即使有收益，也未证明subject/object绑定、次序或量子优势。
attention集中度与value分布的混合效应须由固定mass-control区分，结果仍是暴露过的
单seed开发集筛选，不能直接接受独立L3/T。0新VALID/TEST或COS、Windows0模型/
特征/逐例数组。尚未实现、审核、分配或运行。
'''
(n/'protocol-draft.md').write_bytes(protocol.encode())
decision=dict(candidate=str(n),status='DESIGN_ONLY',new_fits_started=0,new_theory_accepted=False,
    closed_current_profile=s['gate'],reason='Distinct-token deletion hypothesis; add concentration-only control and retain all nine existing outcomes, not a covariance tuning rescue.',
    primary_selection=True,advice_only=True)
save(r/'next-direction.json',decision)
print(json.dumps(dict(candidate=str(n),five_arms=arms,old_references=len(references),gates=13,not_started=True)))
