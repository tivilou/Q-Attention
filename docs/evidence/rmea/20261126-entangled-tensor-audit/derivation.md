# 原生三qubit读出的张量表示：精确对应及其限度

这份推导应用已知Pauli-transfer/tensor方法，不主张新T。对象为native rmea.py SHA9f4bfc57：两层，每层各wire按RY、RZ后CNOT0→1→2→0；12个角度。θ=πtanh(prepare[h,roles])，query角亦πtanh。这里没有改电路/训练。

令P_a是三qubit Pauli strings，Tr(P_a P_b)=8δ_ab。密度坐标v_a(θ)=Tr(P_aρ(θ))/√8为64维实向量。因此

`p_Q(θ,η)=v(θ)^T v(η)`，`p_D(θ,η)=v(θ)^T Π_Z v(η)`，

其中Π_Z保留8个仅由I/Z组成的Pauli strings，等价于计算基完全去相干。两者是相同角度输入上的有效PSD核。

在I/X/Y/Z局部Pauli坐标中，RY(θ)在X/Z平面旋转，RZ(θ)在X/Y平面旋转。每个转移矩阵均为`M(θ)=M0+Mc cosθ+Ms sinθ`。CNOT是固定实正交Pauli转移矩阵。依电路顺序逐次作用，并把每个固定CNOT并入相邻站点，得到

`v(θ)=B t(θ)`，`t(θ)=⊗_{j=1}^{12}(1,cosθ_j,sinθ_j)`。

B有64行、3^12=531441列，但可通过虚维64的矩阵乘积链收缩，无须生成展开矩阵。故`C_Q=B^T B`为PSD、矩阵rank≤64；`C_D=B^T Π_Z B`为PSD、rank≤8。将B链复制并配对得到core的一种MPO表示，其虚维上界64²=4096。矩阵rank与MPO最小bond dimension是不同概念；这里未求最小bond，也未生成531441平方的core。

源p11 Eq67的局部T=(1,cos,sin)/√2；换到该归一化坐标时，core乘2^12即可，是规范缩放而非不同方法。源p9允许一般data preprocessing，故把θ代为可训练πtanh线性映射后，以上点态恒等仍成立；光滑处链式法则也给出相同参数梯度。此处不把representation identity当source泛化定理。

Q−D核对应`C_coh=B^T(I−Π_Z)B`，也是PSD；但PSD order并非每个cross-score非负，更不保证按词归一化后乘带符号utility的logit提高。固定96个合法角度的完整配对中，coherence项有3644正、5476负非对角元素，未选有利子集。已知终端共同unitary/CNOT不变性在旧记录中保留，这里没有重复该fixture。

实际原生alpha还含EPS=1e−6、padding mask及每relation的词和分母；z=Σ_t alpha_rt u_rt+b_r，u_rt可正可负。因此不能把整个句级分类器当成源中固定kernel ridge regression，也不能把v的64维、core的谱或特征间关联直接当自然语言证据结构。源p4明确参数固定后可作核分析，p16示例却限单层编码及简单角度域，pp22–23 Mercer构造还要求相对PX的函数正交化。本项目学习中的角度pushforward分布/归一化/联合utility没有被这些例子的结果覆盖。

独立核验边界：源native复杂statevector与real-transfer实现比较全概率、alpha、logits、全部7个参数tensor和合成h梯度；另用NumPy8×8稠密门重算全部port和双侧角度方向差分。合成案例不是实际训练数据、泛化证据、拟合收益或科学独立评审。小qubit可模拟也不是本项目拒绝量子机制的准则；本迁移不产生新函数，才是这里没有新候选的原因。
