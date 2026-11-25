# 原生 head 合成构造的解析对照

令 e_j 表示 wire0 为最高位的三比特计算基。所有 RZ 及其余 RY 都为0，仅首层 wire0 的 RY=±π/2。
该旋转把 e0 变为 (e0±e4)/sqrt2。第一轮 CNOT(0→1,1→2,2→0) 把二进制100依次映为110、111、011；第二层的单比特门均为单位阵，第二轮把011依次映为011、010、010。因此最终为 (e0±e2)/sqrt2。全零 query 角度给出 e0。

prepare.weight[0,0]=1，人工 h0=±atanh(.5)，可在不改原 π*tanh 映射下给出所需角度。value.weight[0,1]=1、value.weight[1,2]=1；两词 h1=(-1,+1)、h2=(2,-3)，使完整64维 value 在两个 context 中保持相同。decoder[0,0]=1，其余参数全部0；40类中的 row0 是非恒定 signed utility，另外39行恒为0。这证明一个固定的联合构造，未证明任意40个 query/utility 都联合可达。

Context A 为 plus/plus，B 为 plus/minus。两者词态 populations 都是第0、2位各1/2，原 query 的概率均为1/2，含原 EPS=1e-6 的 attention 仍为1/2，因此原 Q、D 的全部 logits 为0。A 的 C=0；B 的非零类满足 C=−|0><2|−|2><0|，B_operator 在此二维子空间上为(1+2EPS)I。

在仅为重复旧代数例的 s=1 下，B 中 query 归一为 [cosh(1)e0−sinh(1)e2]/sqrt(cosh(2))，readout=tanh(2)/(1+2EPS)=0.9640256520245128；A 仍为0。原 generic 构造使用 e1，此处固定映射 e1→e2；只是构造核验前已声明的基重标，不是任务配置救援。

证据来自实际 native head forward、稠密 Kronecker/逐基态 CNOT 参考与保存的完整数值阶段。此例全为实数，real8 控制仍可精确复现；不据此主张复幅必要性、任务准确率、一般线路/梯度认证、实质新颖性或新理论。CPU 计时对 scope 的10秒上界采用保守口径：包括 native-stage.json 的打包/写入时间，不包括后续 trace.json/results.json 的序列化；因此未放宽预算。
