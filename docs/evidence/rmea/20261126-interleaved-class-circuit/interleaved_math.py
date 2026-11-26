"""Data-free dense reference for the proposed operator placement. No task model."""
import numpy as np

EPS=1e-6

def block(angles):
    a=np.asarray(angles,dtype=np.float64)
    if a.shape!=(3,2) or not np.isfinite(a).all():raise ValueError('Expected finite 3x2 angles')
    u=np.eye(8,dtype=np.complex128)
    for wire,(y,z) in enumerate(a):
        c,s=np.cos(y/2),np.sin(y/2)
        ry=np.array([[c,-s],[s,c]],dtype=np.complex128)
        rz=np.diag([np.exp(-.5j*z),np.exp(.5j*z)])
        local=rz@ry
        gate=np.array([[1.]],dtype=np.complex128)
        for j in range(3):gate=np.kron(gate,local if j==wire else np.eye(2))
        u=gate@u
    for control,target in ((0,1),(1,2),(2,0)):
        perm=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
        u=u[perm,:]
    return u

def circuit(angles):
    a=np.asarray(angles,dtype=np.float64)
    if a.shape!=(2,3,2):raise ValueError('Expected 2x3x2 angles')
    return block(a[1])@block(a[0])

def probabilities(word_angles,query_angles):
    w1,w2=[block(a) for a in word_angles];q=circuit(query_angles);qd=q.conj().T
    out=(qd@w2@w1)[0,0]
    mid=(w2@qd@w1)[0,0]
    d=(np.abs(w2)**2@np.abs(qd)**2@np.abs(w1)**2)[0,0]
    return dict(out=float(abs(out)**2),mid=float(abs(mid)**2),dephased=float(d))

def realify(u):
    u=np.asarray(u);return np.block([[u.real,-u.imag],[u.imag,u.real]])

def representations(word_angles,query_angles):
    w1,w2=[block(a) for a in word_angles];q=circuit(query_angles);qd=q.conj().T
    initial=np.eye(8,dtype=np.complex128)[:,0]
    a=w2.conj().T@initial;b=w1@initial
    qvec=q.reshape(-1,order='F')/np.sqrt(8)
    token=np.outer(b,a.conj()).reshape(-1,order='F')
    amp=np.sqrt(8)*np.vdot(qvec,token)
    v=realify(w2)@realify(qd)@realify(w1)@np.r_[initial.real,initial.imag]
    return dict(vec=float(abs(amp)**2),real16=float(v[0]**2+v[8]**2),
                direct=float(abs((w2@qd@w1)[0,0])**2))

def pool(probability,utility,mask):
    p=np.asarray(probability,dtype=np.float64);v=np.asarray(utility,dtype=np.float64)
    m=np.asarray(mask,dtype=bool)
    if p.shape!=v.shape or p.shape!=m.shape or not m.any():raise ValueError('Invalid word inputs')
    if not np.isfinite(p).all() or not np.isfinite(v).all():raise ValueError('Nonfinite input')
    raw=(p+EPS)*m;alpha=raw/raw.sum()
    return float(alpha@v),alpha
