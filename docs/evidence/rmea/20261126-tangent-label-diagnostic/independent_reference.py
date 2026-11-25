"""Dense Kronecker gates and source-column CNOT construction; no native imports."""
import math
import numpy as np

def circuit(angles):
    state=np.zeros(8,dtype=np.complex128);state[0]=1
    for layer in range(2):
        for wire in range(3):
            y,z=angles[layer,wire]
            ry=np.array([[math.cos(y/2),-math.sin(y/2)],[math.sin(y/2),math.cos(y/2)]],dtype=complex)
            rz=np.diag([np.exp(-.5j*z),np.exp(.5j*z)])
            full=np.ones((1,1),dtype=complex)
            for k in range(3):full=np.kron(full,rz@ry if k==wire else np.eye(2))
            state=full@state
        for control,target in [(0,1),(1,2),(2,0)]:
            matrix=np.zeros((8,8),dtype=complex)
            for column in range(8):
                bits=[int(b) for b in format(column,'03b')]
                if bits[control]:bits[target]=1-bits[target]
                row=4*bits[0]+2*bits[1]+bits[2]
                matrix[row,column]=1
            state=matrix@state
    return state

def compute(h,eps):
    angles=np.zeros((2,2,2,3,2));angles[:,:,0,0,0]=np.pi*np.tanh(h[:,:,0])
    psi=np.stack([np.stack([circuit(a) for a in context]) for context in angles])
    phi=np.stack([circuit(np.zeros((2,3,2))) for _ in range(40)])
    values=np.zeros((2,2,64));values[:,:,0]=h[:,:,1];values[:,:,1]=h[:,:,2]
    u=np.zeros((2,40,2));u[:,0]=values[:,:,0]
    p=np.empty((2,40,2));alpha=np.empty_like(p);original=np.zeros((2,40))
    guided=np.zeros_like(original);q=np.zeros((2,40,8),dtype=complex)
    A=np.zeros((2,40,8,8),dtype=complex);B=np.zeros((2,8,8),dtype=complex);C=np.zeros_like(A)
    for b in range(2):
        rhos=[np.outer(v,v.conj()) for v in psi[b]]
        bts=[rho+eps*np.eye(8) for rho in rhos];B[b]=sum(bts)
        for c in range(40):
            p[b,c]=[abs(np.vdot(phi[c],v))**2 for v in psi[b]]
            raw=p[b,c]+eps;alpha[b,c]=raw/raw.sum()
            A[b,c]=sum(u[b,c,t]*bts[t] for t in range(2))
            original[b,c]=np.vdot(phi[c],A[b,c]@phi[c]).real/np.vdot(phi[c],B[b]@phi[c]).real
            C[b,c]=A[b,c]-original[b,c]*B[b]
            lam,vectors=np.linalg.eigh(C[b,c])
            qq=vectors@(np.exp(lam)*(vectors.conj().T@phi[c]));q[b,c]=qq/np.linalg.norm(qq)
            guided[b,c]=np.vdot(q[b,c],A[b,c]@q[b,c]).real/np.vdot(q[b,c],B[b]@q[b,c]).real
    return dict(angles=angles,psi=psi,phi=phi,values=values,utilities=u,probability=p,alpha=alpha,
                original=original,guided=guided,q=q,A=A,B=B,C=C)
