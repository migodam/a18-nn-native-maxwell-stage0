"""Equivalent CPU normal solve for validating failed Woodbury cancellation.

Same projected Galerkin operator, material gauge, regularizer and 1e-10 gate.
One deterministic residual correction with the same Woodbury inverse. No
jitter, PCG, new physics operator, convergence tuning or full-wave action.
"""
import numpy as np
from scipy import linalg as la

def validated_endpoint(engine,U):
    pb=engine.PBflat.reshape(engine.P,engine.q,engine.qc).transpose(0,2,1)
    a=U.conj().T@engine.L@U;sv=la.svdvals(a)
    if sv[-1]/sv[0]<=1e-10:return dict(status='CURRENT_NEAR_SINGULAR')
    F=np.stack([la.solve(a,U.conj().T@b,assume_a='gen') for b in pb])
    cv=engine.S@U;rc=np.linalg.qr(cv,mode='reduced')[1]
    z=np.einsum('jk,pkq->pjq',rc,F).reshape(-1,engine.q)
    wc=engine.r.reshape(engine.P,2*engine.m);wc=wc[:,:engine.m]+1j*wc[:,engine.m:]
    ell=engine.ell[:engine.q]+1j*engine.ell[engine.q:]
    g=np.einsum('pkq,pk->q',F.conj(),wc@cv.conj())+ell
    if engine.q<=256:
        h=z.conj().T@z+engine.lam*np.eye(engine.q);factor=la.cho_factor(h,lower=True)
        inv=lambda b:la.cho_solve(factor,b)
    else:
        h=z@z.conj().T+engine.lam*np.eye(z.shape[0]);factor=la.cho_factor(h,lower=True)
        inv=lambda b:(b-z.conj().T@la.cho_solve(factor,z@b))/engine.lam
    normal=lambda s:z.conj().T@(z@s)+engine.lam*s+g
    s=-inv(g);before=float(la.norm(normal(s))/max(la.norm(g),1e-300))
    s=s-inv(normal(s));after=float(la.norm(normal(s))/max(la.norm(g),1e-300))
    ys=np.einsum('mk,pkq,q->pm',cv,F,s)
    step=np.concatenate([s.real,s.imag]);jstep=np.concatenate([ys.real,ys.imag],axis=1).reshape(-1)
    return dict(status='OK' if np.isfinite(step).all() and after<=1e-10 else 'NORMAL_RESIDUAL_FAILED',
                step=step,j_step=jstep,normal_relative_residual=after,initial_residual=before,
                correction_count=1,scope='EQUIVALENT_NORMAL_EQUATION_VALIDATION_ONLY')
