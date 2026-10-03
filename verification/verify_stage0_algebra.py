"""Equivalent new tests. Original A18 verification source/results not supplied."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[n]='1'
import json,sys,time,resource
from pathlib import Path
import numpy as np
from scipy import linalg as la
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from exchange_core import endpoint,orth,removal_core,swap_from_core,bank_increment,realify,atom_swap,gain_directional,gain_error_bound,OutputErrorBounds
cfg=json.loads((ROOT/'stage0_config.json').read_text())
rng=np.random.default_rng(cfg['verification_seed']);start=time.process_time();cases=[]
def cn(shape):return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
def err(a,b):return float(la.norm(a-b)/max(la.norm(a),la.norm(b),1e-300))
for j in range(cfg['verification_cases']):
    n,p,m,P=18,9,7,6
    L=2*np.eye(n)+.2*cn((n,n))/np.sqrt(n);S=cn((m,n));B=cn((P,n,p))
    U=orth(cn((n,5)));A=orth(cn((n,10))) # deliberately non-nested
    def R(V):return V@la.solve(V.conj().T@L@V,V.conj().T,assume_a='gen')
    ru=R(U);I=np.eye(n);JF=np.stack([S@la.solve(L,b,assume_a='gen') for b in B])
    JU=np.stack([S@ru@b for b in B]);IU=np.stack([(I-L@ru)@b for b in B]);OU=S@(I-ru@L)
    defect=np.stack([OU@la.solve(L,b,assume_a='gen') for b in IU])
    two_sided=err(JF-JU,defect)
    JA=endpoint(L,S,B,A)['J'];jfull=realify(JF);ju=realify(JU)
    anchor=err(jfull-ju,(JA-ju)+(jfull-JA))
    atoms=orth(cn((n,12)));V=atoms[:,:4];aa=atoms[:,4:6];cc=atoms[:,6:8]
    core=removal_core(L,S,B,V,aa);swap=swap_from_core(core,cc)
    schur=err(swap['D'],swap['D_schur'])
    inc=bank_increment(core,cc);E=np.diag(np.exp(1j*rng.uniform(-np.pi,np.pi,2)))
    cp=inc['Z']@E;bar=cp-V@la.solve(V.conj().T@L@V,V.conj().T@L@cp)
    gamma=cp.conj().T@L@bar
    N=np.stack([cp.conj().T@(b-L@kv) for b,kv in zip(B,core.K)])
    phase=max(err(gamma,E.conj().T@inc['Gamma']@E),err(S@bar,inc['Y']@E),
              err(N,np.stack([E.conj().T@nn for nn in inc['N']])))
    ids=(0,1,2,3,4,5);dropped=(4,5);added=(6,7)
    direct=atom_swap(L,S,B,atoms,ids,dropped,added)
    perm=rng.permutation(12);inv=np.argsort(perm)
    other=atom_swap(L,S,B,atoms[:,perm],tuple(inv[list(ids)]),tuple(inv[list(dropped)]),tuple(inv[list(added)]))
    permutation=err(direct['D'],other['D'])
    x=rng.normal(size=p);d=rng.normal(size=p);ell=rng.normal(size=p);lam=.3
    u=cn((P*2*m,));z=cn((P*2*m,));eu_vec=.03*cn(u.shape);ez_vec=.05*cn(z.shape)
    q=gain_directional(u,z,x,d,lam,ell);qh=gain_directional(u-eu_vec,z-ez_vec,x,d,lam,ell)
    bound=gain_error_bound(u-eu_vec,z-ez_vec,OutputErrorBounds(la.norm(eu_vec),la.norm(ez_vec),'synthetic exact error norms',True))
    bound_violation=max(0.,abs(q-qh)-bound)
    # With exact u, a stronger bound follows from Q(u,zhat+e)-Q(u,zhat).
    zhat=z-ez_vec;ez=la.norm(ez_vec);v=la.norm(u+zhat)
    strengthened_violation=max(0.,abs(q-gain_directional(u,zhat,x,d,lam,ell))-v*ez-.5*ez**2)
    margin=float(rng.uniform(.01,2));eps=2*margin/(np.sqrt(v*v+2*margin)+v)
    critical_residual=abs(v*eps+.5*eps**2-margin)/margin
    cases.append(dict(case=j,two_sided_defect=two_sided,Schur_tail=schur,anchor_decomposition=anchor,
                      phase_covariance=phase,atom_permutation=permutation,gain_bound_violation=bound_violation,
                      exact_u_bound_violation=strengthened_violation,epsilon_crit_equation=critical_residual,
                      L_non_Hermitian_norm=float(la.norm(L-L.conj().T))))
maximum={k:max(c[k] for c in cases) for k in cases[0] if k!='case'}
passed=all(maximum[k]<1e-10 for k in maximum if k!='L_non_Hermitian_norm')
out=dict(status='PASS' if passed else 'FAIL',implementation='NEW_EQUIVALENT_TESTS_NOT_ORIGINAL_A18_REPRODUCTION',
         original_script_available=False,cases=len(cases),seed=cfg['verification_seed'],max_errors=maximum,
         cpu_s=time.process_time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         new_full_wave_rhs=0,tests=cases)
(ROOT/'verification/algebra_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='tests'}))
sys.exit(0 if passed else 1)
