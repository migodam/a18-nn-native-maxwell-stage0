"""Projected public physics only. No teacher/evaluator fields or full-wave calls."""
from dataclasses import dataclass
import time
import numpy as np
from scipy import linalg as la


@dataclass(frozen=True)
class PublicContext:
    L: np.ndarray
    S: np.ndarray
    PB: np.ndarray
    atoms: np.ndarray
    anchor: np.ndarray
    physical_embedding: np.ndarray
    r: np.ndarray
    ell: np.ndarray
    lam: float
    families: tuple
    selected_ids: tuple
    x: np.ndarray


def orth(a):
    # Exact no-projection convention of the pinned operators/project_orth.
    u,s,_=la.svd(a,full_matrices=False,check_finite=False)
    return np.ascontiguousarray(u[:,:np.count_nonzero(s>1e-10*s[0])]) if len(s) and s[0] else np.ascontiguousarray(u[:,:0])


def material_complex(d, p):
    d=np.asarray(d,float)
    if d.shape[0]!=2*p: raise ValueError('physical real material packing mismatch')
    return d[:p]+1j*d[p:]


def pack_data(z):
    # Illumination-major [Re(measurement), Im(measurement)].
    return np.concatenate([z.real,z.imag],axis=1).reshape(-1,*z.shape[2:])


def reduced_response(ctx,basis,d):
    u=orth(basis);a=u.conj().T@ctx.L@u
    s=la.svdvals(a)
    if len(s) and (not s[0] or s[-1]/s[0]<=1e-10): raise ValueError('unstable reduced core; no jitter')
    b=np.einsum('pnq,q...->pn...',ctx.PB,material_complex(d,ctx.PB.shape[-1]))
    coeff=np.stack([la.solve(a,u.conj().T@bb,assume_a='gen') for bb in b])
    currents=np.einsum('nk,pk...->pn...',u,coeff)
    return pack_data(np.einsum('mn,pn...->pm...',ctx.S,currents))


def projected_source(ctx,basis,d):
    u=orth(basis);a=u.conj().T@ctx.L@u
    b=np.einsum('pnq,q...->pn...',ctx.PB,material_complex(d,ctx.PB.shape[-1]))
    coeff=np.stack([la.solve(a,u.conj().T@bb,assume_a='gen') for bb in b])
    return b-np.einsum('nk,pk...->pn...',ctx.L@u,coeff)


def features(ctx):
    """One public receiver-direction injection sketch, never child GN steps."""
    costs=[]
    def timed(name,fn):
        w=time.perf_counter();c=time.process_time();out=fn()
        costs.append(dict(feature=name,wall_s=time.perf_counter()-w,cpu_s=time.process_time()-c,
                          scope='PROJECTED_PUBLIC_WARM',new_full_rhs=0))
        return out
    c=ctx.atoms;u=orth(c[:,ctx.selected_ids]);lc=timed('LC',lambda:ctx.L@c)
    sc=timed('SC',lambda:ctx.S@c);lu=ctx.L@u
    a=u.conj().T@lu;sv=la.svdvals(a)
    if sv[-1]/sv[0]<=1e-10:raise ValueError('receiver core unstable')
    cross,back=timed('bidirectional_retained_coupling',lambda:(u.conj().T@lc,c.conj().T@lu))
    coeff=timed('retained_LU_solve',lambda:la.solve(a,cross,assume_a='gen'))
    cbar=c-u@coeff;lcbar=lc-lu@coeff
    y=timed('dressed_observation',lambda:sc-(ctx.S@u)@coeff)
    injection=timed('receiver_direction_projected_injection',lambda:projected_source(ctx,u,ctx.x))
    n=timed('N_single_direction_sketch',lambda:injection@c.conj()) # (P,Natom)
    gamma=timed('Schur_gamma',lambda:np.einsum('ni,ni->i',c.conj(),lcbar))
    norm,ynorm=timed('candidate_and_observation_norms',lambda:(np.linalg.norm(c,axis=0),np.linalg.norm(y,axis=0)))
    def normalized_gamma():
        gden=norm*np.linalg.norm(lcbar,axis=0)
        gnorm=np.divide(np.abs(gamma),gden,out=np.zeros(len(gamma)),where=gden>0)
        # Addition proxy only, not actual swap feasibility.
        stable=(np.linalg.norm(cbar,axis=0)>1e-10*norm)&(gnorm>1e-10)
        inverse=np.divide(1.,gamma,out=np.zeros_like(gamma),where=stable)
        return gnorm,stable,inverse
    gnorm,stable,inverse=timed('Gamma_normalization_and_proxy_mask',normalized_gamma)
    w=timed('signed_transfer_vectors',lambda:pack_data(y[None,:,:]*n[:,None,:]*inverse[None,None,:]))
    ua=timed('public_anchor_response_receiver_step',lambda:ctx.r+reduced_response(ctx,ctx.anchor,ctx.x))
    signed=timed('signed_transfer_alignment',lambda:-(ua@w)-.5*np.sum(w*w,axis=0))
    o=timed('O_normalization',lambda:np.divide(ynorm**2,norm**2,out=np.zeros_like(norm),where=norm>0))
    p=timed('P_normalization',lambda:np.divide(np.sum(abs(cross)**2,axis=0)+np.sum(abs(back)**2,axis=1),norm**2,
                out=np.zeros_like(norm),where=norm>0))
    den=(np.linalg.norm(ctx.x)**2)*norm**2
    m=timed('M_normalization',lambda:np.divide(np.sum(abs(n)**2,axis=0),den,out=np.zeros_like(norm),where=den>0))
    return dict(O=o,P=p,M=m,signed=signed,Y=y,gamma_normalized=gnorm,
                proxy_stable=stable,core_reciprocal_condition=float(sv[-1]/sv[0]),costs=costs)


def balanced(families,selected,count=12):
    order=list(dict.fromkeys(families));by={f:[i for i,x in enumerate(families) if x==f and i not in selected] for f in order}
    out=[];position=0
    while len(out)<count:
        added=False
        for f in order:
            if position<len(by[f]):out.append(by[f][position]);added=True
            if len(out)==count:break
        if not added:break
        position+=1
    return out


def propose(ctx,f,method,seed=None,penalty=.25):
    candidates=[i for i in range(ctx.atoms.shape[1]) if i not in ctx.selected_ids]
    if method=='B0':return balanced(ctx.families,ctx.selected_ids)
    if method=='B1':
        base=balanced(ctx.families,ctx.selected_ids);rng=np.random.default_rng(seed)
        # Canonical original-ID order makes permutation of array storage immaterial.
        counts={family:sum(ctx.families[i]==family for i in base) for family in dict.fromkeys(ctx.families)}
        chosen={family:list(rng.choice([i for i in candidates if ctx.families[i]==family],size=n,replace=False))
                for family,n in counts.items() if n}
        out=[]
        for i in base:out.append(int(chosen[ctx.families[i]].pop(0)))
        return out
    def tied_scores(values):
        # Frozen numerical tie convention, relative to this query's score scale.
        scale=max(float(np.max(np.abs(values[candidates]))),1e-300)
        return np.rint(values/(scale*1e-12))
    ranks={axis:tied_scores(f[axis]) for axis in ['O','P','M']}
    orders={axis:sorted(candidates,key=lambda i:(-float(ranks[axis][i]),i)) for axis in ['O','P','M']}
    if method=='B2':
        out=[];pos={axis:0 for axis in orders}
        while len(out)<min(12,len(candidates)):
            for axis in ['O','P','M']:
                while pos[axis]<len(candidates) and orders[axis][pos[axis]] in out:pos[axis]+=1
                if pos[axis]<len(candidates):out.append(orders[axis][pos[axis]]);pos[axis]+=1
                if len(out)==min(12,len(candidates)):break
        return out
    if method!='B3':raise ValueError(method)
    score=tied_scores(f['signed']);percentile={i:float(np.mean(score[candidates]<score[i])) for i in candidates}
    y=f['Y'];yn=np.linalg.norm(y,axis=0);out=[]
    def overlap(i,j):
        den=yn[i]*yn[j]
        return float(abs(np.vdot(y[:,i],y[:,j]))**2/den**2) if den else 0.
    while len(out)<min(12,len(candidates)):
        def utility(i):return percentile[i]-penalty*max((overlap(i,j) for j in out),default=0.)
        out.append(min((i for i in candidates if i not in out),key=lambda i:(-round(utility(i),12),i)))
    return out


def source_sketch(ctx,d,rank,seed):
    """Physical-current embedding of PROJECTED injection, not full injection.

    Common deterministic physical-row encoder; no material coordinate index
    is assigned cross-object semantic meaning. Fixed-state map is real linear.
    """
    b=projected_source(ctx,ctx.anchor,d)
    e=np.random.default_rng(seed).standard_normal((rank,ctx.physical_embedding.shape[0]))
    eq=(e@ctx.physical_embedding)/np.sqrt(ctx.physical_embedding.shape[0])
    z=np.einsum('sn,pn...->ps...',eq,b)
    return pack_data(z)
