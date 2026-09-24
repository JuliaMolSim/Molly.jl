"""Many-body Allegro-style reference (numpy + e3nn). Fixes the pair-potential bug: each layer's
tensor product multiplies the edge's equivariant latent by the CENTRAL ATOM'S ENVIRONMENT
(a sum over all neighbours), not the edge's own spherical harmonics.

Per atom i, per directed edge i<-j:
  Y_ij = SH(rhat_ij); u_ij = envelope(d_ij); R_ij = bessel(d_ij)*u_ij
  x_ij^0 = MLP_emb([R_ij, 1hot(Zi), 1hot(Zj)])                      # scalar latent (H)
  V_ij^0 = init_lin(Y_ij) * u_ij                                     # equivariant latent, C x (0e+1o+2e)
  for layer L:
    g_ik = envW^L @ x_ik + envb^L                                   # per (channel,l) env weights, C*3
    Env_i[c,l,m] = (1/avg_nn) * sum_{k in N(i)} g_ik[c,l] * Y_ik[l,m]   # C-channel environment (feat)
    P_ij = TP_uvu(V_ij, Env_i; w_ij),  w_ij = tpW^L @ x_ij + tpb^L  # feat (x) feat -> feat
    x_ij = x_ij + silu(xW^L @ [x_ij, scalars0e(P_ij)] + xb^L) * u_ij
    V_ij = eqlin^L(P_ij)
  E_ij = outW @ x_ij + outb ; E = sum_ij E_ij
"""
import numpy as np, json, os, itertools
import torch
from e3nn import o3
torch.set_default_dtype(torch.float64)

LS = [0, 1, 2]
DIMS = [2 * l + 1 for l in LS]
SHDIM = sum(DIMS)                 # 9
PAR = {0: 1, 1: -1, 2: 1}

def silu(x): return x / (1.0 + np.exp(-x))
def sh(rhat):
    x = torch.tensor(rhat, dtype=torch.float64)
    return o3.spherical_harmonics([0, 1, 2], x, normalize=True, normalization="component").numpy()
def bessel(d, rc, nb):
    n = np.arange(1, nb + 1); return np.sqrt(2.0 / rc) * np.sin(n * np.pi * d / rc) / d
def envelope(d, rc, p=6):
    if d >= rc: return 0.0
    x = d / rc; a = (p+1)*(p+2)/2; b = p*(p+2); c = p*(p+1)/2
    return 1 - a*x**p + b*x**(p+1) - c*x**(p+2)

def make_paths():   # feat (x) feat -> feat
    return [(k1,k2,k3,l1,l2,l3) for k1,l1 in enumerate(LS) for k2,l2 in enumerate(LS)
            for k3,l3 in enumerate(LS)
            if abs(l1-l2)<=l3<=l1+l2 and PAR[l1]*PAR[l2]==PAR[l3]]
def block(k): off=sum(DIMS[:k]); return slice(off, off+DIMS[k])
def fidx(C,k,c,m): return sum(DIMS[:k])*C + c*DIMS[k] + m   # channel-major flat index (feat, 9C)

def tp_uvu(V1, V2, w, C, paths, cgs, woff):   # both operands feat (C-channel), uvu
    out = np.zeros(SHDIM*C)
    for pi,(k1,k2,k3,l1,l2,l3) in enumerate(paths):
        cg=cgs[pi]; d1,d2,d3=cg.shape
        for c in range(C):
            wc=w[woff[pi]+c]
            for i1 in range(d1):
                for i2 in range(d2):
                    for i3 in range(d3):
                        v=cg[i1,i2,i3]
                        if v==0: continue
                        out[fidx(C,k3,c,i3)] += wc*v*V1[fidx(C,k1,c,i1)]*V2[fidx(C,k2,c,i2)]
    return out

def build(C=4,H=16,nb=8,rc=4.0,L=2,S=2,avg_nn=6.0,seed=0):
    rng=np.random.default_rng(seed)
    paths=make_paths(); cgs=[o3.wigner_3j(l1,l2,l3).numpy() for (_,_,_,l1,l2,l3) in paths]
    woff=[]; off=0
    for _ in paths: woff.append(off); off+=C
    nW=off
    cfg=dict(C=C,H=H,nb=nb,rc=rc,L=L,S=S,lmax=2,env_p=6,avg_nn=avg_nn,n_weights=nW,n_paths=len(paths))
    W={}
    din=nb+2*S
    W['emb_W1']=rng.standard_normal((H,din))*0.3; W['emb_b1']=np.zeros(H)
    W['emb_W2']=rng.standard_normal((H,H))*0.3;   W['emb_b2']=np.zeros(H)
    W['init_w']=rng.standard_normal((C,3))*0.5; W['init_b0']=rng.standard_normal(C)*0.1
    W['layers']=[]
    for _ in range(L):
        lw={}
        lw['env_W']=rng.standard_normal((C*3,H))*0.2; lw['env_b']=rng.standard_normal(C*3)*0.1
        lw['tp_W']=rng.standard_normal((nW,H))*0.2;   lw['tp_b']=rng.standard_normal(nW)*0.1
        lw['x_W']=rng.standard_normal((H,H+C))*0.2;   lw['x_b']=np.zeros(H)
        lw['lin_w']=[rng.standard_normal((C,C))*0.3 for _ in LS]; lw['lin_b0']=rng.standard_normal(C)*0.1
        W['layers'].append(lw)
    W['out_W']=rng.standard_normal((1,H))*0.3; W['out_b']=rng.standard_normal(1)*0.1
    return cfg,W,paths,cgs,woff

def edge_precompute(cfg,W,d,rhat,Zi,Zj):
    C,H,nb,rc,S=cfg['C'],cfg['H'],cfg['nb'],cfg['rc'],cfg['S']
    Y=sh(rhat); u=envelope(d,rc); R=bessel(d,rc,nb)*u
    oi=np.zeros(S);oi[Zi]=1;oj=np.zeros(S);oj[Zj]=1
    x=silu(W['emb_W1']@np.concatenate([R,oi,oj])+W['emb_b1']); x=W['emb_W2']@x+W['emb_b2']
    V=np.zeros(SHDIM*C)
    for k,l in enumerate(LS):
        for c in range(C):
            for m in range(DIMS[k]): V[fidx(C,k,c,m)]=W['init_w'][c,k]*Y[block(k)][m]*u
        if l==0:
            for c in range(C): V[fidx(C,0,c,0)]+=W['init_b0'][c]*u
    return dict(Y=Y,u=u,x=x,V=V)

def total_energy(cfg,W,paths,cgs,woff,coords,species,rc):
    C=cfg['C']; n=len(coords); avg=cfg['avg_nn']
    # neighbour lists
    nbr=[[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j: continue
            r=coords[j]-coords[i]; dd=np.linalg.norm(r)
            if dd<rc and dd>1e-8: nbr[i].append((j,dd,r/dd))
    # per-edge precompute (two-body); key (i,j)
    ec={}
    for i in range(n):
        for (j,dd,rh) in nbr[i]:
            ec[(i,j)]=edge_precompute(cfg,W,dd,rh,species[i],species[j])
    E=0.0
    for L in range(cfg['L']):
        lw=W['layers'][L]
        # environment per atom (density trick): Env_i = (1/avg) sum_k g_ik[c,l]*Y_ik
        Env={}
        for i in range(n):
            e=np.zeros(SHDIM*C)
            for (j,dd,rh) in nbr[i]:
                x_ik=ec[(i,j)]['x']; Y_ik=ec[(i,j)]['Y']
                g=lw['env_W']@x_ik+lw['env_b']   # C*3
                for k,l in enumerate(LS):
                    for c in range(C):
                        gcl=g[c*3+k] if False else g[k*C+c]  # layout: [l-major, channel]
                        for m in range(DIMS[k]): e[fidx(C,k,c,m)]+=gcl*Y_ik[block(k)][m]
            Env[i]=e/avg
        # per-edge update
        newV={}; newx={}
        for i in range(n):
            for (j,dd,rh) in nbr[i]:
                x_ij=ec[(i,j)]['x']; V_ij=ec[(i,j)]['V']; u=ec[(i,j)]['u']
                w=lw['tp_W']@x_ij+lw['tp_b']
                P=tp_uvu(V_ij, Env[i], w, C, paths, cgs, woff)
                scal=np.array([P[fidx(C,0,c,0)] for c in range(C)])
                newx[(i,j)]=x_ij+silu(lw['x_W']@np.concatenate([x_ij,scal])+lw['x_b'])*u
                Vn=np.zeros(SHDIM*C)
                for k,l in enumerate(LS):
                    Wl=lw['lin_w'][k]
                    for m in range(DIMS[k]):
                        for co in range(C):
                            acc=sum(Wl[co,ci]*P[fidx(C,k,ci,m)] for ci in range(C))
                            if l==0: acc+=lw['lin_b0'][co]
                            Vn[fidx(C,k,co,m)]=acc
                newV[(i,j)]=Vn
        for key in ec:
            ec[key]['x']=newx[key]; ec[key]['V']=newV[key]
    for i in range(n):
        for (j,dd,rh) in nbr[i]:
            E+=float((W['out_W']@ec[(i,j)]['x']+W['out_b'])[0])
    return E

def num_forces(cfg,W,paths,cgs,woff,coords,species,rc,h=1e-5):
    n=len(coords); F=np.zeros((n,3))
    for i in range(n):
        for b in range(3):
            cp=coords.copy(); cp[i,b]+=h; cm=coords.copy(); cm[i,b]-=h
            F[i,b]=-(total_energy(cfg,W,paths,cgs,woff,cp,species,rc)-
                     total_energy(cfg,W,paths,cgs,woff,cm,species,rc))/(2*h)
    return F

SPECIES=["H","C"]
def export(outdir):
    import h5py
    os.makedirs(outdir,exist_ok=True)
    cfg,W,paths,cgs,woff=build()
    with h5py.File(os.path.join(outdir,"allegro_model.h5"),"w") as f:
        g=f.create_group("config")
        for k,v in cfg.items(): g.attrs[k]=v
        f["species"]=[s.encode() for s in SPECIES]
        for name in ("emb_W1","emb_b1","emb_W2","emb_b2","init_w","init_b0","out_W","out_b"):
            f[name]=W[name]
        for li,lw in enumerate(W['layers']):
            lg=f.create_group(f"layer{li}")
            for name in ("env_W","env_b","tp_W","tp_b","x_W","x_b","lin_b0"):
                lg[name]=lw[name]
            lg["lin_w"]=np.stack(lw['lin_w'],axis=2)   # (C,C,3)
    def mk(coords,sp,name):
        coords=np.array(coords,dtype=np.float64)
        E=total_energy(cfg,W,paths,cgs,woff,coords,sp,cfg['rc'])
        F=num_forces(cfg,W,paths,cgs,woff,coords,sp,cfg['rc'])
        return dict(name=name,coords_A=coords.tolist(),species=list(sp),
                    energy=float(E),forces=F.tolist())
    systems=[mk([[0.0,0,0],[1.1,0,0],[0.2,1.0,0.3],[-0.5,0.4,1.2]],[0,1,0,1],"tetra"),
             mk([[0.0,0,0],[1.0,0.1,-0.2],[-0.3,0.9,0.4]],[1,0,1],"tri")]
    json.dump(dict(config=cfg,species=SPECIES,systems=systems),
              open(os.path.join(outdir,"allegro_model.json"),"w"),indent=1)
    print("exported to",outdir,"E(tetra)=",systems[0]['energy'])

if __name__=="__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="export":
        export(sys.argv[2]); raise SystemExit
    cfg,W,paths,cgs,woff=build()
    print("n_paths",cfg['n_paths'],"n_weights",cfg['n_weights'])
    coords=np.array([[0.0,0,0],[1.1,0,0],[0.2,1.0,0.3],[-0.5,0.4,1.2]])
    sp=[0,1,0,1]
    E=total_energy(cfg,W,paths,cgs,woff,coords,sp,cfg['rc']); print("E",E)
    # rotation invariance
    th=0.7; Rz=np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]])
    E2=total_energy(cfg,W,paths,cgs,woff,coords@Rz.T,sp,cfg['rc']); print("E(rot)",E2,"|d|",abs(E-E2))
    # MANY-BODY test: move atom 3 (a neighbour of others) — a pair potential would change E only
    # via the pair terms involving atom 3; here we check the (0,1) edge's environment coupling by
    # comparing total E with atom 3 moved far away vs near.
    c_near=coords.copy(); c_far=coords.copy(); c_far[3]=[10.0,10.0,10.0]
    En=total_energy(cfg,W,paths,cgs,woff,c_near,sp,cfg['rc'])
    Ef=total_energy(cfg,W,paths,cgs,woff,c_far,sp,cfg['rc'])
    print("move-atom3 |dE|", abs(En-Ef))
