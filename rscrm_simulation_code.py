import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from dataclasses import dataclass
import pandas as pd

RNG = np.random.default_rng(20260428)
GROUPS = ["ME", "E", "I", "N"]
N_BY_GROUP = {"ME": 7, "E": 6, "I": 6, "N": 18}
P = 39
K = 2
MODULES = {
    "SM1": np.arange(0, 20),
    "SM2": np.arange(20, 32),
    "SM3": np.arange(32, 39),
}

def make_lambda(boost=None):
    Lam = np.zeros((P,K))
    Lam[MODULES['SM1']] = np.array([0.55,0.25])
    Lam[MODULES['SM2']] = np.array([1.00,0.10])
    Lam[MODULES['SM3']] = np.array([0.10,1.00])
    if boost:
        for idx, vec in boost.items():
            Lam[idx] = np.array(vec)
    # small gene-specific jitter for identifiability
    rng = np.random.default_rng(123)
    Lam += rng.normal(0,0.03,Lam.shape)
    return Lam

def sym_sqrtm(S, inv=False):
    vals, vecs = eigh((S+S.T)/2)
    vals = np.maximum(vals, 1e-8)
    if inv:
        vals = 1/np.sqrt(vals)
    else:
        vals = np.sqrt(vals)
    return (vecs * vals) @ vecs.T

def shrink_cov(Z, alpha=0.05, eps=1e-4):
    Z = np.asarray(Z)
    if len(Z) <= 1:
        S = np.eye(Z.shape[1]) * eps
    else:
        S = np.cov(Z, rowvar=False)
    S = np.atleast_2d(S)
    k=S.shape[0]
    return (1-alpha)*S + alpha*np.trace(S)/k*np.eye(k) + eps*np.eye(k)

def fit_affine(X,Y,lam=0.05):
    X=np.asarray(X); Y=np.asarray(Y)
    mx=X.mean(axis=0, keepdims=True); my=Y.mean(axis=0, keepdims=True)
    Xc=X-mx; Yc=Y-my
    k=X.shape[1]
    A = np.linalg.solve(Xc.T@Xc + lam*np.eye(k), Xc.T@Yc)
    b = my - mx@A
    return A,b.ravel()

def predict_affine(X,A,b):
    return X@A + b

def fit_multi(X1,X2,Y,lam=0.05):
    X=np.hstack([X1,X2])
    mx=X.mean(axis=0, keepdims=True); my=Y.mean(axis=0, keepdims=True)
    Xc=X-mx; Yc=Y-my
    d=X.shape[1]
    B = np.linalg.solve(Xc.T@Xc + lam*np.eye(d), Xc.T@Yc)
    c = my - mx@B
    return B,c.ravel()

def estimate_scores_from_X(X_by_group, k=2):
    # X_by_group group -> array n x 3 x p
    allX = np.vstack([X_by_group[g].reshape(-1, X_by_group[g].shape[-1]) for g in GROUPS])
    Xs = (allX - allX.mean(axis=0)) / (allX.std(axis=0)+1e-8)
    pca = PCA(n_components=k, random_state=0)
    allZ = pca.fit_transform(Xs)
    Z_by_group = {}
    start=0
    for g in GROUPS:
        n = X_by_group[g].shape[0]
        Z_by_group[g] = allZ[start:start+n*3].reshape(n,3,k)
        start += n*3
    return Z_by_group

def compute_defects(Z_by_group, lam=0.05):
    out={}
    params={}
    for g,Z in Z_by_group.items():
        A12,b12=fit_affine(Z[:,0],Z[:,1],lam)
        A23,b23=fit_affine(Z[:,1],Z[:,2],lam)
        A13,b13=fit_affine(Z[:,0],Z[:,2],lam)
        out[g]=norm(A13 - A12@A23, 'fro')
        params[g]={(0,1):(A12,b12),(1,2):(A23,b23),(0,2):(A13,b13)}
    return out, params

def loocv_metrics(Z_by_group, lam=0.05):
    rows=[]
    for g,Z in Z_by_group.items():
        n=Z.shape[0]
        mse_dir=[]; mse_comp=[]; mse_multi=[]
        for i in range(n):
            idx=np.arange(n)!=i
            A12,b12=fit_affine(Z[idx,0],Z[idx,1],lam)
            A23,b23=fit_affine(Z[idx,1],Z[idx,2],lam)
            A13,b13=fit_affine(Z[idx,0],Z[idx,2],lam)
            B,c=fit_multi(Z[idx,0],Z[idx,1],Z[idx,2],lam)
            pred_d=predict_affine(Z[i:i+1,0],A13,b13)[0]
            pred_c=predict_affine(predict_affine(Z[i:i+1,0],A12,b12),A23,b23)[0]
            pred_m=np.hstack([Z[i,0],Z[i,1]])@B + c
            y=Z[i,2]
            mse_dir.append(np.mean((y-pred_d)**2))
            mse_comp.append(np.mean((y-pred_c)**2))
            mse_multi.append(np.mean((y-pred_m)**2))
        md=np.mean(mse_dir); mc=np.mean(mse_comp); mm=np.mean(mse_multi)
        rows.append({"group":g,"direct":md,"composed":mc,"multi":mm,"delta":md-mc})
    return rows

def alignment(mu_g,Sg,mu_h,Sh):
    Sg_half=sym_sqrtm(Sg); Sg_invhalf=sym_sqrtm(Sg, inv=True)
    middle = sym_sqrtm(Sg_half @ Sh @ Sg_half)
    B = Sg_invhalf @ middle @ Sg_invhalf
    c = -mu_g @ B + mu_h
    return B,c

def nat_def_pair(Z_by_group, params, g,h, interval):
    r,s = interval
    Zg = Z_by_group[g][:,r]
    Zh = Z_by_group[h][:,r]
    Zg_s=Z_by_group[g][:,s]
    Zh_s=Z_by_group[h][:,s]
    mu_gr = Zg.mean(axis=0); mu_hr=Zh.mean(axis=0)
    S_gr=shrink_cov(Zg); S_hr=shrink_cov(Zh)
    mu_gs=Zg_s.mean(axis=0); mu_hs=Zh_s.mean(axis=0)
    S_gs=shrink_cov(Zg_s); S_hs=shrink_cov(Zh_s)
    B_r,c_r=alignment(mu_gr,S_gr,mu_hr,S_hr)
    B_s,c_s=alignment(mu_gs,S_gs,mu_hs,S_hs)
    A_g,b_g=params[g][interval]
    A_h,b_h=params[h][interval]
    M=A_g@B_s - B_r@A_h
    d=b_g@B_s + c_s - c_r@A_h - b_h
    val=np.trace(M.T@S_gr@M) + norm(mu_gr@M + d)**2
    return float(np.real(val))

def sym_nat(Z_by_group, params, g,h, interval):
    return 0.5*(nat_def_pair(Z_by_group,params,g,h,interval)+nat_def_pair(Z_by_group,params,h,g,interval))

def natural_components(Z_by_group, params, g="ME", h="N"):
    E=sym_nat(Z_by_group,params,g,h,(0,1))
    L=sym_nat(Z_by_group,params,g,h,(1,2))
    return E,L

def simulate_X_scenario(scenario, rng, n_by_group=N_BY_GROUP, sigma_z=0.15, sigma_x=0.25, hetero=0.0):
    # base matrices
    A12_base=np.array([[0.70,0.10],[0.00,0.80]])
    A23_base=np.array([[0.80,0.00],[0.05,0.75]])
    b12_base=np.array([0.20,0.00]); b23_base=np.array([0.00,0.15])
    # group-specific base means
    mu_base={g:np.array([0.0,0.0]) for g in GROUPS}
    Sigma=np.array([[1.0,0.2],[0.2,0.8]])
    A12={g:A12_base.copy() for g in GROUPS}; A23={g:A23_base.copy() for g in GROUPS}
    b12={g:b12_base.copy() for g in GROUPS}; b23={g:b23_base.copy() for g in GROUPS}
    boost={}
    if scenario=="composable":
        pass
    elif scenario=="intermediate":
        # ME has clear staged drift, N weak/no stage, E/I less structured
        A12["ME"]=np.array([[0.55,0.35],[-0.05,0.80]])
        A23["ME"]=np.array([[0.75,-0.15],[0.20,0.70]])
        b12["ME"]=np.array([0.80,0.10]); b23["ME"]=np.array([0.20,0.70])
        A12["N"]=np.array([[0.90,0.02],[0.00,0.88]])
        A23["N"]=np.array([[0.90,0.01],[0.02,0.90]])
        b12["N"]=np.array([0.02,0.00]); b23["N"]=np.array([0.00,0.02])
        # noise moderate
    elif scenario=="early":
        A12["ME"]=np.array([[0.45,0.45],[-0.05,0.78]])
        b12["ME"]=np.array([0.80,0.10])
        A12["N"]=np.array([[0.95,0.00],[0.00,0.90]])
        b12["N"]=np.array([0.00,0.00])
        A23["ME"]=A23_base.copy(); A23["N"]=A23_base.copy()
        b23["ME"]=b23_base.copy(); b23["N"]=b23_base.copy()
    elif scenario=="late":
        A12["ME"]=A12_base.copy(); A12["N"]=A12_base.copy()
        b12["ME"]=b12_base.copy(); b12["N"]=b12_base.copy()
        A23["ME"]=np.array([[0.55,-0.20],[0.30,0.65]])
        b23["ME"]=np.array([0.15,0.85])
        A23["N"]=np.array([[0.95,0.00],[0.00,0.92]])
        b23["N"]=np.array([0.00,0.00])
    elif scenario=="drivers":
        # choose sparse, non-redundant drivers so leave-one-gene deletion is identifiable.
        driver_mid=MODULES['SM2'][:6]
        driver_late=MODULES['SM3'][:6]
        driver_str=MODULES['SM1'][:2]
        for idx in driver_mid: boost[int(idx)]=(16.0,0.05)
        for idx in driver_late: boost[int(idx)]=(0.05,16.0)
        for idx in driver_str: boost[int(idx)]=(11.0,11.0)
        A12["ME"]=np.array([[0.50,0.30],[-0.05,0.82]])
        b12["ME"]=np.array([0.95,0.10])
        A23["ME"]=np.array([[0.72,-0.10],[0.25,0.68]])
        b23["ME"]=np.array([0.10,0.90])
        A12["N"]=np.array([[0.90,0.00],[0.00,0.88]])
        A23["N"]=np.array([[0.92,0.01],[0.00,0.90]])
        b12["N"]=np.array([0.00,0.00]); b23["N"]=np.array([0.00,0.00])
    else:
        raise ValueError(scenario)
    Lam=make_lambda(boost)
    if scenario == 'drivers':
        Lam = np.random.default_rng(321).normal(0,0.05,(P,K))
        driver_mid=MODULES['SM2'][:6]
        driver_late=MODULES['SM3'][:6]
        driver_str=MODULES['SM1'][:2]
        for idx in driver_mid: Lam[int(idx)] = np.array([16.0,0.05])
        for idx in driver_late: Lam[int(idx)] = np.array([0.05,16.0])
        for idx in driver_str: Lam[int(idx)] = np.array([11.0,11.0])
    X_by_group={}
    Ztrue={}
    for g in GROUPS:
        n=n_by_group[g]
        z1=rng.multivariate_normal(mu_base[g],Sigma,size=n)
        # individual random slopes if hetero
        A12_i=A12[g]; A23_i=A23[g]
        if hetero>0:
            # handled per subject below
            z2=[]; z3=[]
            for i in range(n):
                Ai12=A12[g]+rng.normal(0,hetero,A12[g].shape)
                Ai23=A23[g]+rng.normal(0,hetero,A23[g].shape)
                zz2=z1[i]@Ai12 + b12[g] + rng.normal(0,sigma_z,K)
                zz3=zz2@Ai23 + b23[g] + rng.normal(0,sigma_z,K)
                z2.append(zz2); z3.append(zz3)
            z2=np.vstack(z2); z3=np.vstack(z3)
        else:
            z2=z1@A12_i + b12[g] + rng.normal(0,sigma_z,(n,K))
            z3=z2@A23_i + b23[g] + rng.normal(0,sigma_z,(n,K))
        Z=np.stack([z1,z2,z3],axis=1)
        sigma_x_eff = sigma_x * 0.45 if scenario == 'drivers' else sigma_x
        X=Z@Lam.T + rng.normal(0,sigma_x_eff,(n,3,P))
        if scenario=='drivers':
            # Inject controlled gene-level effects so that leave-one-gene-out has a known target.
            mid = MODULES['SM2'][:6]
            late = MODULES['SM3'][:6]
            strat = MODULES['SM1'][:2]
            driver_idx = np.r_[mid, late, strat]
            background_idx = np.setdiff1d(np.arange(P), driver_idx)
            X[:,:,background_idx] = 0.0
            if g == 'ME':
                mid_w = np.linspace(1.00, 0.55, len(mid))
                late_w = np.linspace(1.00, 0.55, len(late))
                strat_w = np.linspace(1.00, 0.60, len(strat))
                X[:,1,mid] += 10.0 * mid_w     # intermediate-state genes at Day14
                X[:,2,mid] += 5.0 * mid_w      # persistence into Day30
                X[:,2,late] += 10.5 * late_w   # late-divergence genes at Day30
                X[:,:,strat] += (np.array([4.0, 5.8, 7.5])[:,None] * strat_w)[None,:,:]
            elif g == 'N':
                X[:,1,mid] += 0.1
                X[:,2,late] -= 3.0 * np.linspace(1.00, 0.55, len(late))
                X[:,:,strat] -= (np.array([1.5, 2.1, 2.8])[:,None] * np.linspace(1.00, 0.60, len(strat)))[None,:,:]
        X_by_group[g]=X
        Ztrue[g]=Z
    return X_by_group, {"J_mid": MODULES['SM2'][:6], "J_late": MODULES['SM3'][:6], "J_str": MODULES['SM1'][:2]}

def analyze_X(X_by_group, subset=None):
    if subset is not None:
        Xs={g:X_by_group[g][:,:,subset] for g in GROUPS}
    else:
        Xs=X_by_group
    Z=estimate_scores_from_X(Xs,K)
    defs,params=compute_defects(Z)
    loocv=loocv_metrics(Z)
    E,L=natural_components(Z,params,"ME","N")
    return Z,defs,params,loocv,E,L

def one_rep(scenario, rng, sigma_z=0.15, sigma_x=0.25, hetero=0.0):
    X, truth = simulate_X_scenario(scenario, rng, sigma_z=sigma_z, sigma_x=sigma_x, hetero=hetero)
    Z,defs,params,loocv,E,L=analyze_X(X)
    row={"scenario":scenario,"E_ME_N":E,"L_ME_N":L}
    for g in GROUPS:
        row[f"def_{g}"]=defs[g]
        lg=[x for x in loocv if x['group']==g][0]
        for key in ['direct','composed','multi','delta']:
            row[f"{key}_{g}"]=lg[key]
    return row, X, truth

def gene_scores(X):
    def scores_from_standardized_matrix(X_by_group, mask_gene=None):
        allX = np.vstack([X_by_group[g].reshape(-1, X_by_group[g].shape[-1]) for g in GROUPS])
        Xs = (allX - allX.mean(axis=0)) / (allX.std(axis=0)+1e-8)
        pca = PCA(n_components=K, random_state=0).fit(Xs)
        if mask_gene is not None:
            Xs = Xs.copy()
            Xs[:, mask_gene] = 0.0
        allZ = pca.transform(Xs)
        out = {}
        start = 0
        for g in GROUPS:
            n = X_by_group[g].shape[0]
            out[g] = allZ[start:start+n*3].reshape(n,3,K)
            start += n*3
        return out

    # Full model, then leave one gene out in the same latent reference space.
    Z=scores_from_standardized_matrix(X)
    defs,params=compute_defects(Z)
    loocv=loocv_metrics(Z)
    delta_full=[x for x in loocv if x['group']=='ME'][0]['delta']
    D_full=defs['ME']-defs['N']
    iscs=[]; fdcs=[]
    for j in range(P):
        try:
            Z_j=scores_from_standardized_matrix(X, mask_gene=j)
            defs_j,_=compute_defects(Z_j)
            loocv_j=loocv_metrics(Z_j)
            delta_j=[x for x in loocv_j if x['group']=='ME'][0]['delta']
            D_j=defs_j['ME']-defs_j['N']
            iscs.append(delta_full-delta_j)
            fdcs.append(D_full-D_j)
        except Exception:
            iscs.append(np.nan); fdcs.append(np.nan)
    return np.array(iscs), np.array(fdcs)

def run_monte_carlo(R=200, seed=20260428):
    rng=np.random.default_rng(seed)
    rows=[]
    for sc in ['composable','intermediate','early','late']:
        for _ in range(R):
            row,_,_=one_rep(sc,rng)
            rows.append(row)
    # stability variants for intermediate scenario
    for label, params in [('small_sample_high_noise', dict(sigma_z=0.30,sigma_x=0.50,hetero=0.0)),('heterogeneous', dict(sigma_z=0.20,sigma_x=0.35,hetero=0.10))]:
        for _ in range(R):
            row,_,_=one_rep('intermediate',rng,**params)
            row['scenario']=label
            rows.append(row)
    return pd.DataFrame(rows)

def run_driver_reps(R=100, seed=20260429):
    rng=np.random.default_rng(seed)
    rows=[]
    for rep in range(R):
        row,X,truth=one_rep('drivers',rng)
        iscs,fdcs=gene_scores(X)
        J_mid=set(map(int,truth['J_mid'])); J_late=set(map(int,truth['J_late'])); J_str=set(map(int,truth['J_str']))
        iscs_rank_score=np.abs(np.nan_to_num(iscs, nan=0.0))
        fdcs_rank_score=np.abs(np.nan_to_num(fdcs, nan=0.0))
        top_iscs=set(np.argsort(iscs_rank_score)[::-1][:10])
        top_fdcs=set(np.argsort(fdcs_rank_score)[::-1][:10])
        iscs_order=np.argsort(iscs_rank_score)[::-1]
        fdcs_order=np.argsort(fdcs_rank_score)[::-1]
        rows.append({
            'hit_mid_iscs_top10': len(top_iscs & J_mid)/len(J_mid),
            'hit_late_str_fdcs_top10': len(top_fdcs & (J_late|J_str))/len(J_late|J_str),
            'mean_rank_mid_iscs': np.mean([np.where(iscs_order==j)[0][0]+1 for j in J_mid]),
            'mean_rank_late_str_fdcs': np.mean([np.where(fdcs_order==j)[0][0]+1 for j in (J_late|J_str)]),
            'mean_ISCS_mid': np.nanmean(iscs_rank_score[list(J_mid)]),
            'mean_ISCS_other': np.nanmean([iscs_rank_score[j] for j in range(P) if j not in J_mid]),
            'mean_FDCS_driver': np.nanmean(fdcs_rank_score[list(J_late|J_str)]),
            'mean_FDCS_other': np.nanmean([fdcs_rank_score[j] for j in range(P) if j not in (J_late|J_str)]),
        })
    return pd.DataFrame(rows)

def main():
    """Run the author-provided simulations with portable output paths."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="RS-CRM simulation experiments")
    parser.add_argument("--replications", type=int, default=80,
                        help="Replications per main scenario (original default: 80)")
    parser.add_argument("--driver-replications", type=int, default=20,
                        help="Driver-recovery replications (original default: 20)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/simulation"))
    args = parser.parse_args()
    if args.replications < 1 or args.driver_replications < 1:
        parser.error("Replication counts must be positive integers.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = run_monte_carlo(R=args.replications)
    df.to_csv(args.output_dir / "rscrm_simulation_replicates.csv", index=False)
    metrics = ['def_ME', 'def_E', 'def_I', 'def_N', 'delta_ME',
               'direct_ME', 'composed_ME', 'multi_ME', 'E_ME_N', 'L_ME_N']
    summary = df.groupby('scenario')[metrics].agg(['mean', 'std'])
    summary.to_csv(args.output_dir / "rscrm_simulation_summary.csv")
    print(summary.round(4))
    driver = run_driver_reps(R=args.driver_replications)
    driver.to_csv(args.output_dir / "rscrm_driver_replicates.csv", index=False)
    driver_summary = driver.agg(['mean', 'std']).T
    driver_summary.to_csv(args.output_dir / "rscrm_driver_summary.csv")
    print(driver_summary.round(4))
    print("Output directory:", args.output_dir.resolve())


if __name__ == '__main__':
    main()
