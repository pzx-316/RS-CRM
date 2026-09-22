
import os
import re
import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns


SHEET_MAP = {"0天37人数据": 0, "14天37人数据": 14, "30天37人数据": 30}
GROUP_ORDER = ["ME", "E", "I", "N"]
GROUPS = (["ME"] * 7) + (["E"] * 6) + (["I"] * 6) + (["N"] * 18)

def preprocess_pca(X, gene_idx=None, k=2):
    if gene_idx is None:
        gene_idx = np.arange(X.shape[2])
    Xsub = X[:, :, gene_idx]
    Xlog = np.log1p(Xsub.reshape(-1, len(gene_idx)))
    mu = Xlog.mean(axis=0)
    sd = Xlog.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Xstd = ((Xlog - mu) / sd).reshape(Xsub.shape)
    pca = PCA(n_components=min(k, len(gene_idx)))
    Z = pca.fit_transform(Xstd.reshape(-1, len(gene_idx))).reshape(Xsub.shape[0], Xsub.shape[1], -1)
    return Xstd, Z, {"mu": mu, "sd": sd, "pca": pca, "gene_idx": np.array(gene_idx)}

def cov_shrink(Z, eps=1e-4, alpha=0.1):
    if Z.ndim == 1:
        Z = Z[:, None]
    S = np.cov(Z.T, bias=False) if Z.shape[0] > 1 else np.eye(Z.shape[1]) * eps
    if np.ndim(S) == 0:
        S = np.array([[float(S)]])
    k = S.shape[0]
    target = np.eye(k) * np.trace(S) / k
    S = (1 - alpha) * S + alpha * target
    S = S + eps * np.eye(k)
    return S

def mat_sqrt_psd(M):
    vals, vecs = eigh(M)
    vals = np.clip(vals, 0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T

def mat_invsqrt_psd(M):
    vals, vecs = eigh(M)
    vals = np.clip(vals, 1e-10, None)
    return (vecs * (1 / np.sqrt(vals))) @ vecs.T

def gaussian_w2_sq(mu1, S1, mu2, S2):
    term1 = float(np.sum((mu1 - mu2) ** 2))
    S1h = mat_sqrt_psd(S1)
    middle = S1h @ S2 @ S1h
    term2 = np.trace(S1 + S2 - 2 * mat_sqrt_psd(middle))
    return float(np.real(term1 + term2))

def affine_transport(mu_g, S_g, mu_h, S_h):
    Sg_half = mat_sqrt_psd(S_g)
    Sg_invhalf = mat_invsqrt_psd(S_g)
    middle = Sg_half @ S_h @ Sg_half
    B = Sg_invhalf @ mat_sqrt_psd(middle) @ Sg_invhalf
    c = -mu_g @ B + mu_h
    return B, c

def fit_affine_ridge(X, Y, lam=1.0):
    X = np.asarray(X)
    Y = np.asarray(Y)
    xbar = X.mean(axis=0, keepdims=True)
    ybar = Y.mean(axis=0, keepdims=True)
    Xc = X - xbar
    Yc = Y - ybar
    k = X.shape[1]
    A = np.linalg.solve(Xc.T @ Xc + lam * np.eye(k), Xc.T @ Yc)
    b = (ybar - xbar @ A).reshape(-1)
    return A, b

def predict_affine(z, A, b):
    return z @ A + b

def fit_group_functors(Z, group_indices, lam=1.0):
    out = {}
    for g, idx in group_indices.items():
        Zg = Z[idx]
        A12, b12 = fit_affine_ridge(Zg[:, 0, :], Zg[:, 1, :], lam)
        A23, b23 = fit_affine_ridge(Zg[:, 1, :], Zg[:, 2, :], lam)
        A13, b13 = fit_affine_ridge(Zg[:, 0, :], Zg[:, 2, :], lam)
        mus = [Zg[:, r, :].mean(0) for r in range(3)]
        covs = [cov_shrink(Zg[:, r, :]) for r in range(3)]
        out[g] = {"A12": A12, "b12": b12, "A23": A23, "b23": b23, "A13": A13, "b13": b13, "mu": mus, "Sigma": covs, "Zg": Zg}
    return out

def within_defect(fit, g):
    return np.linalg.norm(fit[g]["A13"] - fit[g]["A12"] @ fit[g]["A23"], ord="fro")

def naturality_defect_pair(fit, g, h, stage="early"):
    if stage == "early":
        Ag, bg = fit[g]["A12"], fit[g]["b12"]
        Ah, bh = fit[h]["A12"], fit[h]["b12"]
        mug1, Sgg1 = fit[g]["mu"][0], fit[g]["Sigma"][0]
        muh1, Shh1 = fit[h]["mu"][0], fit[h]["Sigma"][0]
        mug2, Sgg2 = fit[g]["mu"][1], fit[g]["Sigma"][1]
        muh2, Shh2 = fit[h]["mu"][1], fit[h]["Sigma"][1]
        Br, cr = affine_transport(mug1, Sgg1, muh1, Shh1)
        Bs, cs = affine_transport(mug2, Sgg2, muh2, Shh2)
        mu0, S0 = mug1, Sgg1
    else:
        Ag, bg = fit[g]["A23"], fit[g]["b23"]
        Ah, bh = fit[h]["A23"], fit[h]["b23"]
        mug1, Sgg1 = fit[g]["mu"][1], fit[g]["Sigma"][1]
        muh1, Shh1 = fit[h]["mu"][1], fit[h]["Sigma"][1]
        mug2, Sgg2 = fit[g]["mu"][2], fit[g]["Sigma"][2]
        muh2, Shh2 = fit[h]["mu"][2], fit[h]["Sigma"][2]
        Br, cr = affine_transport(mug1, Sgg1, muh1, Shh1)
        Bs, cs = affine_transport(mug2, Sgg2, muh2, Shh2)
        mu0, S0 = mug1, Sgg1
    M = Ag @ Bs - Br @ Ah
    d = bg @ Bs + cs - cr @ Ah - bh
    val = np.trace(M.T @ S0 @ M) + float((mu0 @ M + d) @ (mu0 @ M + d).T)
    return float(val)

def symmetric_naturality(fit, g, h, stage="early"):
    return 0.5 * (naturality_defect_pair(fit, g, h, stage) + naturality_defect_pair(fit, h, g, stage))

def mean_abs_corr(data):
    C = np.corrcoef(data, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    return np.nanmean(np.abs(C[iu]))

def graph_laplacian(C):
    W = np.abs(C).copy()
    np.fill_diagonal(W, 0.0)
    D = np.diag(W.sum(axis=1))
    return D - W

def rewiring_metric(Xstd, gene_clusters, group_indices, g, h):
    def submodule_matrix(group, time_idx):
        idx = group_indices[group]
        Xgt = Xstd[idx, time_idx, :]
        feats = []
        for cl in sorted(set(gene_clusters)):
            cols = np.where(gene_clusters == cl)[0]
            Xsub = Xgt[:, cols]
            if Xsub.shape[1] == 1:
                score = Xsub[:, 0]
            else:
                score = PCA(n_components=1).fit_transform(Xsub).ravel()
            feats.append(score)
        M = np.column_stack(feats)
        C = np.corrcoef(M, rowvar=False)
        if np.ndim(C) == 0:
            C = np.array([[1.0]])
        return C
    Cg0 = submodule_matrix(g, 0)
    Ch0 = submodule_matrix(h, 0)
    Cg3 = submodule_matrix(g, 2)
    Ch3 = submodule_matrix(h, 2)
    return np.linalg.norm(graph_laplacian(Cg3) - graph_laplacian(Ch3), ord="fro") - np.linalg.norm(graph_laplacian(Cg0) - graph_laplacian(Ch0), ord="fro")

def loocv_group_models(Z, group_labels, patient_ids, groups_by_patient, lam=1.0):
    rec = []
    pats = np.array(patient_ids)
    for i, p in enumerate(pats):
        grp = groups_by_patient[p]
        train_idx = [j for j, pp in enumerate(pats) if j != i and groups_by_patient[pp] == grp]
        Zg = Z[train_idx]
        A12, b12 = fit_affine_ridge(Zg[:, 0, :], Zg[:, 1, :], lam)
        A23, b23 = fit_affine_ridge(Zg[:, 1, :], Zg[:, 2, :], lam)
        A13, b13 = fit_affine_ridge(Zg[:, 0, :], Zg[:, 2, :], lam)
        B, c = fit_affine_ridge(np.concatenate([Zg[:, 0, :], Zg[:, 1, :]], axis=1), Zg[:, 2, :], lam)
        z0, z1, z2 = Z[i, 0, :], Z[i, 1, :], Z[i, 2, :]
        pred_dir = predict_affine(z0[None, :], A13, b13)[0]
        pred_comp = predict_affine(predict_affine(z0[None, :], A12, b12), A23, b23)[0]
        pred_multi = predict_affine(np.concatenate([z0, z1])[None, :], B, c)[0]
        rec.append({
            "patient": p,
            "group": grp,
            "mse_dir": ((z2 - pred_dir) ** 2).mean(),
            "mse_comp": ((z2 - pred_comp) ** 2).mean(),
            "mse_multi": ((z2 - pred_multi) ** 2).mean(),
        })
    return pd.DataFrame(rec)

def ci(vals, a=0.025, b=0.975):
    vals = np.asarray(vals)
    return float(np.quantile(vals, a)), float(np.quantile(vals, b))

def main():
    """Run the original empirical workflow on a supplied study workbook."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="RS-CRM empirical analysis")
    parser.add_argument("--input", required=True, type=Path,
                        help="Prepared study workbook; see README for its required layout")
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Input workbook does not exist: " + str(args.input))
    INPUT_XLSX = str(args.input)
    OUTDIR = str(args.output_dir)
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) load workbook
    patient_data = {}
    genes = None
    for sheet, t in SHEET_MAP.items():
        raw = pd.read_excel(INPUT_XLSX, sheet_name=sheet, header=None)
        sample_ids = list(raw.iloc[0, 1:])
        gnames = list(raw.iloc[1:, 0])
        vals = raw.iloc[1:, 1:].apply(pd.to_numeric).to_numpy(dtype=float)
        if genes is None:
            genes = np.array(gnames)
        for sid, col, grp in zip(sample_ids, range(len(sample_ids)), GROUPS):
            base = re.sub(r"-(0|14|30)$", "", sid)
            patient_data.setdefault(base, {"group": grp, "samples": {}})
            patient_data[base]["samples"][t] = vals[:, col]

    patient_ids = sorted(patient_data.keys())
    times = [0, 14, 30]
    groups_by_patient = {p: patient_data[p]["group"] for p in patient_ids}
    group_indices = {g: [i for i, p in enumerate(patient_ids) if groups_by_patient[p] == g] for g in GROUP_ORDER}

    X = np.zeros((len(patient_ids), len(times), len(genes)))
    for i, p in enumerate(patient_ids):
        for ti, t in enumerate(times):
            X[i, ti, :] = patient_data[p]["samples"][t]

    # 2) preprocess + latent space
    Xstd, Z, prep = preprocess_pca(X, k=2)
    fit = fit_group_functors(Z, group_indices, lam=1.0)

    # 3) choose submodules using ME-Day30 correlation
    idx_me = group_indices["ME"]
    corr_me30 = np.corrcoef(Xstd[idx_me, 2, :], rowvar=False)
    dist_me = 1 - np.abs(corr_me30)
    # parsimonious choice q=3 (smallest informative solution close to best silhouette)
    gene_clusters = AgglomerativeClustering(n_clusters=3, metric="precomputed", linkage="average").fit_predict(dist_me)

    # 4) core summaries
    within_summary = []
    for g in GROUP_ORDER:
        within_summary.append({"group": g, "Def": within_defect(fit, g)})
    within_summary = pd.DataFrame(within_summary)

    # simple bootstrap CIs for within-group defect
    rng = np.random.default_rng(123)
    within_boot = {g: [] for g in GROUP_ORDER}
    pair_boot = {f"{g}-{h}": {"B": [], "E": [], "L": [], "R": []} for g, h in itertools.combinations(GROUP_ORDER, 2)}

    def bootstrap_run(B=300):
        for _ in range(B):
            sampled_indices = []
            local_idx = {}
            start = 0
            for g in GROUP_ORDER:
                idx = np.array(group_indices[g])
                samp = rng.choice(idx, size=len(idx), replace=True)
                sampled_indices.extend(samp.tolist())
                local_idx[g] = list(range(start, start + len(idx)))
                start += len(idx)
            Xb = X[sampled_indices, :, :]
            Xstd_b, Zb, _ = preprocess_pca(Xb, k=2)
            fitb = fit_group_functors(Zb, local_idx, lam=1.0)
            for g in GROUP_ORDER:
                within_boot[g].append(within_defect(fitb, g))
            for g, h in itertools.combinations(GROUP_ORDER, 2):
                pair = f"{g}-{h}"
                Bv = gaussian_w2_sq(fitb[g]["mu"][0], fitb[g]["Sigma"][0], fitb[h]["mu"][0], fitb[h]["Sigma"][0])
                Ev = symmetric_naturality(fitb, g, h, "early")
                Lv = symmetric_naturality(fitb, g, h, "late")
                Rv = rewiring_metric(Xstd_b, gene_clusters, local_idx, g, h)
                pair_boot[pair]["B"].append(Bv)
                pair_boot[pair]["E"].append(Ev)
                pair_boot[pair]["L"].append(Lv)
                pair_boot[pair]["R"].append(Rv)

    bootstrap_run(B=300)

    within_out = []
    for g in GROUP_ORDER:
        lo, hi = ci(within_boot[g])
        within_out.append({"group": g,
                           "Def": within_summary[within_summary.group == g]["Def"].iloc[0],
                           "boot_mean": float(np.mean(within_boot[g])),
                           "ci_low": lo, "ci_high": hi})
    within_out = pd.DataFrame(within_out)
    within_out.to_csv(f"{OUTDIR}/table_within_defect.csv", index=False)

    # 5) LOOCV
    loocv = loocv_group_models(Z, GROUP_ORDER, patient_ids, groups_by_patient, lam=1.0)
    delta_rows = []
    rng = np.random.default_rng(321)
    for g in GROUP_ORDER:
        sub = loocv[loocv.group == g].copy().reset_index(drop=True)
        deltas = (sub.mse_dir - sub.mse_comp).to_numpy()
        boots = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(1000)])
        delta_rows.append({
            "group": g,
            "mean_dir": sub.mse_dir.mean(),
            "mean_comp": sub.mse_comp.mean(),
            "mean_multi": sub.mse_multi.mean(),
            "delta_mean": deltas.mean(),
            "ci_low": np.quantile(boots, 0.025),
            "ci_high": np.quantile(boots, 0.975),
        })
    delta_summary = pd.DataFrame(delta_rows)
    delta_summary.to_csv(f"{OUTDIR}/table_loocv_delta.csv", index=False)

    # 6) pairwise structural divergence
    pair_rows = []
    for g, h in itertools.combinations(GROUP_ORDER, 2):
        Bv = gaussian_w2_sq(fit[g]["mu"][0], fit[g]["Sigma"][0], fit[h]["mu"][0], fit[h]["Sigma"][0])
        Ev = symmetric_naturality(fit, g, h, "early")
        Lv = symmetric_naturality(fit, g, h, "late")
        Rv = rewiring_metric(Xstd, gene_clusters, group_indices, g, h)
        pair_rows.append({"pair": f"{g}-{h}", "g": g, "h": h,
                          "B_baseline": Bv, "E_early": Ev, "L_late": Lv, "R_rewire": Rv})
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(f"{OUTDIR}/table_pairwise_structural_components.csv", index=False)

    pair_summary = pair_df.copy()
    for comp in ["B", "E", "L", "R"]:
        pair_summary[f"{comp}_boot_mean"] = [float(np.mean(pair_boot[p][comp])) for p in pair_summary["pair"]]
        pair_summary[f"{comp}_ci_low"] = [ci(pair_boot[p][comp])[0] for p in pair_summary["pair"]]
        pair_summary[f"{comp}_ci_high"] = [ci(pair_boot[p][comp])[1] for p in pair_summary["pair"]]
    pair_summary.to_csv(f"{OUTDIR}/table_pairwise_structural_components_bootstrap.csv", index=False)

    pair_score = pair_df.copy()
    pair_score["R_abs"] = pair_score["R_rewire"].abs()
    for col in ["B_baseline", "E_early", "L_late", "R_abs"]:
        pair_score[col + "_norm"] = (pair_score[col] - pair_score[col].min()) / (pair_score[col].max() - pair_score[col].min() + 1e-9)
    pair_score["S_total"] = pair_score[[c + "_norm" for c in ["B_baseline", "E_early", "L_late", "R_abs"]]].mean(axis=1)
    pair_score.to_csv(f"{OUTDIR}/table_pairwise_structural_scores.csv", index=False)

    # 7) submodule-level signals
    sub_rows = []
    for cl in sorted(set(gene_clusters)):
        gene_idx = np.where(gene_clusters == cl)[0]
        Xstd_sub, Zsub, _ = preprocess_pca(X, gene_idx=gene_idx, k=1)
        fit_sub = fit_group_functors(Zsub, group_indices, lam=1.0)
        E = symmetric_naturality(fit_sub, "ME", "N", "early")
        L = symmetric_naturality(fit_sub, "ME", "N", "late")
        coh_me0 = mean_abs_corr(Xstd_sub[group_indices["ME"], 0, :]) if len(gene_idx) > 1 else np.nan
        coh_me3 = mean_abs_corr(Xstd_sub[group_indices["ME"], 2, :]) if len(gene_idx) > 1 else np.nan
        delta_coh = coh_me3 - coh_me0 if len(gene_idx) > 1 else np.nan
        loocv_sub = loocv_group_models(Zsub, GROUP_ORDER, patient_ids, groups_by_patient, lam=1.0)
        delta_me = (loocv_sub.query("group=='ME'").mse_dir - loocv_sub.query("group=='ME'").mse_comp).mean()
        sub_rows.append({
            "submodule": int(cl) + 1,
            "n_genes": len(gene_idx),
            "genes": ", ".join(genes[gene_idx].tolist()),
            "ME_N_early": E,
            "ME_N_late": L,
            "ME_deltaMSE": delta_me,
            "ME_delta_coherence": delta_coh,
        })
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(f"{OUTDIR}/table_submodule_signals.csv", index=False)
    with open(f"{OUTDIR}/submodule_gene_membership.txt", "w", encoding="utf-8") as f:
        for _, row in sub_df.iterrows():
            f.write(f"Submodule {row['submodule']} ({row['n_genes']} genes): {row['genes']}\n")

    # 8) gene attribution
    delta_ME = (loocv.query("group=='ME'").mse_dir - loocv.query("group=='ME'").mse_comp).mean()
    fd_full = within_defect(fit, "ME") - within_defect(fit, "N")
    E_full = symmetric_naturality(fit, "ME", "N", "early")
    L_full = symmetric_naturality(fit, "ME", "N", "late")
    R_full = rewiring_metric(Xstd, gene_clusters, group_indices, "ME", "N")

    gene_rows = []
    for j, gene in enumerate(genes):
        keep = [x for x in range(len(genes)) if x != j]
        Xstd_j, Z_j, _ = preprocess_pca(X, gene_idx=keep, k=2)
        fit_j = fit_group_functors(Z_j, group_indices, lam=1.0)
        loocv_j = loocv_group_models(Z_j, GROUP_ORDER, patient_ids, groups_by_patient, lam=1.0)
        delta_j = (loocv_j.query("group=='ME'").mse_dir - loocv_j.query("group=='ME'").mse_comp).mean()
        fd_j = within_defect(fit_j, "ME") - within_defect(fit_j, "N")
        labels_j = np.delete(gene_clusters, j)
        E_j = symmetric_naturality(fit_j, "ME", "N", "early")
        L_j = symmetric_naturality(fit_j, "ME", "N", "late")
        R_j = rewiring_metric(Xstd_j, labels_j, group_indices, "ME", "N")
        gene_rows.append({
            "gene": gene,
            "ISCS": delta_ME - delta_j,
            "FDCS": fd_full - fd_j,
            "Gamma_early": E_full - E_j,
            "Gamma_late": L_full - L_j,
            "Gamma_rew": R_full - R_j,
        })
    gene_df = pd.DataFrame(gene_rows)
    gene_df.to_csv(f"{OUTDIR}/table_gene_scores_all.csv", index=False)
    for metric in ["ISCS", "FDCS", "Gamma_early", "Gamma_late", "Gamma_rew"]:
        gene_df.sort_values(metric, ascending=False).head(10).to_csv(f"{OUTDIR}/table_top_{metric}.csv", index=False)

    # 9) figures
    plt.figure(figsize=(6.5, 4.5))
    x = np.arange(len(GROUP_ORDER))
    y = within_out.set_index("group").loc[GROUP_ORDER, "boot_mean"].values
    lo = y - within_out.set_index("group").loc[GROUP_ORDER, "ci_low"].values
    hi = within_out.set_index("group").loc[GROUP_ORDER, "ci_high"].values - y
    plt.bar(GROUP_ORDER, y)
    plt.errorbar(x, y, yerr=[lo, hi], fmt="none", capsize=4)
    plt.ylabel("Functoriality defect")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig1_within_defect.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sub = delta_summary.set_index("group").loc[GROUP_ORDER]
    x = np.arange(len(GROUP_ORDER))
    w = 0.25
    plt.bar(x - w, sub["mean_dir"], width=w, label="Direct")
    plt.bar(x, sub["mean_comp"], width=w, label="Composed")
    plt.bar(x + w, sub["mean_multi"], width=w, label="Multi-input")
    plt.xticks(x, GROUP_ORDER)
    plt.ylabel("Mean LOOCV MSE")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig2_loocv_models.png", dpi=220)
    plt.close()

    def pair_to_matrix(df, value_col):
        mat = pd.DataFrame(np.zeros((len(GROUP_ORDER), len(GROUP_ORDER))), index=GROUP_ORDER, columns=GROUP_ORDER)
        for _, row in df.iterrows():
            mat.loc[row["g"], row["h"]] = row[value_col]
            mat.loc[row["h"], row["g"]] = row[value_col]
        return mat

    for val, name in [("B_baseline", "baseline_w2"), ("E_early", "early_naturality"), ("L_late", "late_naturality"), ("R_rewire", "rewiring")]:
        mat = pair_to_matrix(pair_df, val)
        plt.figure(figsize=(5.4, 4.5))
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis", square=True, cbar_kws={"shrink": 0.8})
        plt.title(name.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/fig_{name}.png", dpi=220)
        plt.close()

    pair_long = pair_df.melt(id_vars=["pair"], value_vars=["B_baseline", "E_early", "L_late", "R_rewire"], var_name="component", value_name="value")
    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=pair_long, x="pair", y="value", hue="component")
    plt.xticks(rotation=45)
    plt.ylabel("Raw structural component")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig3_pair_components_bar.png", dpi=220)
    plt.close()

    for metric in ["ISCS", "FDCS", "Gamma_early", "Gamma_late", "Gamma_rew"]:
        top = gene_df.sort_values(metric, ascending=False).head(10).iloc[::-1]
        plt.figure(figsize=(7, 5))
        plt.barh(top["gene"], top[metric])
        plt.xlabel(metric)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/fig_top_{metric}.png", dpi=220)
        plt.close()

    plt.figure(figsize=(6, 4.5))
    sub_plot = sub_df[["submodule", "ME_N_early", "ME_N_late", "ME_deltaMSE", "ME_delta_coherence"]].set_index("submodule")
    sns.heatmap(sub_plot, annot=True, fmt=".2f", cmap="mako")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_submodule_signals.png", dpi=220)
    plt.close()

    print("RS-CRM analysis finished.")
    print("Output directory:", OUTDIR)


if __name__ == '__main__':
    main()
