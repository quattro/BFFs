import argparse as ap
import os
import sys

from itertools import chain, combinations

import numpy as np
import numpy.linalg as linalg
import pandas as pd
import scipy.stats as stats

from numpy.linalg import multi_dot

mvn = stats.multivariate_normal


def get_resid(zs, SWLD, V):
    """
    Estimate the mean pleiotropic term using the complete covariance structure (i.e. V)
    and return the residual z-scores.
    """
    m, m = V.shape
    m, p = SWLD.shape

    # create mean factor
    X = SWLD.dot(np.ones(p))

    # estimate under the null for variance components, i.e. V = SW LD SW
    Vinv = np.linalg.pinv(V)

    numer = multi_dot([X.T, Vinv, zs])
    denom = multi_dot([X.T, Vinv, X])
    alpha = numer / denom
    resid = zs - X * alpha

    s2 = multi_dot([resid, Vinv, resid]) / (m - 1)
    inter_se = np.sqrt(s2 / denom)
    inter_z = alpha / inter_se

    return (resid, alpha, inter_se, inter_z)


def weighted_bf(zs, idx_set, SWLD, V, prior_chisq, prb, use_log=True, compute_resid=False):
    """
    Compute the prior-weighted BayesFactor
    """
    idx_set = np.array(idx_set)

    m = len(zs)

    # only need genes in the causal configuration using FINEMAP BF trick
    nc = len(idx_set)
    cur_chi2 = prior_chisq / nc
    cur_D = np.eye(nc) * cur_chi2

    cur_V = V[idx_set].T[idx_set].T
    cur_Zp = zs[idx_set]

    # compute SVD for robust estimation
    if nc > 1:
        cur_U, cur_EIG, _ = linalg.svd(cur_V)
        cur_scaled_Zp = (cur_Zp.T.dot(cur_U)) ** 2
    else:
        cur_U, cur_EIG = 1, cur_V
        cur_scaled_Zp = cur_Zp ** 2

    # log BF + log prior
    cur_log_BF = 0.5 * -np.sum(np.log(1 + cur_chi2 * cur_EIG)) + \
        0.5 *  np.sum((cur_chi2 / (1 + cur_chi2 * cur_EIG)) * cur_scaled_Zp) + \
        nc * np.log(prb) + (m - nc) * np.log(1 - prb)

    if use_log:
        return cur_log_BF
    else:
        return np.exp(cur_log_BF)


def weighted_like(zs, idx_set, SWLD, V, prior_chisq, prb, use_log=True, compute_resid=False):
    """
    Compute the prior-weighted Likelihood of the z-score
    """
    idx_set = np.array(idx_set)
    m = len(zs)

    nc = len(idx_set)
    cur_D = np.zeros((m, m))
    cur_chi2 = prior_chisq / nc if nc > 0 else 0
    for idx in idx_set:
        cur_D[idx, idx] = cur_chi2

    # total variance in Z-scores is due to variance in data plus the prior variance on causal effect sizes
    V = V + multi_dot([V, cur_D, V])
    mu = np.zeros(m)

    if compute_resid:
        zs, _, _, _ = get_resid(zs, SWLD, V)

    # evidence of the Z-scores under this causal configuration
    local = mvn.logpdf(zs, mean=mu, cov=V) + nc * np.log(prb) + (m - nc) * np.log(1 - prb)

    if use_log:
        return local
    else:
        return np.exp(local)


def get_matrix(nsnp, beta, alpha):
    """
    Generate a random matrix with condition number kappa = beta / alpha
    Machine precision is limited to about rank~1e17 so we can cheat to
    create a rank-deficient matrix.

    Code was modified from Ben Recht's baller code
    """
    A = stats.norm.rvs(size=(nsnp, nsnp)) # simulate random matrix

    Q, R = linalg.qr(A)
    # create new eigenvalues
    S = stats.norm.rvs(size=nsnp)

    # force to positive
    S = 10 ** S

    # rescale
    Smin = min(S)
    Smax = max(S)
    S = (S - Smin) / (Smax - Smin)

    # hack for rank deficiency. will likely produce rank (nsnp - 2) matrix
    if np.isclose(alpha, 0):
        S = S * beta
        S[-1] = 0
    else:
        S = alpha + S * (beta - alpha)

    # chef's kiss emoji
    A = multi_dot([Q.T, np.diag(S), Q])

    return A


def simulate(args, LD):
    ngwas = args.NGWAS
    neqtl = args.NEQTL

    ngenes = args.M
    h2g_expr = args.h2g

    prior_chisq = ngwas * args.prior_var
    prior_prob = 1 / float(ngenes)

    nsnp, _ = LD.shape

    # simulate eQTL weights
    weights = []
    for idx in range(args.M):
        cidx = np.random.choice(nsnp)
        betas = np.zeros(nsnp)
        betas[cidx] = np.random.normal(0, np.sqrt(h2g_expr))
        weights.append(betas)


    W = np.array(weights).T
    if args.infer_weights:
        Vinv = np.linalg.inv(LD + np.eye(nsnp) * 0.1)
        W_infer = np.zeros((args.P, args.M))
        for idx in range(args.M):
            W_infer.T[idx] = mvn.rvs(mean=W.T[idx], cov=Vinv / args.NEQTL)
    else:
        W_infer = W

    C = multi_dot([W_infer.T, LD, W])
    S = np.diag(1 / np.sqrt(np.diag(C)))
    V_true = multi_dot([S, C, S])

    C = multi_dot([W_infer.T, LD, W_infer])
    S = np.diag(1 / np.sqrt(np.diag(C)))
    V_noise = multi_dot([S, C, S])

    if args.fixed_direct:
        inter = multi_dot([S, W_infer.T, LD, np.ones(nsnp)])
        alpha_snp = np.random.normal(0, np.sqrt(args.prior_var))
        lambda_snp = np.sqrt(ngwas) * inter * alpha_snp
    else:
        lambda_snp = np.zeros(ngenes)

    # simulate Z-scores
    causals = np.random.choice(ngenes)
    D = np.zeros((ngenes, ngenes))
    D[causals, causals] = prior_chisq

    V_model = multi_dot([V_true, D, V_true]) + V_noise

    zscores = mvn.rvs(mean=lambda_snp, cov=V_model)

    return (zscores, W_infer, LD, V_noise, causals, lambda_snp, prior_prob, prior_chisq)


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("NGWAS", type=int, help="GWAS sample size")
    argp.add_argument("NEQTL", type=int, help="eQTL sample size")
    argp.add_argument("P", type=int, help="Number of SNPs")
    argp.add_argument("M", type=int, help="Number of genes")
    argp.add_argument("h2g", type=float, help="Heritability of gene expression")
    argp.add_argument("--prior-var", type=float, default=2e-3, help="Prior variance explained per gene")
    argp.add_argument("-a", "--alpha", type=int, default=1, help="Minimum LD eigenvalue")
    argp.add_argument("-b", "--beta", type=int, default=10, help="Maximum LD eigenvalue")
    argp.add_argument("--fixed-direct", action="store_true", default=False, help="Simulate direct effects under scalar model")
    argp.add_argument("--random-direct", action="store_true", default=False, help="Simulate direct effects under shared eQTL")
    argp.add_argument("--add-diagonal", action="store_true", default=False, help="Add noise to diagonal of covariance matrix for loglike")
    argp.add_argument("--infer-weights", action="store_true", default=False, help="If set, returns inferred eQTL weights, rather than true")
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)

    #NAMES = ["BF", "LL", "BF.N.RESID", "LL.N.RESID", "BF.F.RESID", "LL.F.RESID"]
    NAMES = ["BF", "LL", "BF.N.RESID", "LL.N.RESID", "LL.F.RESID"]

    prior_prob = 1 / float(args.M)
    prior_chisq = args.NGWAS * args.prior_var

    LD = get_matrix(args.P, args.beta, args.alpha)

    zscores, W, LD, V, causals, lambda_snp, prior_prob, prior_chisq = simulate(args, LD)


    S = np.diag(1 / np.sqrt(np.diag(multi_dot([W.T, LD, W]))))
    SWLD = multi_dot([S, W.T, LD])
    zscores_resid, _, _, _ = get_resid(zscores, SWLD, V)

    def weighted_bf_null_resid(zscores, subset, SWLD, V, prior_chisq, prior_prob):
        return weighted_bf(zscores_resid, subset, SWLD, V, prior_chisq, prior_prob, use_log=True)

    def weighted_bf_full_resid(zscores, subset, SWLD, V, prior_chisq, prior_prob):
        return weighted_bf(zscores, subset, SWLD, V, prior_chisq, prior_prob, use_log=True, compute_resid=True)

    def weighted_like_(zscores, subset, SWLD, V, prior_chisq, prior_prob):
        Vll = V + np.eye(args.M) * 0.1 if args.add_diagonal else V
        return weighted_like(zscores, subset, SWLD, Vll, prior_chisq, prior_prob, use_log=True)

    def weighted_like_null_resid(zscores, subset, SWLD, V, prior_chisq, prior_prob):
        Vll = V + np.eye(args.M) * 0.1 if args.add_diagonal else V
        return weighted_like(zscores_resid, subset, SWLD, Vll, prior_chisq, prior_prob, use_log=True)

    def weighted_like_full_resid(zscores, subset, SWLD, V, prior_chisq, prior_prob):
        Vll = V + np.eye(args.M) * 0.1 if args.add_diagonal else V
        return weighted_like(zscores, subset, SWLD, Vll, prior_chisq, prior_prob, use_log=True, compute_resid=True)

    methods = dict()
    methods["BF"] = weighted_bf
    methods["BF.N.RESID"] = weighted_bf_null_resid
    #methods["BF.F.RESID"] = weighted_bf_full_resid
    methods["LL"] = weighted_like_
    methods["LL.N.RESID"] = weighted_like_null_resid
    methods["LL.F.RESID"] = weighted_like_full_resid

    k = 2
    rm = range(args.M)

    pips = dict()
    for name in NAMES:
        pips[name] = np.zeros(args.M)

    nulls = dict()
    nulls["BF"] = args.M * np.log(1 - prior_prob)
    nulls["BF.N.RESID"] = args.M * np.log(1 - prior_prob)
    #nulls["BF.F.RESID"] = args.M * np.log(1 - prior_prob)

    nulls["LL"] = weighted_like_(zscores, [], SWLD, V, prior_chisq, prior_prob)
    nulls["LL.N.RESID"] = weighted_like_null_resid(zscores, [], SWLD, V, prior_chisq, prior_prob)
    nulls["LL.F.RESID"] = nulls["LL.N.RESID"]

    marginals = dict()
    for name in NAMES:
        marginals[name] = nulls[name]

    for subset in chain.from_iterable(combinations(rm, n) for n in range(1, k + 1)):
        for name in NAMES:
            func = methods[name]
            local = func(zscores, subset, SWLD, V, prior_chisq, prior_prob)
            marginals[name] = np.logaddexp(marginals[name], local)

            for idx in subset:
                if np.isclose(pips[name][idx], 0):
                    pips[name][idx] = local
                else:
                    pips[name][idx] = np.logaddexp(local, pips[name][idx])

    for name in NAMES:
        pips[name] = np.exp(pips[name] - marginals[name])

    causal_genes = np.zeros(args.M).astype(bool)
    causal_genes[causals] = True

    df_dict = dict([(name, pips[name]) for name in NAMES])
    df_dict["Zscores"] = zscores
    df_dict["Causal"] = causal_genes
    df = pd.DataFrame(df_dict)
    df = df[["Zscores", "Causal"] + NAMES]

    df.to_csv(args.output, index=False, sep="\t", float_format="%.3e")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
