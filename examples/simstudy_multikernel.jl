using ExpectileRegression
using StableRNGs
using LinearAlgebra
using KernelFunctions
using DataFrames
using SparseArrays
using Random
using Statistics
using StatsBase
using Printf

include("utils.jl")
include("multikernel.jl")

function gendat(rng, icc, X)

    n, p = size(X)
    Ey = X[:, 1] + 1 ./ (1 .+ exp.(-X[:, 2])) + 0.5*(0.5*X[:,3] .- 0.25*X[:,3].^2)

    # Consecutive pairs are correlated.
    e = randn(rng, n)
    n2 = Int(floor(n/2))
    u = randn(n2)
    u = kron(u, [1, 1])
    e = sqrt(icc) * u + sqrt(1 - icc) * e

    y = Ey + e .* (1 .+ 0.3*X[:, 1])
    tgtf = tau -> 2 + 0.6*ExpectileRegression.normal_expectile(tau)

    return y, Ey, tgtf
end

function genAR(rng, n, p, r)
    X = randn(rng, n, p)

    # Treatment indicator
    X[:, 1] .= 1
    n2 = Int(n/2+1)
    X[n2:end, 1] .= -1

    for j in 2:p
        X[:, j] = r*X[:, j-1] + sqrt(1-r^2)*X[:, j]
    end

    return X
end

function fit_multikernel(y, Ey, X, Z, Pen, M, lam, tau, mk; adjust=:hc3)

    XX = copy(X)
    XX[:, 1] .= 1
    Z1 = predict_mat(mk, XX)
    XX[:, 1] .= -1
    Z2 = predict_mat(mk, XX)

    # Estimate the difference between the expected response for
    # treated and untreated people.
    function targetf(er, x, i)
        # Ignore x
        cc = coef(er)[:]
        b1 = dot(Z1[i, :], cc)
        b0 = dot(Z2[i, :], cc)
        return b1 - b0
    end

    er = fit(ExpectReg, Z, y; L2Pen=lam*Pen, tau=[tau])
    cc = coef(er)[:, 1]

    # Model based estimate and SE
    V = vcov_array(er; M=M, adjust=adjust)
    xd = mean(Z1 - Z2; dims=1)[:]
    est_mb = dot(cc, xd)
    se_mb = sqrt(xd' * V * xd)[1,1]

    # Cross fit estimate and SE
    nfold = 50
    fest = crossfit(er, targetf; M=M, nfold=nfold)
    est_cf = mean(fest[:, 1])
    se_cf = sqrt(mean(fest[:, 2]) + var(fest[:, 1]))

    return est_cf, se_cf, est_mb, se_mb
end

function simstudy_multikernel(n, p, icc, lam, tau; nrep=100, adjust=:hc2)

    rng = StableRNG(123)

    # Using same ICC for covariates and residual variation
    X = genAR(Random.default_rng(), n, p, icc)

    ker1 = ConstantKernel()
    ker2 = with_lengthscale(PolynomialKernel(degree=1, c=0), 1.0)
    ker3 = with_lengthscale(PolynomialKernel(degree=2, c=0), 1.0)

    mk = MultiKernel([ker1, ker2, ker3], X; maxdim=10, pwt=Float64[0, 0, 1])
    println("Basis matrix sizes: ", size.(mk.B))
    Z = basis(mk)
    _, s, _ = svd(Z)
    tol = 1e-6
    println(@sprintf("Combined basis has %d singular values smaller than %5.2e", sum(s .< tol), tol))

    Pen = penalty(mk)
    Pen = Z' * Pen * Z
    Pen = Pen + 0.1*I

    # Covariance mask
    M = spzeros(n, n)
    for i in 1:Int(floor(n/2))
        j = 2*(i - 1) + 1
        M[j:j+1, j:j+1] .= 1
    end

    z = zeros(nrep, 7)
    for i in 1:nrep
        y, Ey, tgtf = gendat(rng, icc, X)
        est_cf, sd_est_cf, est, se = fit_multikernel(y, Ey, X, Z, Pen, M, lam, tau, mk; adjust=adjust)
        target = tgtf(tau)
        z[i, :] = [target, est_cf, sd_est_cf, (est_cf - target) / sd_est_cf, est, se, (est - target) / se]
    end

    zf = DataFrame(z, [:target, :est_cf, :sd_cf, :z_cf, :est_model, :se_model, :z_model])
    zf[:, :tau] .= tau
    zf[:, :lambda] .= lam
    zf[:, :icc] .= icc

    return zf
end

function summary(zf)
    println(@sprintf("%10.4f  Tau", first(zf[:, :tau])))
    println(@sprintf("%10.4f  ICC", first(zf[:, :icc])))
    println(@sprintf("%10.4f  lambda", first(zf[:, :lambda])))
    println(@sprintf("%10.4f  Target", first(zf[:, :target])))
    println(@sprintf("%10.4f  Mean crossfit estimate", mean(zf[:, :est_cf])))
    println(@sprintf("%10.4f  Bias of crossfit estimate", mean(zf[:, :est_cf]) - first(zf[:, :target])))
    println(@sprintf("%10.4f  SD of crossfit estimate", std(zf[:, :est_cf])))
    println(@sprintf("%10.4f  Mean of crossfit SE", mean(zf[:, :sd_cf])))
    println(@sprintf("%10.4f  SD of crossfit Z-scores", std(zf[:, :z_cf])))
    println(@sprintf("%10.4f  Mean model-based estimate", mean(zf[:, :est_model])))
    println(@sprintf("%10.4f  Bias of model-based estimate", mean(zf[:, :est_model]) - first(zf[:, :target])))
    println(@sprintf("%10.4f  SD of model-based estimate", std(zf[:, :est_model])))
    println(@sprintf("%10.4f  Mean cluster-robust SE", mean(zf[:, :se_model])))
    println(@sprintf("%10.4f  SD of cluster-robust Z-scores", std(zf[:, :z_model])))
end

n = 500
p = 10
icc = 0.5
tau = 0.01
lam = 0.1/sqrt(n)
nrep = 200
adjust = :hc2
zf = simstudy_multikernel(n, p, icc, lam, tau; nrep=nrep, adjust=adjust)
summary(zf)
