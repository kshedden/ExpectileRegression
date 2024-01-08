using ExpectileRegression
using StableRNGs
using LinearAlgebra
using KernelFunctions
using DataFrames
using SparseArrays
using Random
using Statistics
using StatsBase

include("utils.jl")
include("multikernel.jl")

# Simulate data with no covariate effects at the mean, and with
# heteroscedasticity explained by the treatment indicator X[2].
function gendat(rng, icc, X)

    # Parameters determining the mean structure
    b_mean = Float64[1, 0, 1]

    # Parameters determining the heteroscedasticity structure
    b_het = Float64[1, 1, 0]

    n, p = size(X)
    beta = zeros(p)
    beta[1:3] = b_mean
    Ey = X * beta

    # Consecutive pairs are correlated.
    e = randn(rng, n)
    n2 = Int(floor(n/2))
    u = randn(n2)
    u = kron(u, [1, 1])
    e = sqrt(icc) * u + sqrt(1 - icc) * e

    Xh = X[:, 1:3] * b_het
    y = Ey + e .* (2 .+ Xh)

    # Construct a function that returns the true value of the ATE at expectile tau.
    # Assumes X1 is the treatment variable and is coded 1/-1
    X1 = copy(X)
    X1[:, 1] .= 1
    X2 = copy(X)
    X2[:, 1] .= -1
    u = abs.(2 .+ X1[:, 1:3]*b_het) - abs.(2 .+ X2[:, 1:3]*b_het)
    u = mean(u)
    tgtf = tau -> 2*b_mean[1] + u*ExpectileRegression.normal_expectile(tau)

    return y, Ey, beta, tgtf
end

function fit_rkhs(y, Ey, X, Z, Pen, M, lam, tau, mk)

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

    V = vcov_array(er; M=M, adjust=:hc3)

    xd = mean(Z1 - Z2; dims=1)[:]
    se_mb = sqrt(xd' * V * xd)[1,1]

    nfold = 50
    fest = crossfit(er, targetf; M=M, nfold=nfold)
    est_cf = mean(fest[:, 1])
    sd_est_cf = sqrt(mean(fest[:, 2]) + var(fest[:, 1]))

    return est_cf, sd_est_cf, dot(cc, xd), se_mb
end

function simstudy(n, p, icc, lam, tau; nrep=100)

    rng = StableRNG(123)
    X = genAR(Random.default_rng(), n, p, icc)

    ker1 = ConstantKernel()
    ker2 = with_lengthscale(PolynomialKernel(degree=1, c=0), 1.0)
    ker3 = with_lengthscale(PolynomialKernel(degree=2, c=0), 1.0)

    mk = MultiKernel([ker1, ker2, ker3], X; maxdim=10, pwt=Float64[0, 0, 1])
    println(size.(mk.B))
    Z = basis(mk)

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
        i % 50 == 0 && println(i)
        y, Ey, beta, tgtf = gendat(rng, icc, X)
        est_cf, sd_est_cf, est, se = fit_rkhs(y, Ey, X, Z, Pen, M, lam, tau, mk)
        target = tgtf(tau)
        z[i, :] = [target, est_cf, sd_est_cf, (est_cf - target) / sd_est_cf, est, se, (est - target) / se]
    end

    return z
end


n = 300
p = 10
icc = 0.5
tau = 0.9
lam = 0.0
nrep = 1000
z = simstudy(n, p, icc, lam, tau; nrep=nrep)
