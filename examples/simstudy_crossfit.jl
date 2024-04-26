using ExpectileRegression
using StableRNGs
using LinearAlgebra
using Distributions
using DataFrames
using Random
using SparseArrays
using Printf

function gendat(rng, icc, X)

    n, p = size(X)
    Ey = X[:, 2] + X[:, 3]

    # Consecutive pairs are correlated.
    e = randn(rng, n)
    n2 = Int(floor(n/2))
    u = randn(n2)
    u = kron(u, [1, 1])
    e = sqrt(icc) * u + sqrt(1 - icc) * e

    y = Ey + e .* (1 .+ 0.3*X[:, 2])
    tgtf = tau -> 2 + 0.6*ExpectileRegression.normal_expectile(tau)

    return y, Ey, tgtf
end

function genAR(rng, n, p, r)
    X = randn(rng, n, p)

    # Intercept
    X[:, 1] .= 1

    # Treatment indicator
    X[:, 2] .= 1
    n2 = Int(n/2+1)
    X[n2:end, 2] .= -1

    for j in 3:p
        X[:, j] = r*X[:, j-1] + sqrt(1-r^2)*X[:, j]
    end

    return X
end

# Estimate coverage probabilities for Wald-type intervals of each regression coefficient.
# Also returns the robust covariance estimate V and the empirical covariance of the coefficient
# estimates.
function crossfit_helper(tau::Float64, X, icc, tgtf; rng=StableRNG(123), nrep=100, nfold=100)

    n, p = size(X)

    X1 = copy(X)
    X1[:, 2] .= 1
    X2 = copy(X)
    X2[:, 2] .= -1
    xd = mean(X1 - X2; dims=1)

    # Estimate the difference between the expected response for
    # treated and untreated people.
    targetf = function(er, x, i)
        local z = copy(x)
        z = reshape(z, (1, length(z)))
        z[2] = 1
        b1 = predict(er, z)[1]
        z[2] = -1
        b0 = predict(er, z)[1]
        return b1 - b0
    end

    # Covariance mask
    M = spzeros(n, n)
    for i in 1:Int(floor(n/2))
        j = 2*(i - 1) + 1
        M[j:j+1, j:j+1] .= 1
    end

    target = tgtf(tau)

    z = zeros(nrep, 7)
    for i in 1:nrep
        y, _, _ = gendat(rng, icc, X)

        er = fit(ExpectReg, X, y; tau=[tau])
        cc = coef(er)[:, 1]

        V = vcov_array(er; M=M)
        est_mb = dot(cc, xd)
        se_mb = sqrt(xd * V * xd')[1,1]

        fest = crossfit(er, targetf; M=M, nfold=nfold)
        est_cf = mean(fest[:, 1])
        se_cf = sqrt(mean(fest[:, 2]) + var(fest[:, 1]))
        z[i, :] = [target, est_cf, se_cf, (est_cf - target) / se_cf, est_mb, se_mb, (est_mb - target) / se_mb]
    end

    zf = DataFrame(z, [:target, :est_cf, :sd_cf, :z_cf, :est_model, :se_model, :z_model])
    zf[:, :tau] .= tau
    zf[:, :icc] .= icc

    return zf
end

function simstudy_crossfit(n, p, icc, tau::Float64, tgtf; rng=StableRNG(123), nrep=1000, nfold=100)

    X = genAR(rng, n, p, icc)
    zf = crossfit_helper(tau, X, icc, tgtf; rng=rng, nrep=nrep, nfold=nfold)

    return zf
end

function summary(zf)
    println(@sprintf("%10.4f  Tau", first(zf[:, :tau])))
    println(@sprintf("%10.4f  ICC", first(zf[:, :icc])))
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
tau = 0.01
icc = 0.5

_, _, tgtf = gendat(Random.default_rng(), icc, randn(n, p))
zf = simstudy_crossfit(n, p, icc, tau, tgtf; nrep=500, nfold=50)
summary(zf)
