using ExpectileRegression
using StableRNGs
using LinearAlgebra
using Distributions
using DataFrames
using Random
using SparseArrays
using Printf

# Simulate data with no covariate effects at the mean, and with
# heteroscedasticity explained by the treatment indicator X[2].
function simdat(rng, icc, X)
    n, p = size(X)
    beta = zeros(p)
    beta[1:3] = Float64[1, 0, 0]
    Ey = X * beta

    # Consecutive pairs are correlated.
    e = randn(rng, n)
    n2 = Int(floor(n/2))
    u = randn(n2)
    u = kron(u, [1, 1])
    e = sqrt(icc) * u + sqrt(1 - icc) * e

    y = Ey + e .* (1 .+ X[:, 2])
    tgtf = tau -> ExpectileRegression.normal_expectile(tau)*(2 - 1)

    return y, Ey, beta, tgtf
end

# Generate a n x p design matrix whose columns are autocorrelated
# with parameter r.  The first column is an intercept and the second
# column is a binary indicator.
function genAR(rng, n, p, r)
    X = randn(rng, n, p)

    # Intercept
    X[:, 1] .= 1

    # Treatment indicator
    X[:, 2] .= 0
    n2 = Int(n/2+1)
    X[n2:end, 2] .= 2

    for j in 3:p
        X[:, j] = r*X[:, j-1] + sqrt(1-r^2)*X[:, j]
    end

    X[:, 1] ./= 2

    return X
end

# Estimate coverage probabilities for Wald-type intervals of each regression coefficient.
# Also returns the robust covariance estimate V and the empirical covariance of the coefficient
# estimates.
function study_hom(tau::Float64, X, icc; rng=StableRNG(123), nrep=100, nfold=100)

    n, p = size(X)

    # Estimate the difference between the expected response for
    # treated and untreated people.
    targetf = function(er, x)
        if typeof(x) <: AbstractVector
            x = reshape(x, (1, length(x)))
        end
        z = copy(x)
        z[2] = 1
        b1 = predict(er, z)[1]
        z[2] = 0
        b0 = predict(er, z)[1]
        return b1 - b0
    end

    # Covariance mask
    M = spzeros(n, n)
    for i in 1:Int(floor(n/2))
        j = 2*(i - 1) + 1
        M[j:j+1, j:j+1] .= 1
    end

    R = zeros(nrep, 3)
    for i in 1:nrep
        y, _, _ = simdat(rng, icc, X)

        er = ExpectReg(X, y; tau=[tau])

        V = vcov_array(er; M=M)
        X1 = copy(X)
        X1[:, 2] .= 1
        X2 = copy(X)
        X2[:, 2] .= 0
        xd = mean(X1 - X2; dims=1)
        se_mb = sqrt(xd * V * xd')[1,1]

        fest = crossfit(er, targetf; M=M, nfold=nfold)
        est = mean(fest[:, 1])
        sd_est = sqrt(mean(fest[:, 2]) + var(fest[:, 1]))
        R[i, :] = [est, sd_est, se_mb]
    end

    return R
end

function study_hom(n, p, tau::Float64, icc; rng=StableRNG(123), nrep=1000, nfold=100)

    r = 0.7
    X = genAR(rng, n, p, r)

    R = study_hom(tau, X, icc; rng=rng, nrep=nrep, nfold=nfold)

    return R
end

n = 400
p = 20
for tau in [0.5, 0.75, 0.9, 0.95, 0.99]
    for icc in [0, 0.5]
        _, _, _, tgtf = simdat(Random.default_rng(), icc, randn(n, p))
        R = study_hom(n, p, tau, icc; nrep=200, nfold=50)
        println("tau=", tau)
        println(@sprintf("%12.4f target", tgtf(tau)))
        println(@sprintf("%12.4f mean estimate", mean(R[:, 1])))
        println(@sprintf("%12.4f mean crossfit SE", mean(R[:, 2])))
        println(@sprintf("%12.4f mean model-based SE", mean(R[:, 3])))
        println(@sprintf("%12.4f empirical SE", std(R[:, 1])))
        println("")
    end
end
