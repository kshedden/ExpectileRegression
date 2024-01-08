# Generate a n x p design matrix whose columns are autocorrelated
# with parameter r.  The first column is an intercept and the second
# column is a binary indicator.
function genAR(rng, n, p, r)
    X = randn(rng, n, p)

    # Treatment indicator
    X[:, 1] .= -1
    n2 = Int(n/2+1)
    X[n2:end, 1] .= 1

    for j in 2:p
        X[:, j] = r*X[:, j-1] + sqrt(1-r^2)*X[:, j]
    end

    return X
end

