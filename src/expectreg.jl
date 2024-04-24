
mutable struct ExpectReg{S<:AbstractMatrix} <: StatisticalModel

    # The design matrix
    X::S

    # The response vector
    y::Vector{Float64}

    # The parameter estimates, each column corresponds to a different expectile (tau).
    beta::Matrix{Float64}

    # Optional case weights
    wgt::Vector{Float64}

    # L2 penalty matrix
    L2Pen::AbstractMatrix

    # The expectiles being estimated
    tau::Vector{Float64}
end

"""
    ExpectReg(X::AbstractMatrix, y::AbstractVector; tau::AbstractVector=Float64[0.5], wgt::AbstractVector=Float64[], L2Pen::AbstractMatrix=0*I(size(X,2)))

Construct an expectile regression model for the responses in 'y' and the covariates in 'X'.

* tau - A vector of values in [0, 1] defining the expectiles to be jointly estimated, defaults to [0.5]
    which estimates the conditional mean (equivalent to ordinary least squares).
* wgt - An optional vector of case weights, defaults to uniform weights.
* L2Pen - A positive semidefinite p x p penalization matrix, where p is the number of covariates, such that the estimation is
    penalized by b' * L2Pen * b for the parameter vector 'b'.  This is analogous to ridge regression for ordinary least squares.
    Defaults to a matrix of zeros, giving no penalization.
"""
function ExpectReg(X::AbstractMatrix, y::AbstractVector; tau::AbstractVector=Float64[0.5], wgt::AbstractVector=Float64[], L2Pen::AbstractMatrix=0*I(size(X,2)))

    n, p = size(X)
    q = length(tau)

    if length(y) != n
        error("Sizes of 'X' ($(size(X, 1)) x $(size(X, 2))) and 'y' ($(length(y))) are not compatible")
    end

    if length(wgt) != 0 && length(wgt) != n
        error("Weight vector must be empty or have the same length as 'y'.")
    end

    if length(wgt) == 0
        wgt = ones(n)
    end

    if !(all(size(L2Pen) .== (p, p)))
        error("Size of 'L2Pen' does not match size of 'X'.")
    end

    return ExpectReg(X, y, zeros(p, q), wgt, L2Pen, tau)
end

# The expectile regression loss function.
function expectreg_loss(er::ExpectReg, beta::Vector{T}; j::Int=1) where{T<:Real}
    (; y, X, wgt, L2Pen, tau) = er
    fv = X * beta
    r = y - fv
    pen = size(L2Pen, 1) != 0 ? beta' * L2Pen * beta : 0.0
    return dot(eloss.(r, tau[j]), wgt) + pen
end

# Compute the gradient of the expectile loss function and store
# it into g.
function expectreg_loss_grad!(er::ExpectReg, g::Vector{T}, beta::Vector{T}; j::Int=1) where{T<:Float64}
    (; y, X, wgt, L2Pen, tau) = er
    fv = X * beta
    r = y - fv
    u = elossgrad.(r, tau[j])
    g .= -X' * (wgt .* u)
    if size(L2Pen, 1) > 0
        g .+= 2 * L2Pen * beta
    end
end

function fit!(er::ExpectReg; start=nothing, meth=LBFGS())
    (; y, X, wgt, L2Pen, tau) = er

    for j in eachindex(tau)
        s = isnothing(start) ? nothing : start[:, j]
        fitj!(er, j; start=s, meth=LBFGS())
    end
end

# Fit the parameters for the j'th expectile.
function fitj!(er::ExpectReg, j::Int; start=nothing, meth=LBFGS())
    (; y, X, wgt, L2Pen, tau) = er

    n, p = size(X)

    if tau[j] == 0.5
        W = Diagonal(wgt)
        er.beta[:, j] .= (X'  * W * X + 2*L2Pen) \ (X' * W * y)
        return
    end

    loss1 = beta -> expectreg_loss(er, beta; j=j)
    grad! = (g, beta) -> expectreg_loss_grad!(er, g, beta; j=j)
    beta0 =
    if isnothing(start)
        0.1*randn(p)
    else
        if !(typeof(start) <: Vector && length(start) != p)
            error("'start' must be a vector of length $(p)")
        end
        start
    end

    opts = Optim.Options(g_tol=1e-4)
    r = optimize(loss1, grad!, beta0, meth, opts)
    if !Optim.converged(r)
        @warn("Expectile regression did not converge")
    end
    er.beta[:, j] .= Optim.minimizer(r)
end

function fit(::Type{ExpectReg}, X::AbstractMatrix, y::AbstractVector; tau::Vector=Float64[0.5],
             L2Pen=0*I(size(X,2)), wgt=Float64[], start=nothing, meth=LBFGS())

    er = ExpectReg(X, y; tau=tau, wgt=wgt, L2Pen=L2Pen)
    fit!(er; start=start, meth=meth)
    return er
end

function coef(er::ExpectReg)
    return er.beta
end

function fitted(er::ExpectReg)
    (; y, X, beta) = er
    return X * beta
end

function response(er::ExpectReg)
    return er.y
end

function residuals(er::ExpectReg)
    return response(er) - fitted(er)
end

function nobs(er::ExpectReg)
    return length(er.y)
end

"""
     vcov(er::ExpectReg; M=I(nobs(er)))

Return the estimated variance-covariance matrix of the parameter estimates.
This function returns a block matrix with blocks corresponding to the expectile
values in 'tau'.  To access the (co)variance matrix between the j'th and k'th
value of tau, use 'vcov(er)[Block(j, k)]'.

M is a mask that defines the pairs of observations that may be non-independent.
The default value of M is the identity matrix, impling that all pairs of observations
are independent.

See 'vcov_array' to compute the same covariance as a Matrix.
"""
function vcov(er::ExpectReg; M::T=I(nobs(er)), adjust=:hc1) where{T<:AbstractMatrix}

    (; X, y, beta, tau, wgt, L2Pen) = er

    if !(adjust in [:none, :hc1, :hc2, :hc3])
        error("Invalid 'adjust' in 'vcov'")
    end

    n, p = size(X)

    # Degrees of freedom of the mean model
    H = X * ((X'*X + L2Pen) \ X')
    edf = n - tr((I - H)*(I - H))

    # The bread for each expectile point
    B, EX = [], []
    r = residuals(er)
    for j in eachindex(tau)
        cr = check.(r[:, j], tau[j]) .* wgt

        # Add a bit of diagonal regularization to make sure that the bread is invertible.
        # TODO need a better way to set this constant
        b = X' * Diagonal(cr) * X + L2Pen
        a, _ = eigen(Symmetric(b))
        c = (0.0001*p/n) * maximum(a)
        push!(B, b + c*I(p))

        eta = cr.* r[:, j]
        if adjust == :hc2
            eta ./= (1 .- diag(H))
        end
        push!(EX, Diagonal(eta) * X)
    end

    q = length(tau)
    V = BlockArray{Float64}(undef, fill(p, q), fill(p, q))
    for j1 in eachindex(tau)
        for j2 in 1:j1
            meat = EX[j1]' * M * EX[j2]
            view(V, Block(j1, j2)) .= B[j1] \ meat / B[j2]
        end
    end

    if adjust == :hc1
        f = n / (n - edf)
        # V .*= f sometimes errors
        for b in eachblock(V)
            b .*= f
        end
    end

    return V
end

function vcov_array(er::ExpectReg; M::T=I(nobs(er)), adjust=:hc1) where{T<:AbstractMatrix}
    V = vcov(er; M=M, adjust=adjust)
    return Symmetric(Array(V))
end


function predict(er::ExpectReg)
    (; X, beta) = er
    return X * beta
end

function predict(er::ExpectReg, x::T) where{T<:AbstractArray}
    return x * coef(er)
end

"""
    crossfit(er::ExpectReg, targetf::Function; nfold=100)

Use cross-fitting to estimate a target quantity and provide confidence intervals
for the estimate.

`targetf` is a function that takes a fitted `ExpectReg` and a vector of
covariates `x`.  It returns the plug-in estimate of the target quantity at `x`
"""
function crossfit(er::ExpectReg, targetf::Function; M=I(nobs(er)), nfold=100)

    (; X, y, wgt, tau, L2Pen) = er

    n, p = size(X)

    # Get the low-order eigenvectors of the graph Laplacian of the graph determined
    # by the covariance mask.
    L = M - I
    d = sum(L; dims=1)[:]
    L = Diagonal(d) - L

    nev = 20
    a = zeros(0)
    b = zeros(0, 0)
    while nev < size(L, 1)
        dec, hist = partialschur(Symmetric(L); nev=nev, which=:SR)
        a, b = partialeigen(dec)
        b = real.(b)
        if all(abs2.(a) .< 1e-10)
            break
        end
        nev *= 2
    end

    mom = zeros(nfold, 2)
    for k in 1:nfold

        # Split in a way that minimizes between-split covariances.
        u = b * randn(size(b, 2))
        mm = median(u)

        itr = findall(u .<= mm)
        ite = findall(u .> mm)
        Xtr = X[itr, :]
        Xte = X[ite, :]
        ytr = y[itr]
        wgr = wgt[itr]
        wge = wgt[ite]

        ee = ExpectReg(Xtr, ytr; tau=tau, wgt=wgr, L2Pen=L2Pen)
        fit!(ee)

        # The estimated target function for each case
        tgt = [targetf(ee, X[i, :], i) for i in ite]

        # The point estimate in the current fold.
        est = mean(tgt, weights(wge))

        # Get the robust variance estimate for the average point estimate
        # in the current fold.
        mm = M[ite, ite]
        m = length(ite)
        br = (wge' * mm * wge)[1, 1]
        resid = tgt .- est
        u = wge .* resid
        mt = (u' * mm * u)[1, 1]
        va = (m / (m - 1)) * mt / br^2

        mom[k, :] = [est, va / m]
    end

    return mom
end
