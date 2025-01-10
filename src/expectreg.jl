
mutable struct ExpectReg{S<:AbstractMatrix} <: StatisticalModel

    # The design matrix
    X::S

    # The response vector
    y::Vector{Float64}

    # The parameter estimates, each column corresponds to a different expectile (tau).
    beta::Matrix{Float64}

    # Optional case weights
    weights::Vector{Float64}

    # L2 penalty matrix
    L2Pen::AbstractMatrix

    # The expectiles being estimated
    tau::Vector{Float64}

    # A formula used to specify the model, if present.
    formula::Union{FormulaTerm,Nothing}

    # The mask determines which pairs of observations may be non-independent.
    mask::AbstractMatrix
end

"""
    ExpectReg(X::AbstractMatrix, y::AbstractVector; tau::AbstractVector=Float64[0.5], weights::AbstractVector=Float64[], L2Pen::AbstractMatrix=0*I(size(X,2)))

Construct an expectile regression model for the responses in 'y' and the covariates in 'X'.

* tau - A vector of values in [0, 1] defining the expectiles to be jointly estimated, defaults to [0.5]
    which estimates the conditional mean (equivalent to ordinary least squares).
* weights - An optional vector of case weights, defaults to uniform weights.
* L2Pen - A positive semidefinite p x p penalization matrix, where p is the number of covariates, such that the estimation is
    penalized by b' * L2Pen * b for the parameter vector 'b'.  This is analogous to ridge regression for ordinary least squares.
    Defaults to a matrix of zeros, giving no penalization.
"""
function ExpectReg(X::AbstractMatrix, y::AbstractVector; tau::AbstractVector=Float64[0.5], weights::AbstractVector=Float64[], L2Pen::AbstractMatrix=0*I(size(X,2)), mask=I(length(y)))

    n, p = size(X)
    q = length(tau)

    if length(y) != n
        error("Sizes of 'X' ($(size(X, 1)) x $(size(X, 2))) and 'y' ($(length(y))) are not compatible")
    end

    if length(weights) != 0 && length(weights) != n
        error("Weight vector must be empty or have the same length as 'y'.")
    end

    if length(weights) == 0
        weights = ones(n)
    end

    if !(all(size(L2Pen) .== (p, p)))
        error("Size of 'L2Pen' does not match size of 'X'.")
    end

    return ExpectReg(X, y, zeros(p, q), weights, L2Pen, tau, nothing, mask)
end

# The expectile regression loss function.
function expectreg_loss(er::ExpectReg, beta::Vector{T}; j::Int=1) where{T<:Real}
    (; y, X, weights, L2Pen, tau) = er
    fv = X * beta
    r = y - fv
    pen = size(L2Pen, 1) != 0 ? beta' * L2Pen * beta : 0.0
    return dot(eloss.(r, tau[j]), weights) + pen
end

# Compute the gradient of the expectile loss function with respect to the parameters beta, and store
# it into g.
function expectreg_loss_grad!(er::ExpectReg, g::Vector{T}, beta::Vector{T}; j::Int=1) where{T<:Float64}
    (; y, X, weights, L2Pen, tau) = er
    fv = X * beta
    r = y - fv
    u = elossgrad.(r, tau[j])
    g .= -X' * (weights .* u)
    if size(L2Pen, 1) > 0
        g .+= 2 * L2Pen * beta
    end
end

# Compute the gradient of the expectile loss function with respect to the parameters beta, for each observation,
# and store it into g.
function expectreg_loss_grad_obs!(er::ExpectReg, gobs::Matrix{T}, beta::Vector{T}; j::Int=1) where{T<:Float64}
    (; y, X, weights, L2Pen, tau) = er
    n = length(y)
    fv = X * beta
    r = y - fv
    u = elossgrad.(r, tau[j])
    gobs .= -X' * Diagonal(weights .* u)
    if size(L2Pen, 1) > 0
        gobs .+= 2 * L2Pen * beta / n
    end
end

# Compute the Hessian of the expectile loss function with respect to the parameters beta, and store
# it into H.
function expectreg_loss_hess!(er::ExpectReg, H::Matrix{T}, beta::Vector{T}; j::Int=1) where{T<:Float64}
    (; y, X, weights, L2Pen, tau) = er
    fv = X * beta
    r = y - fv
    u = elosshess.(r, tau[j])
    H .= X' * Diagonal(weights .* u) * X
    if size(L2Pen, 1) > 0
        H .+= 2 * L2Pen
    end
end

function fit!(er::ExpectReg; start=nothing, meth=LBFGS(), opts=Dict{Symbol,Any}(:g_tol=>1e-4), verbosity::Int=0)
    (; y, X, weights, L2Pen, tau) = er

    for j in eachindex(tau)
        s = isnothing(start) ? nothing : start[:, j]
        fitj!(er, j; start=s, meth=LBFGS(), opts=opts, verbosity=verbosity)
    end
end

# Fit the parameters for the j'th expectile.
function fitj!(er::ExpectReg, j::Int; start=nothing, opts=Dict{Symbol,Any}(:g_tol=>1e-4), meth=LBFGS(), verbosity::Int=0)
    (; y, X, weights, L2Pen, tau) = er

    n, p = size(X)

    if tau[j] == 0.5
        if verbosity > 0
            println("Using least squares for tau=0.5")
        end
        # OLS
        W = Diagonal(weights)
        er.beta[:, j] .= (X'  * W * X + 2*L2Pen) \ (X' * W * y)
        return
    end

    if verbosity > 0
        tn = Base.typename(typeof(meth)).wrapper
        println("Using gradient descent ($tn) for tau=$(tau[j])")
        opts = copy(opts)
        opts[:show_trace] = true
    end

    loss1 = beta -> expectreg_loss(er, beta; j=j)
    grad! = (g, beta) -> expectreg_loss_grad!(er, g, beta; j=j)
    beta0 = if isnothing(start)
        # Use OLS if no starting values are provided
        W = Diagonal(weights)
        (X'  * W * X + 2*L2Pen) \ (X' * W * y)
    else
        if !(typeof(start) <: Vector && length(start) != p)
            error("'start' must be a vector of length $(p)")
        end
        start
    end

    optsx = Optim.Options(; opts...)
    r = optimize(loss1, grad!, beta0, meth, optsx)
    if !Optim.converged(r)
        @warn("Expectile regression did not converge")
    end
    er.beta[:, j] .= Optim.minimizer(r)
end

function fit(::Type{ExpectReg}, X::AbstractMatrix, y::AbstractVector; tau::Vector=Float64[0.5],
             mask=nothing, L2Pen=0*I(size(X,2)), weights=Float64[], start=nothing,
             opts=Dict{Symbol,Any}(:g_tol=>1e-4), meth=LBFGS(),
             dofit::Bool=true, verbosity::Int=0)

    mask = isnothing(mask) ? I(length(y)) : mask

    er = ExpectReg(X, y; tau=tau, weights=weights, L2Pen=L2Pen, mask=mask)
    if dofit
        fit!(er; start=start, meth=meth, opts=opts, verbosity=verbosity)
    end
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
    return response(er) .- fitted(er)
end

function nobs(er::ExpectReg)
    return length(er.y)
end

function coefnames(mm::ExpectReg)
    (; tau) = mm

    na = mm.formula === nothing ? ["x$i" for i in 1:size(coef(mm), 1)] : coefnames(mm.formula.rhs)
    nax = [@sprintf("%s [%.2f]", x, y) for x in na, y in tau]
    return nax
end

function coeftable(mm::ExpectReg; level::Real=0.95)
    cc = coef(mm)
    se = stderror(mm)
    zz = cc ./ se
    p = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se*quantile(Normal(), (1-level)/2)
    levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)
    cn = coefnames(mm)
    CoefTable(hcat(vec(cc), vec(se), vec(zz), vec(p), vec(cc+ci), vec(cc-ci)),
              ["Coef.","Std. Error","z","Pr(>|z|)","Lower $levstr%","Upper $levstr%"],
              vec(cn), 4, 3)
end

function stderror(er::ExpectReg; adjust=:hc1, level::Real=0.95)

    (; tau) = er

    vc = vcov(er)
    se = hcat([sqrt.(diag(vc[Block(j, j)])) for j in eachindex(tau)]...)

    return se
end

"""
     vcov(er::ExpectReg)

Return the estimated variance-covariance matrix of the parameter estimates.
This function returns a block matrix with blocks corresponding to the expectile
values in 'tau'.  To access the (co)variance matrix between the j'th and k'th
value of tau, use 'vcov(er)[Block(j, k)]'.

M is a mask that defines the pairs of observations that may be non-independent.
The default value of M is the identity matrix, impling that all pairs of observations
are independent.

See 'vcov_array' to compute the same covariance as a Matrix.
"""
function vcov(er::ExpectReg; adjust=:hc1)

    (; X, y, beta, tau, weights, L2Pen, mask) = er

    if !(adjust in [:none, :hc1, :hc2, :hc3])
        error("Invalid 'adjust' in 'vcov'")
    end

    n, p = size(X)

    # Effective degrees of freedom of the mean model
    # This is efficient calculation of n - tr((I - H)*(I - H))
    # where H = X * ((X'*X + L2Pen) \ X')
    qq = qr(X)
    R = qq.R
    XtX = R' * R
    Q = XtX + L2Pen
    C = Q \ XtX
    edf = 2*tr(C) - tr(C*C)

    # We need the diagonal of H
    hd = sum(X .* (X / Q); dims=2)[:]

    # The bread for each expectile point
    B, EX = [], []
    r = residuals(er)
    for j in eachindex(tau)
        cr = check.(r[:, j], tau[j]) .* weights

        # Add a bit of diagonal regularization to make sure that the bread is invertible.
        # TODO need a better way to set this constant
        b = X' * Diagonal(cr) * X + L2Pen
        a, _ = eigen(Symmetric(b))
        c = (0.0001*p/n) * maximum(a)
        push!(B, b + c*I(p))

        eta = cr.* r[:, j]
        if adjust == :hc2
            eta ./= (1 .- hd)
        end
        push!(EX, Diagonal(eta) * X)
    end

    q = length(tau)
    V = BlockArray{Float64}(undef, fill(p, q), fill(p, q))
    for j1 in eachindex(tau)
        for j2 in 1:j1
            meat = EX[j1]' * mask * EX[j2]
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

function vcov_array(er::ExpectReg; adjust=:hc1)
    V = vcov(er; adjust=adjust)
    return Symmetric(Array(V))
end


function predict(er::ExpectReg)
    (; X, beta) = er
    return X * beta
end

function predict(er::ExpectReg, X::T) where{T<:AbstractArray}
    return X * coef(er)
end

"""
    crossfit(er::ExpectReg, targetf::Function; nfold=100)

Use cross-fitting to estimate a target quantity and provide confidence intervals
for the estimate.

`targetf` is a function that takes a fitted `ExpectReg` and a vector of
covariates `x`.  It returns the plug-in estimate of the target quantity at `x`
"""
function crossfit(er::ExpectReg, targetf::Function; M=I(nobs(er)), nfold=100)

    (; X, y, weights, tau, L2Pen) = er

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
        wgr = weights[itr]
        wge = weights[ite]

        ee = ExpectReg(Xtr, ytr; tau=tau, weights=wgr, L2Pen=L2Pen)
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

function fit(::Type{ExpectReg}, f::FormulaTerm, data; tau::Vector=Float64[0.5],
             L2Pen=nothing, weights=nothing, start=nothing, meth=LBFGS(),
             mask=nothing, dofit::Bool=true, verbosity::Int=0)

    f = apply_schema(f, schema(f, data), ExpectReg)
    y, X = modelcols(f, data)

    mask = isnothing(mask) ? I(length(y)) : mask

    kwds = Dict(:tau=>tau, :mask=>mask)
    if !isnothing(weights)
        kwds[:weights] = Tables.getcolumn(data, weights)
    end
    if !isnothing(L2Pen)
        kwds[:L2Pen] = L2Pen
    end

    er = ExpectReg(X, y; kwds...)
    er.formula = f
    if dofit
        fit!(er; start=start, meth=meth, verbosity=verbosity)
    end

    return er
end
