

mutable struct ExpectLR{T<:Real} <: StatisticalModel

    # Data matrix
    X::AbstractMatrix # n x p

    # Fitted values
    F::Matrix{T} # n x p

    # Rank to fit
    r::Int

    # Expectile parameter
    tau::Float64

    # The row centers, (n-1)-dimensional.  The last row center
    # is fixed at zero for identification.
    rcen::Vector{T}

    # The column centers, p-dimensional
    ccen::Vector{T}

    # Factors
    U::Matrix{T} # n x r, orthogonal
    V::Matrix{T} # p x r

    opt_results
end

"""
    ExpectLR(X, r; tau=0.5)

Fit a rank-r model to a matrix X of the form X[i, j] = row[i] + col[j] + sum_k U[i,k]*V[j,k],
where U is orthogonal.  The target matrix X may have missing values.
"""
function ExpectLR(X, r; tau=0.5)
    n, p = size(X)
    F = zeros(n, p)
    rcen = zeros(n-1)
    ccen = zeros(p)
    U = zeros(n, r)
    V = zeros(p, r)
    return ExpectLR(X, F, r, tau, rcen, ccen, U, V, nothing)
end

function unpack(elr::ExpectLR, par::Vector{T}) where {T<:Real}

    (; X, r) = elr

    n, p = size(X)
    n1 = n - 1

    rc = @view par[1:n1]
    cc = @view par[n1+1:n1+p]
    u = @view par[n1+p+1:n1+p+n*r]
    v = @view par[n1+p+n*r+1:end]

    rcen = reshape(rc, n1, 1)
    ccen = reshape(cc, 1, p)
    U = reshape(u, n, r)
    V = reshape(v, p, r)
    return rcen, ccen, U, V
end

function expectlr_loss(elr::ExpectLR, par::Vector{T}) where {T<:Real}

    (; X, F, tau) = elr

    n, p = size(X)
    rcen, ccen, U, V = unpack(elr, par)
    F .= U * V' .+ ccen
    F[1:end-1, :] .+= rcen

    z = 0.0
    for i in 1:n
        for j in 1:p
            if !ismissing(X[i, j])
                z += eloss(X[i, j] - F[i, j], tau)
            end
        end
    end

    return z
end

function setfit!(elr::ExpectLR)
    (; F, U, V, rcen, ccen) = elr
    ccen1 = reshape(ccen, (1, length(ccen)))
    rcen1 = reshape(rcen, (length(rcen), 1))
    F .= U * V' .+ ccen1
    F[1:end-1, :] .+= rcen1
end

function expectlr_loss_grad!(elr::ExpectLR, grad::Vector{T}, par::Vector{T}) where {T<:Real}

    (; X, F, r, tau) = elr

    n, p = size(X)
    rcen, ccen, U, V = unpack(elr, par)
    F .= U * V' .+ ccen
    F[1:end-1, :] .+= rcen

    rceng, cceng, Ug, Vg = unpack(elr, grad)

    rceng .= 0
    cceng .= 0
    Ug .= 0
    Vg .= 0

    z = 0.0
    for i in 1:n
        for j in 1:p
            if !ismissing(X[i, j])
                g = elossgrad(X[i, j] - F[i, j], tau)
                if i < n
                    rceng[i] -= g
                end
                cceng[j] -= g
                for k in 1:r
                    Ug[i, k] -= g*V[j, k]
                    Vg[j, k] -= g*U[i, k]
                end
            end
        end
    end
end

function additive_start(elr::ExpectLR)
    (; X) = elr
    gm = mean(skipmissing(X))
    rm = [mean(skipmissing(x)) for x in eachrow(X)]
    cm = [mean(skipmissing(x)) for x in eachcol(X)]
    rm .-= gm
    x = last(rm)
    rm .-= x
    cm .+= x
    return rm, cm
end

function center(X, rm, cm)
    n, p = size(X)
    Z = copy(X)

    for i in 1:n
        for j in 1:p
            if !ismissing(Z[i, j])
                Z[i, j] -= rm[i] + cm[j]
            end
        end
    end

    return Z
end

function outermissing(X)
    n, p = size(X)
    Q = zeros(p, p)

    for j in 1:p
        for k in 1:j
            m = 0
            for i in 1:n
                if !ismissing(X[i, j]) && !ismissing(X[i, k])
                    Q[j, k] += X[i, j] * X[i, k]
                    m += 1
                end
            end

            # Since the data are centered, if a particular pair of columns have no corresponding
            # non-missing values, Q will be zero.
            if m > 0
                Q[j, k] *= n / m
            end
            if j != k
                Q[k, j] = Q[j, k]
            end
        end
    end

    return Symmetric(Q)
end

function dotmissing(Z, V)

    n, p = size(Z)
    pp, r = size(V)
    @assert p == pp

    Q = zeros(n, r)
    for i in 1:n
        for j in 1:r
            for k in 1:p
                if !ismissing(Z[i, k])
                    Q[i, j] += Z[i, k] * V[k, j]
                end
            end
        end
    end

    return Q
end

# TODO: these starting values will work best when tau ~ 0.5
function get_start(elr::ExpectLR)
    (; X, r) = elr
    n, p = size(X)
    n1 = n - 1
    d = n1 + p + n*r + p*r

    rm, cm = additive_start(elr)
    Z = center(X, rm, cm)

    # Z'Z
    R = outermissing(Z)
    vals, vecs = eigen(R)
    ii = sortperm(vals; rev=true)
    vals = vals[ii]
    vecs = vecs[:, ii]
    S = sqrt.(vals[1:r])
    V = vecs[:, 1:r] * Diagonal(S)

    U = dotmissing(Z, V / Diagonal(S.^2))

    # Orthogonalize
    Q = Symmetric(U' * U)
    vals, vecs = eigen(Q)
    J = vecs * Diagonal(sqrt.(1 ./ vals))

    U = U * J
    V = V / J'

    return vcat(rm[1:end-1], cm, vec(U), vec(V))
end


function admm(elr::ExpectLR, par::Vector{Float64}, rho::Float64, last::Bool; meth=LBFGS(), verbosity=0)

    (; r) = elr

    if verbosity > 0
        println("rho=", rho)
    end

    rcen0, ccen0, U0, V0 = unpack(elr, par)

    loss = function(par)
        rcen, ccen, U, V = unpack(elr, par)
        loss = expectlr_loss(elr, par)
        loss += rho * sum(abs2, U - U0) / 2
        return loss
    end

    grad! = function(g, par)
        rcen, ccen, U, V = unpack(elr, par)
        expectlr_loss_grad!(elr, g, par)
        rceng, cceng, Ug, Vg = unpack(elr, g)
        Ug .+= rho * (U - U0)
    end

    show_trace = verbosity > 0
    show_every = verbosity > 0 ? 1 : 0

    g_tol = last ? 0.1 : 1.0
    opts = Optim.Options(g_tol=g_tol, show_trace=show_trace, show_every=show_every)

    rr = optimize(loss, grad!, par, meth, opts)
    if !Optim.converged(rr)
        @warn("Expectile low rank fitting did not converge")
    end

    par = Optim.minimizer(rr)
    rcen, ccen, U, V = unpack(elr, par)
    F = U * V'
    u, s, v = svd(F)
    U .= u[:, 1:r]
    V .= v[:, 1:r] * Diagonal(s[1:r])

    return par, rr
end

function fit!(elr::ExpectLR; meth=LBFGS(), verbosity=0)

    (; X, r) = elr
    n, p = size(X)

    # Standardize before fitting, undo this transformation before returning results.
    loc = mean(skipmissing(X))
    scl = std(skipmissing(X))
    X_save = copy(X)
    X .= (X .- loc) ./ scl

    par = get_start(elr)
    if verbosity > 0
        println("Found starting values")
    end

    rho = 1.0
    rho_max = 10.0
    rr = nothing
    while rho < rho_max
        last = 2 * rho >= rho_max
        par, rr = admm(elr, par, rho, last; meth=meth, verbosity=verbosity)
        rho *= 1.5
    end

    rcen, ccen, U, V = unpack(elr, par)

    # Undo the standardizing transformation.
    rcen .*= scl
    ccen .*= scl
    V .*= scl
    ccen .+= loc

    elr.X .= X_save
    elr.rcen .= rcen[:]
    elr.ccen .= ccen[:]
    elr.U .= U
    elr.V .= V
    setfit!(elr)
    elr.opt_results = rr
end

function fit(::Type{ExpectLR}, X; r=1, tau=0.5, meth=LBFGS(), dofit=true, verbosity=0)

    elr = ExpectLR(X, r; tau=tau)

    if dofit
        fit!(elr; meth=meth, verbosity=verbosity)
    end

    return elr
end

function fitted(elr::ExpectLR)
    return elr.F
end
