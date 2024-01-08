using KernelFunctions
using LinearAlgebra

# For testing
using ExpectileRegression
using StableRNGs
using Random
using Statistics
include("utils.jl")

mutable struct MultiKernel{T<:Real}

    # The kernels to use for producing basis functions
    kernels::Vector

    # The covariate data used to fit the model.
    X::Matrix{T}

    # Basis functions for each kernel
    B::Vector{Matrix{T}}

    # Eigenvalues, such that K[i] ~ B[i] * Diagonal(a[i]) * B[i]'
    a::Vector{Vector{T}}

    # A penalty matrix for each kernel
    Kpen::Vector{Matrix{T}}

    # Weight for each penalty matrix
    pwt::Vector{Float64}
end

function setup_kernel(ker, X; maxdim=10, mineig=1e-4)

    n, p = size(X)
    K = kernelmatrix(ker, RowVecs(X))
    a, b = eigen(Symmetric(K))

    j = findfirst(a .> mineig)
    j = max(j, n - maxdim + 1)
    B = b[:, j:end]
    a = a[j:end]

    Kpen = B * Diagonal(1 ./ a) * B'

    # Scale so that the penalty matrix has maximum eigenvalue equal to 1.
    Kpen .*= minimum(a)

    # K ~ B * diag(a) * B'
    # B ~ K * B * diag(1/a)

    return B, a, Kpen
end

function MultiKernel(kernels, X; maxdim=10, mineig=1e-4, pwt=ones(length(kernels)))

    B, a, Kpen = Matrix{Float64}[], Vector{Float64}[], Matrix{Float64}[]

    for ker in kernels
        BB, aa, Kp = setup_kernel(ker, X; maxdim=maxdim, mineig=mineig)
        push!(B, BB)
        push!(a, aa)
        push!(Kpen, Kp)
    end

    return MultiKernel(kernels, X, B, a, Kpen, pwt)
end

function basis(mk::MultiKernel)
    (; B) = mk
    return hcat(B...)
end

function penalty(mk::MultiKernel)
    (; Kpen, pwt) = mk
    return sum([a*b for (a,b) in zip(Kpen, pwt)])
end

function predict_mat(mk::MultiKernel, Xnew)
    (; kernels, X, B, a) = mk

    XX = []
    for (ker, BB, aa) in zip(kernels, B, a)
        K = kernelmatrix(ker, RowVecs(Xnew), RowVecs(X))
        B1 = K * BB * Diagonal(1 ./ aa)
        push!(XX, B1)
    end

    return hcat(XX...)
end

function test_multi1()

    rng = StableRNG(123)
    ker1 = with_lengthscale(SqExponentialKernel(), 0.25)
    ker2 = with_lengthscale(PolynomialKernel(degree=1, c=0), 5.0)

    n = 300
    p = 3
    icc = 0.4

    X = genAR(Random.default_rng(), n, p, icc)
    mk = MultiKernel([ker1, ker2], X)
    Z = basis(mk)
    ZZ = predict_mat(mk, X)
    @assert isapprox(Z, ZZ)
end

function test_multi2()

    rng = StableRNG(123)
    ker0 = ConstantKernel()
    ker1 = with_lengthscale(PolynomialKernel(degree=1, c=0), 1.0)
    ker2 = with_lengthscale(PolynomialKernel(degree=2, c=0), 1.0)

    n = 300
    p = 10
    icc = 0.4

    X = genAR(rng, n, p, icc)
    regfun = X -> 1 .+ X[:, 3] + X[:, 2] + X[:, 2] .* X[:, 3] + X[:, 3].^2 + X[:, 8] .* X[:, 9]
    Ey = regfun(X)
    y = Ey + randn(rng, n)

    mk = MultiKernel([ker0, ker1, ker2], X; maxdim=20)
    Z = basis(mk)
    Pen = penalty(mk)
    Pen = Z' * Pen * Z

    lam = 0.1
    er = fit(ExpectReg, Z, y; L2Pen=lam*Pen, tau=[0.5])
    cc = coef(er)[:, 1]

    yhat = Z * cc
    @assert cor(Ey, yhat) > 0.9
    @assert abs(mean((yhat - Ey).^2) - 1) < 0.1

    # Predict on an independent set
    X = genAR(rng, n, p, icc)
    Ey = regfun(X)
    Z = predict_mat(mk, X)
    yhat = Z * cc
    @assert cor(Ey, yhat) > 0.8
    @assert (mean((yhat - Ey).^2) - 1) < 0.6
end

#test_multi1()
#test_multi2()
