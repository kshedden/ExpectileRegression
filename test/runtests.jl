using ExpectileRegression
using Test
using StableRNGs
using FiniteDifferences
using LinearAlgebra
using Statistics
using SparseArrays
using DataFrames
using StatsModels

function simdat(rng, X)
    n, p = size(X)
    beta = Float64[0, -1, 0]
    Ey = X * beta
    y = Ey + randn(rng, n)
    return y, Ey, beta
end

function genAR(rng, n, p)
    X = randn(rng, n, p)
    r = 0.7
    for j in 2:p
        X[:, j] = r*X[:, j-1] + sqrt(1-r^2)*X[:, j]
    end
    X[:, 1] .= 1
    return X
end

include("expectreg.jl")
include("low_rank.jl")
