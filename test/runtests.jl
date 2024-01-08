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

@testset "check formula" begin

    n = 500
    p = 3
    tau= [0.2, 0.5]
    rng = StableRNG(123)
    X = genAR(rng, n, p)
    wgt = 1 .+ 2 .* rand(rng, n)

    y, Ey, beta0 = simdat(rng, X)

    df = DataFrame(y=y)
    for j in 1:p
        df[:, "V$(j)"] = X[:,j]
    end

    # Fit with and without weights
    m1 = fit(ExpectReg, @formula(y ~ 0 + V1 + V2 + V3), df; tau=tau)
    m2 = fit(ExpectReg, @formula(y ~ 0 + V1 + V2 + V3), df; wgt=wgt, tau=tau)

    for m in [m1, m2]
        bhat = coef(m1)
        for j in eachindex(tau)
            beta = copy(beta0)
            beta[1] = ExpectileRegression.normal_expectile(tau[j])
            @test isapprox(beta, bhat[:, j], atol=0.1, rtol=0.1)
        end
    end
end

@testset "test gradient" begin

    n = 1000
    p = 3
    rng = StableRNG(123)

    X = randn(rng, n, p)
    X[:, 1] .= 1
    y, Ey, _ = simdat(rng, X)
    wgt = 1 .+ rand(rng, n)

    for l2pen in [false, true]
        for tau in [0.5, 0.8]
            er = l2pen ? ExpectReg(X, y; wgt=wgt, tau=[tau], L2Pen=I(p)) : ExpectReg(X, y; wgt=wgt, tau=[tau])
            for _ in 1:10
        	    f = beta -> ExpectileRegression.expectreg_loss(er, beta)
	            beta = randn(rng, p)
    	        ngrad = grad(central_fdm(5, 1), f, beta)[1]
    	        agrad = zeros(p)
    	        ExpectileRegression.expectreg_loss_grad!(er, agrad, beta)
		        @test isapprox(ngrad, agrad, atol=1e-6, rtol=1e-6)
		    end
        end
    end
end

@testset "basic fit test" begin

    n = 1000
    p = 3
    rng = StableRNG(123)

    X = randn(rng, n, p)
    X[:, 1] .= 1
    y, Ey, _ = simdat(rng, X)

    for tau in [0.1, 0.5, 0.8]
        m = ExpectReg(X, y; tau=[tau])
        fit!(m)
        bhat = coef(m)
        beta = [ExpectileRegression.normal_expectile(tau), -1, 0]
        @test isapprox(beta, bhat, atol=0.1, rtol=0.1)
    end
end

@testset "check vcov independent" begin

    n = 200
    p = 3
    tau = 0.2
    nrep = 1000
    rng = StableRNG(123)
    X = genAR(rng, n, p)
    wgt = 1 .+ 2 .* rand(rng, n)

    _, _, beta0 = simdat(rng, X)
    beta_true = copy(beta0)
    beta_true[1] += ExpectileRegression.normal_expectile(tau)

    Z1 = zeros(nrep, p)
    Z2 = zeros(nrep, p)
    B = zeros(nrep, p)
    for i in 1:nrep
        y, Ey, _ = simdat(rng, X)
        m = ExpectReg(X, y; tau=[tau], wgt=wgt)
        fit!(m)
        bhat = coef(m)
        v1 = vcov_array(m)
        v2 = vcov_array(m; M=sparse(I, n, n))
        B[i, :] = bhat
        Z1[i, :] = (bhat - beta_true) ./ sqrt.(diag(v1))
        Z2[i, :] = (bhat - beta_true) ./ sqrt.(diag(v2))
    end
    bias = mean(B; dims=1)[:] - beta_true
    @test isapprox(bias, zeros(p); atol=0.05, rtol=0.05)
    @test isapprox(std(Z1; dims=1)[:], ones(p); atol=0.05, rtol=0.05)
    @test isapprox(std(Z2; dims=1)[:], ones(p); atol=0.05, rtol=0.05)
end

@testset "check ridge" begin

    n = 200
    p = 3
    tau = 0.2
    rng = StableRNG(123)
    X = genAR(rng, n, p)
    wgt = 1 .+ 2 .* rand(rng, n)

    y, Ey, _ = simdat(rng, X)
    m = ExpectReg(X, y; tau=[tau], wgt=wgt, L2Pen=I(p))
    nm = []
    for i in 1:10
        m.L2Pen = 2 * m.L2Pen
        fit!(m)
        push!(nm, sum(abs2, coef(m)))
    end

    @test all(nm[2:end] .< nm[1:end-1])
end

@testset "check OLS" begin

    n = 200
    p = 3
    tau = 0.2
    rng = StableRNG(123)
    X = genAR(rng, n, p)
    wgt = 1 .+ 2 .* rand(rng, n)
    L2Pen = randn(p, p)
    L2Pen = L2Pen' * L2Pen

    for i in 1:10
        y, Ey, _ = simdat(rng, X)

        # When tau = 0.5 the calculations are done by direct linear algebra.
        m1 = ExpectReg(X, y; tau=[0.5], L2Pen=L2Pen, wgt=wgt)
        fit!(m1)
        m2 = ExpectReg(X, y; tau=[0.5000001], L2Pen=L2Pen, wgt=wgt)
        fit!(m2)

        @assert isapprox(coef(m1), coef(m2), rtol=1e-5, atol=1e-5)
    end
end

