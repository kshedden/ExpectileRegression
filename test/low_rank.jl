using Missings

function simLR(rng, n, p, r, sig; pmiss=0.0)

    row = randn(rng, n, 1)
    col = randn(rng, 1, p)
    U = randn(rng, n, r)
    V = randn(rng, p, r)
    F = row .+ col .+ U*V'

    X = F + sig*randn(rng, n, p)
    X = allowmissing(X)

    for i in 1:n
        for j in 1:p
            if rand(rng) < pmiss
                X[i, j] = missing
            end
        end
    end

    return X, F
end

@testset "ExpectLR gradient" begin

    rng = StableRNG(123)
    n = 10
    p = 3
    n1 = n - 1

    X = randn(rng, n, p)

    for tau in [0.5, 0.75]
        for r in [0, 1, 2]
            d = n1 + p + n*r + p*r
            esvd = ExpectLR(X, r; tau=tau)
            for _ in 1:5
        	    f = par -> ExpectileRegression.expectlr_loss(esvd, par)
	            par = randn(rng, d)
    	        ngrad = grad(central_fdm(5, 1), f, par)[1]
    	        agrad = zeros(d)
    	        ExpectileRegression.expectlr_loss_grad!(esvd, agrad, par)
		        @test isapprox(ngrad, agrad, atol=1e-6, rtol=1e-6)
		    end
        end
    end

end

@testset "ExpectLR starting values" begin

    rng = StableRNG(123)
    n = 100
    p = 30

    for r in [0, 1, 2]
        X, F0 = simLR(rng, n, p, r, 0.0)
        elr = fit(ExpectLR, X; r=r, dofit=false)
        start = ExpectileRegression.get_start(elr)
        row, col, U, V = ExpectileRegression.unpack(elr, start)
        elr.rcen .= row[:]
        elr.ccen .= col[:]
        elr.U .= U
        elr.V .= V
        ExpectileRegression.setfit!(elr)
        F = fitted(elr)
        rmse = sqrt(mean((X - F).^2))
        @test isapprox(rmse, 0, atol=0.02, rtol=0.1)
    end
end

@testset "ExpectLR fitting" begin

    rng = StableRNG(123)
    n = 1000
    p = 100

    for tau in [0.75, 0.5]
        for r in [0, 2, 4]
            for sig in [0.3]
                for pmiss in [0.1]
                    X, F0 = simLR(rng, n, p, r, sig; pmiss=pmiss)

                    # The target values
                    F1 = F0 .+ sig*ExpectileRegression.normal_expectile(tau)

                    # Estimate
                    elr = fit(ExpectLR, X; r=r, tau=tau, verbosity=0)
                    F = fitted(elr)

                    sca = sqrt(mean(F1.^2))
                    bias = mean(F) - mean(F1)
                    @test abs(bias) < 0.01 * sca
                    rmse = sqrt(mean((F - F1).^2))
                    @test rmse / sca <= 0.05

                    if tau == 0.5
                        # This value is known from the simulation
                        rmse = sqrt(mean(skipmissing((X - F).^2)))
                        @test isapprox(rmse, sig, atol=0.02, rtol=0.1)
                    end
                end
            end
        end
    end
end
