module ExpectileRegression

using Optim, SparseArrays, Roots, Distributions, LinearAlgebra, BlockArrays, Random, StatsBase, ArnoldiMethod

import StatsAPI: fit, fit!, coef, vcov, residuals, fitted, response, nobs, StatisticalModel, predict

export ExpectReg, vcov_array, crossfit
export fit, fit!, coef, vcov, residuals, fitted, response, nobs, predict

include("expectreg.jl")
include("population_expectiles.jl")

end
