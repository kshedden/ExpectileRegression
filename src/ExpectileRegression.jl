module ExpectileRegression

using Optim, SparseArrays, Roots, Distributions, LinearAlgebra, BlockArrays, Random, StatsBase, ArnoldiMethod, Printf, Tables
using StatsModels: FormulaTerm, @formula, schema, apply_schema, modelcols

import StatsAPI: fit, fit!, coef, vcov, stderror, residuals, fitted, response, nobs, StatisticalModel, predict, coeftable, coefnames

export ExpectReg, vcov_array, crossfit
export ExpectLR
export fit, fit!, coef, vcov, stderror, residuals, fitted, response, nobs, predict, coeftable, coefnames

include("common.jl")
include("expectreg.jl")
include("low_rank.jl")
include("population_expectiles.jl")

end
