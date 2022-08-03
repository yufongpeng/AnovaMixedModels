module AnovaMixedModels

using Statistics, StatsBase, LinearAlgebra, Distributions, Reexport, Printf, GLM
@reexport using MixedModels, AnovaBase
import StatsBase: fit!, fit
import MixedModels: FeMat, createAL, reweight!, getθ,
                     _iscomparable, _criterion,
                     deviance, dof, dof_residual, nobs
import StatsModels: TableRegressionModel, vectorize, asgn
import AnovaBase: lrt_nested, formula, anova, nestedmodels, _diff, subformula, dof, dof_residual, deviance, nobs, coefnames

export anova_lme, lme, glme, calcdof

include("anova.jl")
include("fit.jl")
include("io.jl")

end