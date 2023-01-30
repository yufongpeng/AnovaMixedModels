# ================================================================================
# IO
# anovatable api
function anovatable(aov::AnovaResult{<: FullModel{M}, FTest}; rownames = prednames(aov)) where {M <: LinearMixedModel}
    AnovaTable([
                    dof(aov), 
                    dof_residual(aov), 
                    teststat(aov), 
                    pval(aov)
                ],
              ["DOF", "Res.DOF", "F value", "Pr(>|F|)"],
              rownames, 4, 3)
end

function anovatable(aov::AnovaResult{NestedModels{M, N}, LRT}; 
                    rownames = string.(1:N)) where {M <: Union{GLM_MODEL, MixedModel}, N}
    if last(aov.anovamodel.model).optsum.REML 
        AnovaTable([
                        dof(aov), 
                        [NaN, _diff(dof(aov))...], 
                        dof_residual(aov), 
                        deviance(aov), 
                        teststat(aov), 
                        pval(aov)
                    ],
                ["DOF", "ΔDOF", "Res.DOF", "-2 logLik", "χ²", "Pr(>|χ²|)"],
                rownames, 6, 5)
    else
        AnovaTable([
                        dof(aov), 
                        [NaN, _diff(dof(aov))...], 
                        dof_residual(aov), 
                        aic.(aov.anovamodel.model),
                        bic.(aov.anovamodel.model),
                        deviance(aov), 
                        teststat(aov), 
                        pval(aov)
                    ],
                ["DOF", "ΔDOF", "Res.DOF", "AIC", "BIC", "-2 logLik", "χ²", "Pr(>|χ²|)"],
                rownames, 8, 7)
    end
end