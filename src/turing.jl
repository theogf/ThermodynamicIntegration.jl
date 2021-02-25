using .Turing: DynamicPPL, Prior

function (alg::ThermInt)(model::DynamicPPL.Model)
    ΔlogZ = @showprogress [evaluate_loglikelihood(model, alg, β) for β in alg.schedule]
    return trapz(alg.schedule, ΔlogZ)
end

function evaluate_loglikelihood(model::DynamicPPL.Model, alg::ThermInt, β::Real)
    powerlogπ = power_logjoint(model, β)
    loglikelihood = get_loglikelihood(model)
    x_init = vec(Array(sample(model, Prior(), 1))) # Bad ugly hack cause I don't know how to sample from the prior
    samples = sample_powerlogπ(powerlogπ, alg, x_init)
    return mean(loglikelihood, samples)
end

function power_logjoint(model, β)
    ctx = DynamicPPL.MiniBatchContext(DynamicPPL.DefaultContext(), β)
    spl = DynamicPPL.SampleFromPrior()
    vi = DynamicPPL.VarInfo(model)
    return function f(z)
        varinfo = DynamicPPL.VarInfo(vi, spl, z)
        model(varinfo, spl, ctx)
        return DynamicPPL.getlogp(varinfo)
    end
end

function get_loglikelihood(model)
    ctx = DynamicPPL.LikelihoodContext()
    spl = DynamicPPL.SampleFromPrior()
    vi = DynamicPPL.VarInfo(model)
    return function f(z)
        varinfo = DynamicPPL.VarInfo(vi, spl, z)
        model(varinfo, spl, ctx)
        return DynamicPPL.getlogp(varinfo)
    end
end
