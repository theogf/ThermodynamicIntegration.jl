using .Turing: DynamicPPL, Prior

function (alg::ThermInt)(
    model::DynamicPPL.Model, ::TISerial=TISerial(); progress=true, kwargs...
)
    nsteps = length(alg.schedule)
    p = Progress(nsteps; enabled=progress, desc="TI Sampling")
    ΔlogZ = [
        begin
            ProgressMeter.next!(p)
            evaluate_loglikelihood(model, alg, β)
        end for β in alg.schedule
    ]
    return trapz(alg.schedule, ΔlogZ)
end

function (alg::ThermInt)(model::DynamicPPL.Model, ::TIThreads; progress=true, kwargs...)
    check_threads()
    nsteps = length(alg.schedule)
    nthreads = min(Threads.nthreads(), nsteps)
    ΔlogZ = zeros(Float64, nsteps)
    algs = [deepcopy(alg) for _ in 1:nthreads]
    p = Progress(nsteps; enabled=progress, desc="TI Multithreaded Sampling")
    Threads.@threads for i in 1:nsteps
        id = Threads.threadid()
        ΔlogZ[i] = evaluate_loglikelihood(model, algs[id], alg.schedule[i]; kwargs...)
        ProgressMeter.next!(p)
    end
    return trapz(alg.schedule, ΔlogZ)
end

function (alg::ThermInt)(
    model::DynamicPPL.Model, ::TIDistributed; progress=false, kwargs...
)
    check_processes()
    progress && @warn "progress is not possible with distributed computing for now."
    # p = Progress(nsteps; enabled=progress, desc="TI sampling")
    pool = Distributed.CachingPool(Distributed.workers())
    function local_eval(β)
        return evaluate_loglikelihood(copy(model), alg, β; kwargs...)
    end
    ΔlogZ = pmap(local_eval, pool, alg.schedule)
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
