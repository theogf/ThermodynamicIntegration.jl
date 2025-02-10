using .Turing: DynamicPPL, Prior

function (alg::ThermInt)(
    model::DynamicPPL.Model, ::TISerial=TISerial(); progress=SHOW_PROGRESS_BARS, kwargs...
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

function (alg::ThermInt)(
    model::DynamicPPL.Model, ::TIThreads; progress=SHOW_PROGRESS_BARS, kwargs...
)
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
    model::DynamicPPL.Model, ::TIDistributed; progress=SHOW_PROGRESS_BARS, kwargs...
)
    check_processes()
    progress && @warn "progress is not possible with distributed computing for now."
    # p = Progress(nsteps; enabled=progress, desc="TI sampling")
    pool = Distributed.CachingPool(Distributed.workers())
    function local_eval(β)
        return evaluate_loglikelihood(deepcopy(model), alg, β; kwargs...)
    end
    ΔlogZ = pmap(local_eval, pool, alg.schedule)
    return trapz(alg.schedule, ΔlogZ)
end

function evaluate_loglikelihood(model::DynamicPPL.Model, alg::ThermInt, β::Real)
    logprior = get_logprior(model)
    loglikelihood = get_loglikelihood(model)
    x_init = vec(Array(sample(model, Prior(), 1; progress=false))) # Bad ugly hack cause I don't know how to sample from the prior
    pj = PowerJoint(β, length(x_init), loglikelihood, logprior)
    samples = sample_powerlogπ(pj, alg, x_init)
    return mean(loglikelihood, samples)
end

"""
Build a logprior function acting on the flattened version of the parameters.
"""
function get_logprior(model)
    spl = DynamicPPL.SampleFromPrior()
    vi = DynamicPPL.VarInfo(model)
    return function f(z)
        varinfo = DynamicPPL.VarInfo(vi, spl, z)
        return DynamicPPL.logprior(model, varinfo)
    end
end

"""
Build a loglikelihood function acting on the flattened version of the parameters.
"""
function get_loglikelihood(model)
    spl = DynamicPPL.SampleFromPrior()
    vi = DynamicPPL.VarInfo(model)
    return function f(z)
        varinfo = DynamicPPL.VarInfo(vi, spl, z)
        return DynamicPPL.loglikelihood(model, varinfo)
    end
end
