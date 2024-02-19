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
        return evaluate_loglikelihood(deepcopy(model), alg, β; kwargs...)
    end
    ΔlogZ = pmap(local_eval, pool, alg.schedule)
    return trapz(alg.schedule, ΔlogZ)
end

function evaluate_loglikelihood(model::DynamicPPL.Model, alg::ThermInt, β::Real)
    θ_init = rand(model)
    powerlogπ = power_logjoint(model, β)
    loglikelihood = get_loglikelihood(model)
    samples = sample_powerlogπ(powerlogπ, alg, θ_init)
    return mean(loglikelihood, samples)
end

function power_logjoint(model::DynamicPPL.Model, β::Real)
    return PowerProblem{LogDensityProblems.dimension(DynamicPPL.LogDensityFunction(model))}(;
        vi=DynamicPPL.VarInfo()θ -> logprior(model, θ), θ -> loglikelihood(model, θ), β
    )
end

get_loglikelihood(model) = θ -> loglikelihood(model, θ)
