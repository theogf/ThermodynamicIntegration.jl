"""
    ThermInt([rng::AbstractRNG]; n_steps=30, n_samples=2000, n_warmup=500)
    ThermInt([rng::AbstractRNG], schedule; n_samples=2000, n_warmup=500)

- `schedule` can be any iterable object
- `n_steps` is the number of steps for the schedule using the formula
`(1:n_steps) ./ n_steps).^5`

A `ThermInt` object can then be used as a function:

```julia
    alg = ThermInt(30)
    alg(loglikelihood, logprior, x_init)
```
"""
struct ThermInt{AD,TRNG<:AbstractRNG,V}
    schedule::V
    n_samples::Int
    n_warmup::Int
    rng::TRNG
end

function ThermInt(rng::AbstractRNG, schedule; n_samples::Int=2000, n_warmup::Int=500)
    return ThermInt{ADBACKEND[],typeof(rng),typeof(schedule)}(
        schedule, n_samples, n_warmup, rng
    )
end

function ThermInt(schedule; n_samples::Int=2000, n_warmup::Int=500)
    return ThermInt(default_rng(), schedule; n_samples=n_samples, n_warmup=n_warmup)
end

function ThermInt(rng::AbstractRNG; n_steps::Int, n_samples::Int=2000, n_warmup::Int=500)
    return ThermInt(
        rng, range(0, 1; length=n_steps) .^ 5; n_samples=n_samples, n_warmup=n_warmup
    )
end

function ThermInt(; n_steps::Int=30, n_samples::Int=2000, n_warmup::Int=500)
    return ThermInt(
        default_rng(),
        range(0, 1; length=n_steps) .^ 5;
        n_samples=n_samples,
        n_warmup=n_warmup,
    )
end

abstract type TIEnsemble end
struct TISerial <: TIEnsemble end
struct TIThreads <: TIEnsemble end
@deprecate TIParallelThreads TIThreads
struct TIDistributed <: TIEnsemble end

function (alg::ThermInt)(
    loglikelihood,
    logprior,
    x_init::AbstractVector,
    ::TISerial=TISerial();
    progress=true,
    kwargs...,
)
    p = ProgressMeter.Progress(length(alg.schedule); enabled=progress, desc="TI Sampling:")
    ΔlogZ = map(alg.schedule) do β
        val = evaluate_loglikelihood(loglikelihood, logprior, alg, x_init, β; kwargs...)
        ProgressMeter.next!(p)
        val
    end
    return trapz(alg.schedule, ΔlogZ)
end

function check_threads()
    return Threads.nthreads() > 1 ||
           @warn "Only one thread available, parallelization will not happen. Start Julia with `julia --threads n`"
end

function (alg::ThermInt)(
    loglikelihood, logprior, θ_init::AbstractVector, ::TIThreads; progress=true, kwargs...
)
    check_threads()
    nsteps = length(alg.schedule)
    nthreads = min(Threads.nthreads(), nsteps)
    ΔlogZ = zeros(Float64, nsteps)
    algs = [deepcopy(alg) for _ in 1:nthreads]
    p = ProgressMeter.Progress(
        length(alg.schedule); enabled=progress, desc="TI Multithreaded Sampling:"
    )
    Threads.@threads for i in 1:nsteps
        id = Threads.threadid()
        ΔlogZ[i] = evaluate_loglikelihood(
            loglikelihood, logprior, algs[id], θ_init, alg.schedule[i]; kwargs...
        )
        ProgressMeter.next!(p)
    end
    return trapz(alg.schedule, ΔlogZ)
end

function check_processes()
    return Distributed.nworkers() > 1 ||
           @warn "Only one process available, parallelization will not happen. Start Julia with `julia -p n`"
end

function (alg::ThermInt)(
    loglikelihood,
    logprior,
    θ_init::AbstractVector,
    ::TIDistributed;
    progress=false,
    kwargs...,
)
    check_processes()
    progress && @warn "progress is not possible with distributed computing for now."
    # p = ProgressMeter.Progress(
    # length(alg.schedule); enabled=progress, desc="TI (multiple processes) Sampling :"
    # )

    pool = Distributed.CachingPool(Distributed.workers())
    function local_eval(β)
        return evaluate_loglikelihood(loglikelihood, logprior, alg, θ_init, β; kwargs...)
    end
    ΔlogZ = pmap(local_eval, pool, alg.schedule)
    return trapz(alg.schedule, ΔlogZ)
end

function (alg::ThermInt)(
    loglikelihood, logprior, x_init::Real, method::TIEnsemble=TISerial(); kwargs...
)
    throw(
        ArgumentError(
            "Given your `x_init`, it looks like you are trying to work" *
            "with a model with one scalar random variable." *
            "Unfortunately, only an `AbstractVector` can be passed." *
            "An easy work around it to do `x_init`->`[x_init]` and to modify" *
            "`logprior(x) = f(x)` by `logprior(x)=f(only(x))` (and similarly for `loglikelihood`",
        ),
    )
end

struct PowerProblem{N,LP,LL}
    logprior::LP
    loglikelihood::LL
    β::Float64
    function PowerProblem{N}(logprior::T1, loglikelihood::T2, β::Real) where {N,T1,T2}
        return new{N,T1,T2}(logprior, loglikelihood, β)
    end
end

function PowerProblem(logprior, loglikelihood, β::Real, θ_init)
    return PowerProblem{length(θ_init)}(logprior, loglikelihood, β)
end

function LogDensityProblems.capabilities(::Type{<:PowerProblem})
    return LogDensityProblems.LogDensityOrder{0}()
end
LogDensityProblems.dimension(::PowerProblem{N}) where {N} = N
function LogDensityProblems.logdensity(pp::PowerProblem, θ)
    return pp.β * pp.loglikelihood(θ) + pp.logprior(θ)
end

function evaluate_loglikelihood(loglikelihood, logprior, alg::ThermInt, θ_init, β::Real)
    powerlogπ(θ) = β * loglikelihood(θ) + logprior(θ)
    powerlogπ = PowerProblem(logprior, loglikelihood, β, θ_init)
    samples = sample_powerlogπ(powerlogπ, alg, θ_init)
    θ_init .= samples[end] # Update the initial sample to be the last one of the chain
    return mean(loglikelihood, samples)
end

function sample_powerlogπ(powerlogπ, alg::ThermInt, θ_init)
    N = length(θ_init)
    metric = DiagEuclideanMetric(N)
    hamiltonian = Hamiltonian(metric, powerlogπ, ADBACKEND[])

    initial_ϵ = find_good_stepsize(hamiltonian, θ_init)
    integrator = Leapfrog(initial_ϵ)

    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    samples, _ = sample(
        alg.rng,
        hamiltonian,
        kernel,
        θ_init,
        alg.n_samples,
        adaptor,
        alg.n_warmup;
        verbose=false,
        progress=false,
    )
    return samples
end
