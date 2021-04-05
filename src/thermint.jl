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
struct ThermInt{AD,TRNG,V}
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
    return ThermInt(GLOBAL_RNG, schedule; n_samples=n_samples, n_warmup=n_warmup)
end

function ThermInt(rng::AbstractRNG; n_steps::Int, n_samples::Int=2000, n_warmup::Int=500)
    return ThermInt(rng, ((1:n_steps) ./ n_steps) .^ 5, n_samples, n_warmup, rng)
end

function ThermInt(;n_steps::Int=30, n_samples::Int=2000, n_warmup::Int=500)
    return ThermInt(
        GLOBAL_RNG, ((1:n_steps) ./ n_steps) .^ 5; n_samples=n_samples, n_warmup=n_warmup
    )
end

struct TIParallelThreads end

function (alg::ThermInt)(loglikelihood, logprior, x_init::AbstractVector; progress=true, kwargs...)
    p = ProgressMeter.Progress(length(alg.schedule); enabled=progress, desc="TI Sampling :")
    ΔlogZ = [
        begin
            ProgressMeter.next!(p);
            evaluate_loglikelihood(loglikelihood, logprior, alg, x_init, β; kwargs...)
        end
        for β in alg.schedule
    ]
    return trapz(alg.schedule, ΔlogZ)
end

function (alg::ThermInt)(loglikelihood, logprior, x_init::AbstractVector, ::TIParallelThreads; progress=true, kwargs...)
    Threads.nthreads() > 1 || warn("Only one thread available, parallelization will not happen. Start Julia with `--threads n`")
    nsteps = length(alg.schedule)
    nthreads = min(Threads.nthreads(), nsteps)
    ΔlogZ = zeros(Float64, nsteps)
    algs = [deepcopy(alg) for _ in 1:nthreads]
    p = ProgressMeter.Progress(length(alg.schedule); enabled=progress, desc="TI Sampling :")
    Threads.@threads for i in 1:nsteps
        id = Threads.threadid()
        ΔlogZ[i] = evaluate_loglikelihood(loglikelihood, logprior, algs[id], x_init, alg.schedule[i]; kwargs...)
        ProgressMeter.next!(p)
    end
    return trapz(alg.schedule, ΔlogZ)
end

function evaluate_loglikelihood(loglikelihood, logprior, alg::ThermInt, x_init, β::Real)
    powerlogπ(θ) = β * loglikelihood(θ) + logprior(θ)
    samples = sample_powerlogπ(powerlogπ, alg, x_init)
    x_init .= samples[end] # Update the initial sample to be the last one of the chain
    return mean(loglikelihood, samples)
end

function sample_powerlogπ(powerlogπ, alg::ThermInt, x_init)
    D = length(x_init)
    metric = DiagEuclideanMetric(D)
    hamiltonian = get_hamiltonian(metric, powerlogπ, alg)

    initial_ϵ = find_good_stepsize(hamiltonian, x_init)
    integrator = Leapfrog(initial_ϵ)

    proposal = AdvancedHMC.NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    samples, stats = sample(
        alg.rng,
        hamiltonian,
        proposal,
        x_init,
        alg.n_samples,
        adaptor,
        alg.n_warmup;
        verbose=false,
        progress=false,
    )
    return samples
end

function get_hamiltonian(metric, powerlogπ, ::ThermInt{:ForwardDiff})
    return Hamiltonian(metric, powerlogπ, ForwardDiff)
end
function get_hamiltonian(metric, powerlogπ, ::ThermInt{:Zygote})
    return Hamiltonian(metric, powerlogπ, Zygote)
end
function get_hamiltonian(metric, powerlogπ, ::ThermInt{:ReverseDiff})
    return Hamiltonian(metric, powerlogπ, ReverseDiff)
end
