module ThermodynamicIntegration

using AdvancedHMC
using Requires
using Trapz
using Statistics
using ForwardDiff
using ProgressMeter

export ThermInt

struct ThermInt{V}
    schedule::V
    n_samples::Int
    n_warmup::Int
end

function ThermInt(schedule; n_samples::Int=2000, n_warmup::Int=500)
    ThermInt(schedule, n_samples, n_warmup)
end

function ThermInt(n_steps::Int=30; n_samples::Int=2000, n_warmup::Int=500)
    ThermInt(((1:n_steps) ./ n_steps).^5, n_samples, n_warmup)
end

function (alg::ThermInt)(loglikelihood, logprior, θ::AbstractVector)
    ΔlogZ = @showprogress [evaluate_loglikelihood(loglikelihood, logprior, alg, θ, β) for β in alg.schedule]
    return trapz(alg.schedule, ΔlogZ)
end

function evaluate_loglikelihood(loglikelihood, logprior, alg::ThermInt, θ, β::Real)
    logπ(θ) = β * loglikelihood(θ) + logprior(θ)

    D = length(θ)
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, logπ, ForwardDiff)

    initial_ϵ = find_good_stepsize(hamiltonian, θ)
    integrator = Leapfrog(initial_ϵ)

    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    samples, stats = sample(hamiltonian, proposal, θ, alg.n_samples, adaptor, alg.n_warmup; verbose=false, progress=false)
    θ .= samples[end]
    return mean(loglikelihood, samples) 
end

end
