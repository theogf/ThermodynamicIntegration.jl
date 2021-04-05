using ThermodynamicIntegration
using Distributions


nsamples = 10000
function do_therm_int_parallel(nsamples)
    prior = MvNormal(ones(3))
    logprior(θ) = logpdf(prior, θ)
    loglikelihood(θ) = logpdf(prior, θ)
    alg = ThermInt(n_steps=30, n_samples=nsamples)
    logZ = alg(logprior, loglikelihood, rand(prior), TIParallelThreads())
end

@time do_therm_int_parallel(nsamples)

function do_therm_int(nsamples)
    prior = MvNormal(ones(3))
    logprior(θ) = logpdf(prior, θ)
    loglikelihood(θ) = logpdf(prior, θ)
    alg = ThermInt(n_steps=30, n_samples=nsamples)
    logZ = alg(logprior, loglikelihood, rand(prior))
end

@time do_therm_int(nsamples)