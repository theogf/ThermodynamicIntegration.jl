using ThermodynamicIntegration
using Distributions
using Test
using ForwardDiff
using LinearAlgebra
@testset "ThermodynamicIntegration.jl" begin
    D = 5
    prior = MvNormal(0.5 * ones(D))
    likelihood = MvNormal(2.0 * ones(D))
    logprior(x) = logpdf(prior, x)
    loglikelihood(x) = logpdf(likelihood, x)
    alg = ThermInt(n_samples=5000)
    logZ = alg(logprior, loglikelihood, rand(prior))

    Σ = Diagonal(inv(inv(cov(prior)) + inv(cov(likelihood))))
    posterior = MvNormal(Diagonal(Σ))
    true_logZ = 0.5 * (logdetcov(posterior) - D * log(2π))
    @test logZ ≈ true_logZ atol=1e-2
end
