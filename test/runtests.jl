using ThermodynamicIntegration
using Distributions
using Test
using ForwardDiff
@testset "ThermodynamicIntegration.jl" begin
    D = 5
    prior = MvNormal(0.5 * ones(D))
    likelihood = MvNormal(2.0 * ones(D))
    logprior(x) = logpdf(prior, x)
    loglikelihood(x) = logpdf(likelihood, x)
    alg = ThermInt(n_samples=5000)
    logZ = alg(logprior, loglikelihood, rand(prior))
    # @model function gauss(y, β)
    #     x ~ prior
    #     y ~ MvNormal(x, cov(likelihood))^β
    # end

    # m  = gauss(zeros(D))
    # Turing.loglikelihood((y=zeros(D),), (x=rand(D),),m,nothing)
    true_logZ = -0.5 * (logdet(cov(prior) + cov(likelihood)) + D * log(2π))
    @test logZ ≈ true_logZ atol=1e-2
end
