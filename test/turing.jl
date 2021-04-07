@testset "turing" begin
    D = 5
    prior = MvNormal(0.5 * ones(D))
    likelihood = MvNormal(2.0 * ones(D))
        @model function gauss(y)
        x ~ prior
        y ~ MvNormal(x, cov(likelihood))
    end
    m = gauss(zeros(D))
    alg = ThermInt(n_samples=5000)
    logZ = alg(m)
    true_logZ = -0.5 * (logdet(cov(prior) + cov(likelihood)) + D * log(2π))
    @test logZ ≈ true_logZ atol=1e-1

    logZparallel = alg(m, TIParallelThreads(); progress=false)
    @test logZ ≈ logZparallel atol=1e-1
end