function test_basic_model(alg::ThermInt, TImethod::ThermodynamicIntegration.TIEnsemble=TISerial(); D=5, atol=1e-1)
    prior = MvNormal(0.5 * ones(D))
    likelihood = MvNormal(2.0 * ones(D))
    logprior(x) = logpdf(prior, x)
    loglikelihood(x) = logpdf(likelihood, x)
    logZ = alg(logprior, loglikelihood, rand(prior), TImethod)
    true_logZ = -0.5 * (logdet(cov(prior) + cov(likelihood)) + D * log(2π))
    @test logZ ≈ true_logZ atol = atol
    @test_throws ArgumentError alg(logprior, loglikelihood, first(rand(prior)), TImethod)
    @test_throws ArgumentError alg(
        logprior, loglikelihood, first(rand(prior)), TImethod
    )
end

@testset "Basic model" begin
    alg = ThermInt(;n_samples=5000)
    # Test serialized version
    test_basic_model(alg, TISerial())

    # Test multithreaded version
    test_basic_model(alg, TIThreads())

    # Test distributed version
    addprocs(Sys.iswindows() ? div(Sys.CPU_THREADS::Int, 2) : Sys.CPU_THREADS::Int; exeflags=`--project=$(Base.active_project())`)
    @everywhere begin
        using ThermodynamicIntegration
        using Distributions
    end
    test_basic_model(alg, TIDistributed())
end
