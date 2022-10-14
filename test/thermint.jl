function test_basic_model(
    alg::ThermInt, method::ThermodynamicIntegration.TIEnsemble=TISerial(); D=5, atol=1e-1
)
    prior = MvNormal(0.5 * ones(D))
    likelihood = MvNormal(2.0 * ones(D))
    logprior(x) = logpdf(prior, x)
    loglikelihood(x) = logpdf(likelihood, x)
    logZ = alg(logprior, loglikelihood, rand(prior), method)
    true_logZ = -0.5 * (logdet(cov(prior) + cov(likelihood)) + D * log(2π))

    @test logZ ≈ true_logZ atol = atol
    @test_throws ArgumentError alg(logprior, loglikelihood, first(rand(prior)), method)
end

function test_basic_turing(
    alg::ThermInt, method::ThermodynamicIntegration.TIEnsemble=TISerial(); D=5, atol=1e-1
)
    prior = MvNormal(0.5 * ones(D))
    likelihood = MvNormal(2.0 * ones(D))
    @model function gauss(y)
        x ~ prior
        return y ~ MvNormal(x, cov(likelihood))
    end
    m = gauss(zeros(D))
    logZ = alg(m, method)
    true_logZ = -0.5 * (logdet(cov(prior) + cov(likelihood)) + D * log(2π))

    @test logZ ≈ true_logZ atol = atol
end

@testset "Basic model" begin
    alg = ThermInt(; n_samples=5000)
    # Test serialized version
    test_basic_model(alg, TISerial())
    test_basic_turing(alg, TISerial())

    # Test multithreaded version
    test_basic_model(alg, TIThreads())
    test_basic_turing(alg, TIThreads())

    # Test distributed version
    addprocs(
        Sys.iswindows() ? div(Sys.CPU_THREADS::Int, 2) : Sys.CPU_THREADS::Int;
        exeflags=`--project=$(Base.active_project())`,
    )
    @everywhere begin
        using ThermodynamicIntegration
        using Distributions
        using Turing
    end
    test_basic_model(alg, TIDistributed())
    test_basic_turing(alg, TIDistributed())
end
