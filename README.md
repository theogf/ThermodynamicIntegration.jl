# ThermodynamicIntegration

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://theogf.github.io/ThermodynamicIntegration.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://theogf.github.io/ThermodynamicIntegration.jl/dev)
[![Build Status](https://github.com/theogf/ThermodynamicIntegration.jl/workflows/CI/badge.svg)](https://github.com/theogf/ThermodynamicIntegration.jl/actions)
[![Coverage Status](https://codecov.io/gh/theogf/ThermodynamicIntegration.jl/graph/badge.svg?token=EXHDBOH123)](https://codecov.io/gh/theogf/ThermodynamicIntegration.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

## Thermodynamic Integration

[Thermodynamic integration]() is a technique from physics to get an accurate estimate of the log evidence.
By creating a schedule going from the prior to the posterior and estimating the log likelihood at each step one gets a stable ad robust estimate of the log evidence.
You can find a good reference for the method in the paper ["Computing Bayes Factors Using Thermodynamic Integration"](https://academic.oup.com/sysbio/article/55/2/195/1620800?login=true)
Additionally I wrote a [short blog post](https://theogf.github.io/bayesiantribulations/blogposts/thermint/) about it.

For a different way of computing the evidence integral see also my [BayesianQuadrature package](https://github.com/theogf/BayesianQuadrature.jl).

## A simple example

A simple package to compute Thermodynamic Integration for computing the evidence in a Bayesian setting.
You need to provide the `logprior` and the `loglikelihood` as well as an initial sample:

```julia
    using Distributions, ThermodynamicIntegration
    D = 5
    prior = MvNormal(Diagonal(0.5 * ones(D))) # The prior distribution
    likelihood = MvNormal(Diagonal(2.0 * ones(D)))
    logprior(x) = logpdf(prior, x) # The log-prior function
    loglikelihood(x) = logpdf(likelihood, x) # The log-likelihood function

    alg = ThermInt(n_samples=5000) # We are going to sample 5000 samples at every step

    logZ = alg(logprior, loglikelihood, rand(prior)) # Compute the log evidence
    # -8.244829688529377
    true_logZ = -0.5 * (logdet(cov(prior) + cov(likelihood)) + D * log(2π)) # we compare twith the true value
    # -8.211990123364176
```

You can also simply pass a Turing model:

```julia
    using Turing
    @model function gauss(y)
        x ~ prior
        y ~ MvNormal(x, cov(likelihood))
    end

    alg = ThermInt(n_samples=5000)
    model = gauss(zeros(D))
    turing_logZ = alg(model)
    # # -8.211990123364176
```

## Parallel sampling

The algorithm also works on multiple threads by calling :

```julia
    alg = ThermInt(n_samples=5000) 
    logZ = alg(logprior, loglikelihood, rand(prior), TIThreads())
```

or on multiple processes:

```julia
    alg = ThermInt(n_samples=5000) 
    logZ = alg(logprior, loglikelihood, rand(prior), TIDistributed())
```

Note that you need to load `ThermodynamicIntegration` and other necessary external packages on your additional processes via `@everywhere`.

## Sampling methods

Right now sampling is based on [`AdvancedHMC.jl`](https://github.com/TuringLang/AdvancedHMC.jl), with the [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) AD backend.
To change the backend to [`Zygote`](https://github.com/FluxML/Zygote.jl) or [`ReverseDiff`](https://github.com/JuliaDiff/ReverseDiff.jl) (recommended for variables with large dimensions) you can do:

```julia
    using Zygote # (or ReverseDiff)
    ThermoDynamicIntegration.set_adbackend(:Zygote) # (or :ReverseDiff)
```

More samplers will be available in the future.

## Further options

You can disactivate the progress by calling `progress=false`

## Reference

[Lartillot, N., & Philippe, H. (2006). Computing Bayes factors using thermodynamic integration](https://academic.oup.com/sysbio/article/55/2/195/1620800?login=true)
