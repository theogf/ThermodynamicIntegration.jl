# ThermodynamicIntegration

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://theogf.github.io/ThermodynamicIntegration.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://theogf.github.io/ThermodynamicIntegration.jl/dev)
[![Build Status](https://github.com/theogf/ThermodynamicIntegration.jl/workflows/CI/badge.svg)](https://github.com/theogf/ThermodynamicIntegration.jl/actions)
[![Coverage Status](https://coveralls.io/repos/github/theogf/ThermodynamicIntegration.jl/badge.svg?branch=main)](https://coveralls.io/github/theogf/ThermodynamicIntegration.jl?branch=main)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

## Thermodynamic Integration

[Thermodynamic integration]() is a technique from physics to get an accurate estimate of the log evidence.
By creating a schedule going from the prior to the posterior and estimating the log likelihood at each step one gets a stable ad robust estimate of the log evidence.

## A simple example

A simple package to compute Thermodynamic Integration for computing the evidence in a Bayesian setting.
You need to provide the `logprior` and the `loglikelihood` as well as an initial sample:
```julia
    using Distributions, ThermodynamicIntegration
    D = 5
    prior = MvNormal(0.5 * ones(D)) # The prior distribution
    likelihood = MvNormal(2.0 * ones(D))
    logprior(x) = logpdf(prior, x) # The log-prior function
    loglikelihood(x) = logpdf(likelihood, x) # The log-likelihood function

    alg = ThermInt(n_samples=5000) # We are going to sample 5000 samples at every step

    logZ = alg(logprior, loglikelihood, rand(prior)) # Compute the log evidence
    # -8.244829688529377
    true_logZ = -0.5 * (logdet(cov(prior) + cov(likelihood)) + D * log(2π)) # we compare twith the true value
    # -8.211990123364176
```

You can also simply pass a Turing model :
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
    logZ = alg(logprior, loglikelihood, rand(prior), TIParallelThreads())
```

## Sampling methods

Right now sampling is based on [`AdvancedHMC.jl`](https://github.com/TuringLang/AdvancedHMC.jl), with the `ForwardDiff` AD backend.
To change the backend to `Zygote` or `ReverseDiff` (recommended for variables with large dimensions you can do:
```julia
    using Zygote # (or ReverseDiff)
    ThermoDynamicIntegration.set_adbackend(:Zygote) # (or :ReverseDiff)
```

More samplers will be available in the future.

## Further options

You can disactivate the progress by calling `progress=false`
```julia
    alg()
```