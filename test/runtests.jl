using ThermodynamicIntegration
using Distributed
using Distributions
using Test
using LinearAlgebra
using Turing
using Suppressor
@testset "ThermodynamicIntegration.jl" begin
    include("thermint.jl")
end
