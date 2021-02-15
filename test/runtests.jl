using ThermodynamicIntegration
using Distributions
using Test
using LinearAlgebra
using Turing
@testset "ThermodynamicIntegration.jl" begin
    include("thermint.jl")
    include("turing.jl")
end
