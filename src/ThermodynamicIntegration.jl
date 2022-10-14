module ThermodynamicIntegration

using AdvancedHMC
using Distributed
using ForwardDiff
using ProgressMeter
using Random
using Requires
using Statistics
using Trapz

export ThermInt
export TISerial, TIThreads, TIDistributed

const GLOBAL_RNG = Random.MersenneTwister(42)

const ADBACKEND = Ref(:ForwardDiff)

set_adbackend(ad::String) = set_adbackend(Symbol(ad))
set_adbackend(ad::Symbol) = set_adbackend(Val(ad))
set_adbackend(::Val{:ForwardDiff}) = ADBACKEND[] = :ForwardDiff
function set_adbackend(::Any)
    return error(
        "ad should be :ForwardDiff, :Zygote or :ReverseDiff\n" *
        "For Zygote and ReverseDiff, make sure to have" *
        " `using Zygote/ReverseDiff` in your script",
    )
end

include("thermint.jl")

function __init__()
    @require Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0" include("turing.jl")
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        set_adbackend(::Val{:Zygote}) = ADBACKEND[] = :Zygote
    end
    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        set_adbackend(::Val{:ReverseDiff}) = ADBACKEND[] = :ReverseDiff
    end
end

end
