using ThermodynamicIntegration
using Documenter

makedocs(;
    modules=[ThermodynamicIntegration],
    authors="Theo Galy-Fajou <theo.galyfajou@gmail.com> and contributors",
    repo="https://github.com/theogf/ThermodynamicIntegration.jl/blob/{commit}{path}#L{line}",
    sitename="ThermodynamicIntegration.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://theogf.dev/ThermodynamicIntegration.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/theogf/ThermodynamicIntegration.jl")
