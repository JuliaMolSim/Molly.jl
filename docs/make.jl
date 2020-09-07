using Documenter
using Molly

makedocs(
    sitename = "Molly.jl",
    pages = [
        "Home"                      => "index.md",
        "Documentation"             => "docs.md",
        "Differentiable simulation" => "differentiable.md",
        "Examples"                  => "examples.md",
        "API"                       => "api.md",
    ]
)

deploydocs(repo="github.com/JuliaMolSim/Molly.jl.git")
