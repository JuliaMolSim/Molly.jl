using Documenter
using Molly

makedocs(
    sitename = "Molly.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home"                      => "index.md",
        "Documentation"             => "docs.md",
        "Differentiable simulation" => "differentiable.md",
        "Examples"                  => "examples.md",
        "API"                       => "api.md",
    ]
)

deploydocs(
    repo="github.com/JuliaMolSim/Molly.jl.git",
    push_preview = true
)
