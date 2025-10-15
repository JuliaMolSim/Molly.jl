using Documenter
using Molly

makedocs(
    sitename="Molly.jl",
    format=Documenter.HTML(
        prettyurls=(get(ENV, "CI", nothing) == "true"),
        size_threshold_ignore=["api.md"],
    ),
    modules=[Molly],
    pages=[
        "Home"                      => "index.md",
        "Documentation"             => "documentation.md",
        "Differentiable simulation" => "differentiable.md",
        "Free Energy"               => "freeenergy.md",
        "Examples"                  => "examples.md",
        "Exercises"                 => "exercises.md",
        "Publications"              => "publications.md",
        "Related software"          => "related.md",
        "Development"               => "development.md",
        "API"                       => "api.md",
    ],
)

deploydocs(
    repo="github.com/JuliaMolSim/Molly.jl.git",
    push_preview=true,
)
