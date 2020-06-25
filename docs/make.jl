using Documenter
using Molly

makedocs(
    format = :html,
    sitename = "Molly.jl",
    pages = [
        "Home"          => "index.md",
        "Documentation" => "docs.md",
        "Examples"      => "examples.md",
        "API"           => "api.md",
    ],
    authors = "Joe G Greener"
)

deploydocs(
    repo = "github.com/JuliaMolSim/Molly.jl.git",
    julia = "1.0",
    osname = "linux",
    target = "build",
    deps = nothing,
    make = nothing
)
