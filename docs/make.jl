using Documenter
using Molly

makedocs(
    sitename="Molly.jl",
    pages=[
        "Home"          => "index.md",
        "Documentation" => "docs.md"
    ],
)

deploydocs(
    repo = "github.com/jgreener64/Molly.jl.git",
)
