using Documenter
using Molly

makedocs(
    sitename="Molly.jl",
    pages=[
        "Home"          => "index.md",
        "Documentation" => "docs.md"
    ],
    authors="Joe G Greener"
)

deploydocs(
    repo = "github.com/jgreener64/Molly.jl.git",
)
