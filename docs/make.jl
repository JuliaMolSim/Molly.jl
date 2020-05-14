using Documenter
using Molly

makedocs(
    format = :html,
    sitename = "Molly.jl",
    pages = [
        "Home"         => "index.md",
        "Documentation"=> "docs.md",
    ],
    authors = "Joe G Greener"
)

deploydocs(
    repo = "github.com/jgreener64/Molly.jl.git",
    julia = "1.0",
    osname = "linux",
    target = "build",
    deps = nothing,
    make = nothing
)
