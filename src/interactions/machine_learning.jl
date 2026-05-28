export MachineLearningPotential

"""
    MachineLearningPotential(model, ps, st)

A potential energy surface defined by a machine learning model (e.g., a Lux.jl neural network).
"""
struct MachineLearningPotential{M, P, S}
    model::M
    ps::P
    st::S
end

# This specific function signature to calculate total system energy
function Molly.potential_energy(s::System, inter::MachineLearningPotential, neighbors=nothing)
    flat_coords = Float32.(reduce(vcat, s.coords))
    
    # Evaluating the neural network
    energy, _ = inter.model(flat_coords, inter.ps, inter.st)
    
    # Returning a scalar
    return energy[1] 
end

function Molly.forces(s::System, inter::MachineLearningPotential, neighbors=nothing)
    # Calculating the gradient of the energy with respect to coordinates
    flat_coords = Float32.(reduce(vcat, s.coords))
    grad = Zygote.gradient(c -> first(inter.model(c, inter.ps, inter.st)[1]), flat_coords)[1]
    
    # Reshaping the flat gradient (-∇E) back into Molly's Vector{SVector{3}} format
    n_atoms = length(s.coords)
    neg_grad = -grad
    return [SVector{3}(neg_grad[3i-2], neg_grad[3i-1], neg_grad[3i]) for i in 1:n_atoms]
end