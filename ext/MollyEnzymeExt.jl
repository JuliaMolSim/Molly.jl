# Code for taking gradients with Enzyme
# This file is only loaded when Enzyme is imported

module MollyEnzymeExt

using Molly
using Enzyme

EnzymeRules.inactive(::typeof(Molly.check_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.n_infinite_dims), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocity), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocities), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocities!), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_force_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_energy_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.atoms_bonded_to_N), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.lookup_table), args...) = nothing
EnzymeRules.inactive(::typeof(find_neighbors), args...) = nothing
EnzymeRules.inactive_type(::Type{DistanceNeighborFinder}) = nothing
EnzymeRules.inactive(::typeof(visualize), args...) = nothing
EnzymeRules.inactive(::typeof(place_atoms), args...) = nothing
EnzymeRules.inactive(::typeof(place_diatomics), args...) = nothing

end
