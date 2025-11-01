# Code for taking gradients with Enzyme
# This file is only loaded when Enzyme is imported

module MollyEnzymeExt

using Molly
using Enzyme
using FFTW

EnzymeRules.inactive(::typeof(Molly.check_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.n_infinite_dims), args...) = nothing
EnzymeRules.inactive(::typeof(random_coord), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocity), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocities), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocities!), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_force_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_energy_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.atoms_bonded_to_N), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.lookup_table), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.needs_virial_schedule), args...) = nothing
EnzymeRules.inactive(::typeof(find_neighbors), args...) = nothing
EnzymeRules.inactive_type(::Type{DistanceNeighborFinder}) = nothing
EnzymeRules.inactive(::typeof(visualize), args...) = nothing
EnzymeRules.inactive(::typeof(place_atoms), args...) = nothing
EnzymeRules.inactive(::typeof(place_diatomics), args...) = nothing
EnzymeRules.inactive(::typeof(read_frame!), args...) = nothing

# Differentiable PME

# See https://github.com/EnzymeAD/Enzyme.jl/issues/2298
EnzymeRules.inactive(::typeof(plan_fft ), args...) = nothing
EnzymeRules.inactive(::typeof(plan_bfft), args...) = nothing

# See fft and bfft rrules in AbstractFFTs.jl
function EnzymeRules.augmented_primal(config, ::Const{typeof(Molly.grad_safe_fft!)}, t,
                                      charge_grid, fft_plan)
    fft_plan.val * charge_grid.val
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(config, ::Const{typeof(Molly.grad_safe_fft!)}, dret, tape,
                             charge_grid, fft_plan)
    charge_grid.dval .= bfft(charge_grid.dval)
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(config, ::Const{typeof(Molly.grad_safe_bfft!)}, t,
                                      charge_grid, bfft_plan)
    bfft_plan.val * charge_grid.val
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(config, ::Const{typeof(Molly.grad_safe_bfft!)}, dret, tape,
                             charge_grid, bfft_plan)
    charge_grid.dval .= fft(charge_grid.dval)
    return (nothing, nothing)
end

end
