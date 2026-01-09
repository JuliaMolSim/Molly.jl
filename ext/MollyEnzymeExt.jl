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
EnzymeRules.inactive(::typeof(Molly.ref_kabsch), args...) = nothing

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

# Calculate gradient of collective variable to bias simulation
# Works for systems with and without units
function Molly.cv_gradient(cv_type, coords, atoms, boundary, velocities)
    d_coords = zero(coords)
    unit_arr = Any[u"nm"]

    _, cv_val_ustrip = autodiff(
        set_runtime_activity(ReverseWithPrimal), # set_runtime_activity necessary for units
        Molly.calculate_cv_ustrip!,
        Active,
        Const(unit_arr),
        Const(cv_type),
        Duplicated(coords, d_coords),
        Const(atoms),
        Const(boundary),
        Const(velocities)
    )

    # Correct the units after the ustrip
    u = only(unit_arr)
    d_coords = d_coords .* u ./ unit(d_coords[1][1])^2

    return d_coords, cv_val_ustrip * u
end

end
