# Code for taking gradients with Enzyme
# This file is only loaded when Enzyme is imported

module MollyEnzymeExt

using Molly
using Enzyme
using FFTW

EnzymeRules.inactive(::typeof(is_on_gpu), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.default_strictness), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_strictness), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.has_infinite_boundary), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.n_infinite_dims), args...) = nothing
EnzymeRules.inactive(::typeof(random_coord), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocity), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocities), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocities!), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_force_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_energy_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_gbsa_n_threads), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.atoms_bonded_to_N), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.lookup_table), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.radius_classes), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.gb_log_scaling), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.default_show_progress), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.default_check_nans), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_array_nans), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.setup_progress), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.setup_progress_minimizer), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.next_nograd!), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.update_nograd!), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.needs_virial_schedule), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.parse_splitting), args...) = nothing
EnzymeRules.inactive(::typeof(use_neighbors), args...) = nothing
EnzymeRules.inactive(::typeof(find_neighbors), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.uses_gpu_neighbor_finder), args...) = nothing
EnzymeRules.inactive_type(::Type{<:NoNeighborFinder}) = true
EnzymeRules.inactive_type(::Type{<:GPUNeighborFinder}) = true
EnzymeRules.inactive_type(::Type{<:DistanceNeighborFinder}) = true
EnzymeRules.inactive_type(::Type{<:TreeNeighborFinder}) = true
EnzymeRules.inactive_type(::Type{<:CellListMapNeighborFinder}) = true
EnzymeRules.inactive(::typeof(visualize), args...) = nothing
EnzymeRules.inactive(::typeof(place_atoms), args...) = nothing
EnzymeRules.inactive(::typeof(place_diatomics), args...) = nothing
EnzymeRules.inactive(::typeof(read_frame!), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.kabsch_rotation_nograd), args...) = nothing

# Differentiable PME

# See https://github.com/EnzymeAD/Enzyme.jl/issues/2298
EnzymeRules.inactive(::typeof(plan_rfft ), args...) = nothing
EnzymeRules.inactive(::typeof(plan_brfft), args...) = nothing

# See rfft and brfft rrules in AbstractFFTs.jl

# The modes of the half spectrum that are not their own conjugate each stand for two modes
# of the full mesh. The real transforms fold that factor of two in, so the adjoints have to
# take it back out.
function hermitian_scale(charge_grid::AbstractArray{T, 3}, recip_grid) where T
    n = size(charge_grid, 1)
    return reshape([(isone(i) || 2*(i-1) == n ? one(T) : T(2))
                    for i in 1:size(recip_grid, 1)], :, 1, 1)
end

function EnzymeRules.augmented_primal(config, ::Const{typeof(Molly.grad_safe_fft!)}, t,
                                      charge_grid, recip_grid, fft_plan)
    Molly.grad_safe_fft!(charge_grid.val, recip_grid.val, fft_plan.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(config, ::Const{typeof(Molly.grad_safe_fft!)}, dret, tape,
                             charge_grid, recip_grid, fft_plan)
    scale = hermitian_scale(charge_grid.val, recip_grid.val)
    # The real grid is read but not written, so its shadow is added to
    charge_grid.dval .+= brfft(recip_grid.dval ./ scale, size(charge_grid.val, 1))
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(config, ::Const{typeof(Molly.grad_safe_bfft!)}, t,
                                      charge_grid, recip_grid, bfft_plan)
    Molly.grad_safe_bfft!(charge_grid.val, recip_grid.val, bfft_plan.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(config, ::Const{typeof(Molly.grad_safe_bfft!)}, dret, tape,
                             charge_grid, recip_grid, bfft_plan)
    scale = hermitian_scale(charge_grid.val, recip_grid.val)
    # The real grid is overwritten, so its shadow is consumed here
    recip_grid.dval .= scale .* rfft(charge_grid.dval)
    charge_grid.dval .= zero(eltype(charge_grid.dval))
    return (nothing, nothing, nothing)
end

# Calculate the gradient of a CV with respect to the input coordinates
# Works for systems with and without units
function Molly.cv_gradient(cv_type, coords, atoms=nothing, boundary=nothing, velocities=nothing)
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
        Const(velocities),
    )

    # Correct the units after the ustrip
    u = only(unit_arr)
    d_coords = d_coords .* u ./ unit(d_coords[1][1])^2

    return d_coords, cv_val_ustrip * u
end

end
