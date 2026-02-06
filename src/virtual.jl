# Virtual sites

export
    TwoParticleAverageSite,
    ThreeParticleAverageSite,
    OutOfPlaneSite,
    place_virtual_sites!,
    distribute_forces!

struct VirtualSiteTemplate{T, IC}
    type::Int # 1/2/3 for TwoParticleAverageSite/ThreeParticleAverageSite/OutOfPlaneSite
    name::String
    atom_name_1::String
    atom_name_2::String
    atom_name_3::String
    weight_1::T
    weight_2::T
    weight_3::T
    weight_12::T
    weight_13::T
    weight_cross::IC # Units are 1/L
end

struct VirtualSite{T, IC}
    type::Int # 1/2/3 for TwoParticleAverageSite/ThreeParticleAverageSite/OutOfPlaneSite
    atom_ind::Int
    atom_1::Int
    atom_2::Int
    atom_3::Int
    weight_1::T
    weight_2::T
    weight_3::T
    weight_12::T
    weight_13::T
    weight_cross::IC # Units are 1/L
end

"""


Optional weight_cross
"""
function TwoParticleAverageSite(atom_ind, atom_1, atom_2, weight_1::T, weight_2::T,
                                weight_cross=(zero(T) * u"nm^-1")) where T
    return VirtualSite(1, atom_ind, atom_1, atom_2, 0, weight_1, weight_2,
                       zero(T), zero(T), zero(T), weight_cross)
end

"""

"""
function ThreeParticleAverageSite(atom_ind, atom_1, atom_2, atom_3, weight_1::T, weight_2::T,
                                  weight_3::T, weight_cross=(zero(T) * u"nm^-1")) where T
    return VirtualSite(2, atom_ind, atom_1, atom_2, atom_3, weight_1, weight_2,
                       weight_3, zero(T), zero(T), weight_cross)
end

"""

"""
function OutOfPlaneSite(atom_ind, atom_1, atom_2, atom_3, weight_12::T,
                        weight_13::T, weight_cross) where T
    return VirtualSite(3, atom_ind, atom_1, atom_2, atom_3, zero(T), zero(T),
                       zero(T), weight_12, weight_13, weight_cross)
end

function calc_virtual_site_flags(virtual_sites, atom_masses, AT=Array)
    n_atoms = length(atom_masses)
    virtual_site_flags = falses(n_atoms)
    virtual_sites_cpu = from_device(virtual_sites)
    for (vi, vs) in enumerate(virtual_sites_cpu)
        i = vs.atom_ind
        if !(vs.type in (1, 2, 3))
            error("unrecognised virtual site type $(vs.type), should be 1/2/3")
        end
        if i > n_atoms
            error("virtual site $vi defines atom number $i but there are only " *
                  "$n_atoms atoms present")
        end
        if virtual_site_flags[i]
            error("virtual site $vi defines atom number $i but a previous virtual " *
                  "site already defined this atom")
        end
        virtual_site_flags[i] = true
    end
    for (vi, vs) in enumerate(virtual_sites_cpu)
        if (vs.atom_1 > 0 && virtual_site_flags[vs.atom_1]) ||
                (vs.atom_2 > 0 && virtual_site_flags[vs.atom_2]) ||
                (vs.atom_3 > 0 && virtual_site_flags[vs.atom_3])
            error("virtual site $vi is defined in terms of an atom that " *
                  "is itself a virtual site")
        end
    end
    warn_vs, warn_nvs = false, false
    for (vsf, atom_mass) in zip(virtual_site_flags, from_device(atom_masses))
        if vsf && !iszero(atom_mass)
            warn_vs = true
        elseif !vsf && iszero(atom_mass)
            warn_nvs = true
        end
    end
    if warn_vs
        @warn "One or more virtual sites has a non-zero mass, this may lead to problems"
    end
    if warn_nvs
        @warn "One or more atoms not marked as a virtual site has zero mass, " *
              "this may lead to problems"
    end
    return to_device(virtual_site_flags, AT)
end

"""

Assumes each virtual site is only defined once.
"""
function place_virtual_sites!(sys, virtual_sites=sys.virtual_sites)
    if length(virtual_sites) > 0
        backend = get_backend(sys.coords)
        n_threads_dev = 32
        kernel! = place_virtual_sites_kernel!(backend, n_threads_dev)
        kernel!(sys.coords, sys.boundary, virtual_sites; ndrange=length(virtual_sites))
    end
    return sys
end

"""

"""
@kernel function place_virtual_sites_kernel!(coords, boundary, @Const(virtual_sites))
    i = @index(Global, Linear)
    if i <= length(virtual_sites)
        vs = virtual_sites[i]
        if vs.type == 1
            vs_coord = vs.weight_1 * coords[vs.atom_1] + vs.weight_2 * coords[vs.atom_2]
        elseif vs.type == 2
            vs_coord = vs.weight_1 * coords[vs.atom_1] + vs.weight_2 * coords[vs.atom_2] +
                       vs.weight_3 * coords[vs.atom_3]
        elseif vs.type == 3
            r12 = vector(coords[vs.atom_1], coords[vs.atom_2], boundary)
            r13 = vector(coords[vs.atom_1], coords[vs.atom_3], boundary)
            cross_r12_r13 = cross(r12, r13) # Units L^2
            vs_coord = coords[vs.atom_1] + vs.weight_12 * r12 + vs.weight_13 * r13 +
                       vs.weight_cross * cross_r12_r13
        end
        coords[vs.atom_ind] = wrap_coords(vs_coord, boundary)
    end
end

"""

Assumes each virtual site is only defined once.
"""
function distribute_forces!(fs, sys::System{D, <:Any, T}, buffers,
                            virtual_sites=sys.virtual_sites) where {D, T}
    if length(virtual_sites) > 0
        buffers.fs_mat .= reshape(reinterpret(T, ustrip_vec.(fs)), D, length(sys))
        backend = get_backend(sys.coords)
        n_threads_dev = 32
        kernel! = distribute_forces_kernel!(backend, n_threads_dev)
        kernel!(buffers.fs_mat, sys.coords, sys.boundary, virtual_sites;
                ndrange=length(virtual_sites))
        fs_mat_flat = reshape(buffers.fs_mat, length(sys) * D)
        fs .= reinterpret(SVector{D, T}, fs_mat_flat) .* sys.force_units
    end
    return fs
end

@kernel function distribute_forces_kernel!(fs_mat::AbstractMatrix{T}, @Const(coords),
                        boundary::AbstractBoundary{D}, @Const(virtual_sites)) where {T, D}
    i = @index(Global, Linear)
    if i <= length(virtual_sites)
        vs = virtual_sites[i]
        if vs.type == 1
            for dim in 1:D
                f = fs_mat[dim, vs.atom_ind]
                Atomix.@atomic fs_mat[dim, vs.atom_1] += vs.weight_1 * f
                Atomix.@atomic fs_mat[dim, vs.atom_2] += vs.weight_2 * f
            end
        elseif vs.type == 2
            for dim in 1:D
                f = fs_mat[dim, vs.atom_ind]
                Atomix.@atomic fs_mat[dim, vs.atom_1] += vs.weight_1 * f
                Atomix.@atomic fs_mat[dim, vs.atom_2] += vs.weight_2 * f
                Atomix.@atomic fs_mat[dim, vs.atom_3] += vs.weight_3 * f
            end
        elseif vs.type == 3
            r12 = vector(coords[vs.atom_1], coords[vs.atom_2], boundary)
            r13 = vector(coords[vs.atom_1], coords[vs.atom_3], boundary)
            f = SVector(fs_mat[1, vs.atom_ind], fs_mat[2, vs.atom_ind], fs_mat[3, vs.atom_ind]) *
                                                        unit(eltype(r12))
            f2 = SVector(
                vs.weight_12 * f[1] - vs.weight_cross * r13[3] * f[2] + vs.weight_cross * r13[2] * f[3],
                vs.weight_cross * r13[3] * f[1] + vs.weight_12 * f[2] - vs.weight_cross * r13[1] * f[3],
                -vs.weight_cross * r13[2] * f[1] + vs.weight_cross * r13[1] * f[2] + vs.weight_12 * f[3],
            )
            f3 = SVector(
                vs.weight_13 * f[1] + vs.weight_cross * r12[3] * f[2] - vs.weight_cross * r12[2] * f[3],
                -vs.weight_cross * r12[3] * f[1] + vs.weight_13 * f[2] + vs.weight_cross * r12[1] * f[3],
                vs.weight_cross * r12[2] * f[1] - vs.weight_cross * r12[1] * f[2] + vs.weight_13 * f[3],
            )
            f1 = f - f2 - f3
            for dim in 1:D
                Atomix.@atomic fs_mat[dim, vs.atom_1] += ustrip(f1[dim])
                Atomix.@atomic fs_mat[dim, vs.atom_2] += ustrip(f2[dim])
                Atomix.@atomic fs_mat[dim, vs.atom_3] += ustrip(f3[dim])
            end
        end
        # Now the virtual site force has been distributed onto the other atoms,
        #   it can be set to zero
        for dim in 1:D
            fs_mat[dim, vs.atom_ind] = zero(T)
        end
    end
end

# Apply F = ma but give virtual sites zero acceleration
calc_accels(f, m, vsf) = (vsf ? zero(f / oneunit(m)) : f / m)

function pick_non_virtual_site(rng, sys)
    if iszero(length(sys.virtual_sites))
        return rand(rng, eachindex(sys))
    else
        flags = from_device(sys.virtual_site_flags)
        found = false
        i = 0
        while !found
            i = rand(rng, eachindex(sys))
            if !flags[i]
                found = true
            end
        end
        return i
    end
end
