# Virtual sites

export
    OneParticleSite,
    TwoParticleAverageSite,
    ThreeParticleAverageSite,
    OutOfPlaneSite,
    LocalCoordinatesSite,
    place_virtual_sites!

@public VirtualSite

"""
A virtual site, with coordinates determined by the coordinates of other atoms.

Created with [`OneParticleSite`](@ref), [`TwoParticleAverageSite`](@ref),
[`ThreeParticleAverageSite`](@ref), [`OutOfPlaneSite`](@ref) or [`LocalCoordinatesSite`](@ref)
and given to a [`System`](@ref) as `virtual_sites`.
"""
struct VirtualSite{T, IC, P}
    type::Int # 1/2/3/4/5 for OneParticleSite/TwoParticleAverageSite/ThreeParticleAverageSite/OutOfPlaneSite/LocalCoordinatesSite
    atom_ind::Int
    atom_1::Int
    atom_2::Int # 0 in OneParticleSite case
    atom_3::Int # 0 in OneParticleSite/TwoParticleAverageSite case
    weight_1::T # Weights are zero when not relevant
    weight_2::T
    weight_3::T
    weight_12::T
    weight_13::T
    weight_cross::IC # Units are 1/L
    local_weights::SVector{9, T} # Origin, x and y weights in the LocalCoordinatesSite case
    local_position::SVector{3, P} # Units are L
end

struct VirtualSiteTemplate{T, IC, P}
    type::Int
    name::String
    atom_name_1::String
    atom_name_2::String
    atom_name_3::String
    weight_1::T
    weight_2::T
    weight_3::T
    weight_12::T
    weight_13::T
    weight_cross::IC
    local_weights::SVector{9, T}
    local_position::SVector{3, P}
end

# Only a LocalCoordinatesSite uses the local coordinates fields, so the other site types are
#   constructed without them, with the length type taken from weight_cross
function VirtualSite(type, atom_ind, atom_1, atom_2, atom_3, weight_1, weight_2::T, weight_3,
                     weight_12, weight_13, weight_cross) where T
    return VirtualSite(type, atom_ind, atom_1, atom_2, atom_3, weight_1, weight_2, weight_3,
                       weight_12, weight_13, weight_cross, zero(SVector{9, T}),
                       zero(SVector{3, typeof(inv(oneunit(weight_cross)))}))
end

function VirtualSiteTemplate(type, atom_name, atom_1, atom_2, atom_3, weight_1, weight_2::T,
                             weight_3, weight_12, weight_13, weight_cross) where T
    return VirtualSiteTemplate(type, atom_name, atom_1, atom_2, atom_3, weight_1, weight_2,
                               weight_3, weight_12, weight_13, weight_cross, zero(SVector{9, T}),
                               zero(SVector{3, typeof(inv(oneunit(weight_cross)))}))
end

@doc raw"""
    OneParticleSite(atom_ind, atom_1)

A virtual site defined to have the same coordinates as another atom.

Returns a [`Molly.VirtualSite`](@ref) defined by:
```math
\mathbf{r} = \mathbf{r}_1
```
This can be useful in alchemical simulations when multiple versions of an atom are required.

Not compatible with gradient calculation using Enzyme.
"""
function OneParticleSite(atom_ind::Integer, atom_1::Integer, weight_cross=0.0u"nm^-1")
    # Optional weight_cross allows arrays of different virtual site types to be
    #   concretely typed with and without units
    T = typeof(ustrip(weight_cross))
    return VirtualSite(1, atom_ind, atom_1, 0, 0, zero(T), zero(T),
                       zero(T), zero(T), zero(T), weight_cross)
end

@doc raw"""
    TwoParticleAverageSite(atom_ind, atom_1, atom_2, weight_1, weight_2)

A virtual site defined by the weighted average of the coordinates of two atoms.

Returns a [`Molly.VirtualSite`](@ref) defined by:
```math
\mathbf{r} = w_1 \mathbf{r}_1 + w_2 \mathbf{r}_2
```
where ``w_1 + w_2`` must equal 1.

Not compatible with gradient calculation using Enzyme.
"""
function TwoParticleAverageSite(atom_ind::Integer, atom_1::Integer, atom_2::Integer, weight_1::T,
                                weight_2::T, weight_cross=(zero(T) * u"nm^-1")) where T
    if !isapprox(weight_1 + weight_2, 1)
        throw(ArgumentError("weight_1 + weight_2 must equal 1 for a TwoParticleAverageSite, " *
                            "found $(weight_1 + weight_2)"))
    end
    return VirtualSite(2, atom_ind, atom_1, atom_2, 0, weight_1, weight_2,
                       zero(T), zero(T), zero(T), weight_cross)
end

@doc raw"""
    ThreeParticleAverageSite(atom_ind, atom_1, atom_2, atom_3, weight_1, weight_2, weight_3)

A virtual site defined by the weighted average of the coordinates of three atoms.

Returns a [`Molly.VirtualSite`](@ref) defined by:
```math
\mathbf{r} = w_1 \mathbf{r}_1 + w_2 \mathbf{r}_2 + w_3 \mathbf{r}_3
```
where ``w_1 + w_2 + w_3`` must equal 1.

Not compatible with gradient calculation using Enzyme.
"""
function ThreeParticleAverageSite(atom_ind::Integer, atom_1::Integer, atom_2::Integer,
                                  atom_3::Integer, weight_1::T, weight_2::T, weight_3::T,
                                  weight_cross=(zero(T) * u"nm^-1")) where T
    if !isapprox(weight_1 + weight_2 + weight_3, 1)
        throw(ArgumentError("weight_1 + weight_2 + weight_3 must equal 1 for a " *
                            "ThreeParticleAverageSite, found $(weight_1 + weight_2 + weight_3)"))
    end
    return VirtualSite(3, atom_ind, atom_1, atom_2, atom_3, weight_1, weight_2,
                       weight_3, zero(T), zero(T), weight_cross)
end

@doc raw"""
    OutOfPlaneSite(atom_ind, atom_1, atom_2, atom_3, weight_12, weight_13, weight_cross)

A virtual site defined by the weighted average of the coordinates of three atoms
and the cross product of their relative displacements.

Returns a [`Molly.VirtualSite`](@ref) defined by:
```math
\mathbf{r} = \mathbf{r}_1 + w_{12} \mathbf{r}_{12} + w_{13} \mathbf{r}_{13} + w_{\mathrm{cross}} (\mathbf{r}_{12} \times \mathbf{r}_{13})
```

Only compatible with 3D systems.
Not compatible with virial calculation.
Not compatible with gradient calculation using Enzyme.
"""
function OutOfPlaneSite(atom_ind::Integer, atom_1::Integer, atom_2::Integer, atom_3::Integer,
                        weight_12::T, weight_13::T, weight_cross) where T
    return VirtualSite(4, atom_ind, atom_1, atom_2, atom_3, zero(T), zero(T),
                       zero(T), weight_12, weight_13, weight_cross)
end

@doc raw"""
    LocalCoordinatesSite(atom_ind, atom_1, atom_2, atom_3, origin_weights, x_weights,
                         y_weights, local_position)

A virtual site defined by a position in a local coordinate system given by three atoms,
matching the `LocalCoordinatesSite` of OpenMM.

Returns a [`Molly.VirtualSite`](@ref) defined by:
```math
\begin{aligned}
\mathbf{o} &= \sum_i w_i^o \mathbf{r}_i &
\mathbf{x} &= \sum_i w_i^x \mathbf{r}_i &
\mathbf{y} &= \sum_i w_i^y \mathbf{r}_i \\
\hat{\mathbf{x}} &= \frac{\mathbf{x}}{|\mathbf{x}|} &
\hat{\mathbf{z}} &= \frac{\mathbf{x} \times \mathbf{y}}{|\mathbf{x} \times \mathbf{y}|} &
\hat{\mathbf{y}} &= \hat{\mathbf{z}} \times \hat{\mathbf{x}}
\end{aligned}
```
```math
\mathbf{r} = \mathbf{o} + p_1 \hat{\mathbf{x}} + p_2 \hat{\mathbf{y}} + p_3 \hat{\mathbf{z}}
```
where ``\mathbf{p}`` is `local_position`, ``\sum_i w_i^o`` must equal 1 and ``\sum_i w_i^x``
and ``\sum_i w_i^y`` must equal 0.

Since the axes follow the atoms rather than the box, this places sites at a fixed distance
and orientation, such as the lone pairs of a CHARMM force field.

Only compatible with 3D systems.
Not compatible with virial calculation.
Not compatible with gradient calculation using Enzyme.
"""
function LocalCoordinatesSite(atom_ind::Integer, atom_1::Integer, atom_2::Integer,
                              atom_3::Integer, origin_weights, x_weights, y_weights,
                              local_position)
    check_local_weights(origin_weights, x_weights, y_weights, ArgumentError)
    local_weights = SVector{9}(origin_weights..., x_weights..., y_weights...)
    p = SVector{3}(local_position...)
    T = eltype(local_weights)
    return VirtualSite(5, atom_ind, atom_1, atom_2, atom_3, zero(T), zero(T), zero(T), zero(T),
                       zero(T), zero(inv(oneunit(eltype(p)))), local_weights, p)
end

# The weights of a LocalCoordinatesSite have to give an origin and two directions, as in OpenMM
function check_local_weights(origin_weights, x_weights, y_weights, error_type)
    if !isapprox(sum(origin_weights), 1)
        throw(error_type("origin_weights must sum to 1 for a LocalCoordinatesSite, found " *
                         "$(sum(origin_weights))"))
    end
    for (name, weights) in (("x_weights", x_weights), ("y_weights", y_weights))
        if !isapprox(sum(weights), 0; atol=1e-6)
            throw(error_type("$name must sum to 0 for a LocalCoordinatesSite, found " *
                             "$(sum(weights))"))
        end
    end
    return nothing
end

# The axes of a LocalCoordinatesSite and the inverse norms used by the force distribution;
#   a zero direction gives zero axes, as in OpenMM, and the site sits at the origin
@inline function local_axes(xdir, ydir)
    zdir = cross(xdir, ydir)
    inv_norm_x, inv_norm_z = inv_norm_or_zero(xdir), inv_norm_or_zero(zdir)
    dx, dz = xdir * inv_norm_x, zdir * inv_norm_z
    return dx, cross(dz, dx), dz, inv_norm_x, inv_norm_z
end

@inline function inv_norm_or_zero(v)
    n = norm(v)
    return iszero(n) ? zero(inv(oneunit(n))) : inv(n)
end

function setup_virtual_sites(virtual_sites, atom_masses, constraints, AT, D,
                             strictness=default_strictness())
    n_atoms = length(atom_masses)
    virtual_site_flags = falses(n_atoms)
    virtual_sites_cpu = from_device(virtual_sites)

    for (vi, vs) in enumerate(virtual_sites_cpu)
        i = vs.atom_ind
        if !(vs.type in 1:5)
            error("unrecognised virtual site type $(vs.type), should be 1/2/3/4/5")
        end
        if D != 3 && vs.type in (4, 5)
            site_name = (vs.type == 4 ? "OutOfPlaneSite" : "LocalCoordinatesSite")
            error("$site_name is only compatible with 3D systems")
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
        if vsf && !iszero_value(atom_mass)
            warn_vs = true
        elseif !vsf && iszero_value(atom_mass)
            warn_nvs = true
        end
    end
    if warn_vs
        report_issue(
            "One or more virtual sites has a non-zero mass, this may lead to problems",
            strictness,
        )
    end
    if warn_nvs
        err_str = "One or more atoms not marked as a virtual site has zero mass, " *
                  "this may lead to problems"
        report_issue(err_str, strictness)
    end

    for i in constrained_atom_inds(constraints)
        if virtual_site_flags[i]
            error("atom $i is a virtual site but is also in a constraint")
        end
    end
    return to_device(virtual_site_flags, AT)
end

"""
    place_virtual_sites!(sys, virtual_sites=sys.virtual_sites; n_threads=Threads.nthreads())

Set the coordinates of virtual sites based on the coordinates of the atoms that define them.
"""
function place_virtual_sites!(sys, virtual_sites=sys.virtual_sites;
                              n_threads::Integer=Threads.nthreads())
    # Assumes that each virtual site is only defined once
    if length(virtual_sites) > 0
        place_virtual_sites_coords!(sys.coords, sys.boundary, virtual_sites, n_threads)
    end
    return sys
end

# The number of chunks that `n_items` virtual sites are split into on the CPU, one per thread
#   but with at least `vs_min_chunk` sites in each, since the work of a site is small
const vs_min_chunk = 256

vs_n_chunks(n_items, n_threads) = clamp(cld(n_items, vs_min_chunk), 1, max(n_threads, 1))

@inline vs_chunk_range(chunk_i, n_chunks, n_items) =
            (((chunk_i - 1) * n_items) ÷ n_chunks + 1):((chunk_i * n_items) ÷ n_chunks)

#=
The virtual sites of CPU systems are placed, and their forces distributed, in loops rather
than with kernels, since Enzyme can not differentiate kernels of the CPU backend of
KernelAbstractions 0.10 (PoCL), which it would have to compile even for a system without
virtual sites
=#
function place_virtual_sites_coords!(coords, boundary, virtual_sites, n_threads)
    for i in eachindex(virtual_sites)
        place_virtual_site!(coords, boundary, virtual_sites[i])
    end
    return coords
end

# Each virtual site only writes its own coordinates and only reads those of atoms that are
#   not virtual sites, so the sites can be placed in parallel in any order
function place_virtual_sites_coords!(coords, boundary,
                                     virtual_sites::AbstractVector{<:VirtualSite}, n_threads)
    n_vs = length(virtual_sites)
    n_chunks = vs_n_chunks(n_vs, n_threads)
    @maybe_threads (n_chunks > 1) for chunk_i in 1:n_chunks
        for i in vs_chunk_range(chunk_i, n_chunks, n_vs)
            place_virtual_site!(coords, boundary, virtual_sites[i])
        end
    end
    return coords
end

function place_virtual_sites_coords!(coords::AbstractGPUArray, boundary,
                                     virtual_sites::AbstractGPUVector{<:VirtualSite},
                                     n_threads)
    backend = get_backend(coords)
    kernel! = place_virtual_sites_kernel!(backend, 128)
    launch_kernel!(kernel!, coords, boundary, virtual_sites; ndrange=length(virtual_sites))
    return coords
end

@kernel inbounds=true function place_virtual_sites_kernel!(coords, boundary,
                                                           @Const(virtual_sites))
    i = @index(Global, Linear)
    if i <= length(virtual_sites)
        place_virtual_site!(coords, boundary, virtual_sites[i])
    end
end

@inline function place_virtual_site!(coords, boundary, vs)
    if vs.type == 1
        vs_coord = coords[vs.atom_1]
    elseif vs.type == 2
        # w1 r1 + w2 r2 can't be used here since r1 and r2 may be in different periodic images
        # Only one absolute coordinate, r1, can be used, with others defined relatively
        # This is why w1 + w2 must equal 1, which is assumed here and checked on creation
        r12 = vector(coords[vs.atom_1], coords[vs.atom_2], boundary)
        vs_coord = coords[vs.atom_1] + vs.weight_2 * r12
    elseif vs.type == 3
        r12 = vector(coords[vs.atom_1], coords[vs.atom_2], boundary)
        r13 = vector(coords[vs.atom_1], coords[vs.atom_3], boundary)
        vs_coord = coords[vs.atom_1] + vs.weight_2 * r12 + vs.weight_3 * r13
    elseif vs.type == 4
        # Assumes 3D
        r12 = vector(coords[vs.atom_1], coords[vs.atom_2], boundary)
        r13 = vector(coords[vs.atom_1], coords[vs.atom_3], boundary)
        cross_r12_r13 = cross(r12, r13) # Units L^2
        vs_coord = coords[vs.atom_1] + vs.weight_12 * r12 + vs.weight_13 * r13 +
                   vs.weight_cross * cross_r12_r13
    else # vs.type == 5
        # Assumes 3D
        # The origin weights sum to 1 and the direction weights to 0, so the sums over the
        #   atoms can use r1 and relative vectors and stay in one periodic image
        r12 = vector(coords[vs.atom_1], coords[vs.atom_2], boundary)
        r13 = vector(coords[vs.atom_1], coords[vs.atom_3], boundary)
        w, p = vs.local_weights, vs.local_position
        dx, dy, dz = local_axes(w[5] * r12 + w[6] * r13, w[8] * r12 + w[9] * r13)
        vs_coord = coords[vs.atom_1] + w[2] * r12 + w[3] * r13 +
                   p[1] * dx + p[2] * dy + p[3] * dz
    end
    coords[vs.atom_ind] = wrap_coords(vs_coord, boundary)
    return nothing
end

# The number of atoms that define a virtual site
@inline n_virtual_site_atoms(vs) = (vs.type == 1 ? 1 : (vs.type == 2 ? 2 : 3))

#=
The order in which the forces on the virtual sites of a CPU system are distributed by
threads. The sites are split into groups in which no two sites are defined by the same
atom, by greedy coloring, so the sites of a group can add forces to the atoms defining
them in parallel without races. The groups are found once by `init_buffers!` and stored in
the buffers. For systems such as TIP4P water, where no atom defines more than one site,
there is one group.
`order` holds the indices of the virtual sites group by group, with group `g` from
`group_starts[g]` to `group_starts[g + 1] - 1`, or is empty if the coloring needed more
than 64 groups, in which case the sites are distributed serially. `virtual_sites` is the
array of virtual sites the groups were found for.
=#
struct VirtualSiteGroups{V}
    virtual_sites::V
    order::Vector{Int}
    group_starts::Vector{Int}
end

virtual_site_groups(virtual_sites, n_atoms) = nothing

function virtual_site_groups(virtual_sites::AbstractVector{<:VirtualSite}, n_atoms)
    n_vs = length(virtual_sites)
    used_groups = zeros(UInt64, n_atoms) # Bit g is set if a site of group g uses the atom
    site_groups = zeros(Int, n_vs)
    n_groups = 0
    for (vi, vs) in enumerate(virtual_sites)
        atom_inds = (vs.atom_1, vs.atom_2, vs.atom_3)
        n_atoms_vs = n_virtual_site_atoms(vs)
        taken = zero(UInt64)
        for k in 1:n_atoms_vs
            taken |= used_groups[atom_inds[k]]
        end
        if taken == typemax(UInt64)
            return VirtualSiteGroups(virtual_sites, Int[], Int[])
        end
        g = trailing_ones(taken) + 1
        for k in 1:n_atoms_vs
            used_groups[atom_inds[k]] |= (one(UInt64) << (g - 1))
        end
        site_groups[vi] = g
        n_groups = max(n_groups, g)
    end
    group_starts = zeros(Int, n_groups + 1)
    group_starts[1] = 1
    for g in site_groups
        group_starts[g + 1] += 1
    end
    cumsum!(group_starts, group_starts)
    next_pos = group_starts[1:n_groups]
    order = zeros(Int, n_vs)
    for (vi, g) in enumerate(site_groups)
        order[next_pos[g]] = vi
        next_pos[g] += 1
    end
    return VirtualSiteGroups(virtual_sites, order, group_starts)
end

function distribute_forces!(fs, sys, buffers, virtual_sites=sys.virtual_sites;
                            n_threads::Integer=Threads.nthreads())
    # Assumes that each virtual site is only defined once
    if length(virtual_sites) > 0
        distribute_virtual_site_forces!(fs, sys.coords, sys.boundary, virtual_sites,
                                        buffers, n_threads)
    end
    return fs
end

function distribute_virtual_site_forces!(fs, coords, boundary, virtual_sites, buffers,
                                         n_threads)
    for i in eachindex(virtual_sites)
        distribute_virtual_site_force!(fs, coords, boundary, virtual_sites[i])
    end
    return fs
end

function distribute_virtual_site_forces!(fs, coords, boundary,
                                         virtual_sites::AbstractVector{<:VirtualSite},
                                         buffers, n_threads)
    groups = (buffers isa BuffersCPU ? buffers.virtual_site_groups : nothing)
    if n_threads == 1 || !(groups isa VirtualSiteGroups) ||
                groups.virtual_sites !== virtual_sites || isempty(groups.order) ||
                length(groups.order) != length(virtual_sites)
        for i in eachindex(virtual_sites)
            distribute_virtual_site_force!(fs, coords, boundary, virtual_sites[i])
        end
    else
        order = groups.order
        for g in 1:(length(groups.group_starts) - 1)
            offset = groups.group_starts[g] - 1
            n_vs_group = groups.group_starts[g + 1] - groups.group_starts[g]
            n_chunks = vs_n_chunks(n_vs_group, n_threads)
            @maybe_threads (n_chunks > 1) for chunk_i in 1:n_chunks
                for oi in vs_chunk_range(chunk_i, n_chunks, n_vs_group)
                    vs = virtual_sites[order[offset + oi]]
                    distribute_virtual_site_force!(fs, coords, boundary, vs)
                end
            end
        end
    end
    return fs
end

# Move the force on a virtual site onto the atoms that define it and set it to zero
@inline function distribute_virtual_site_force!(fs, coords, boundary, vs)
    f = fs[vs.atom_ind]
    f1, f2, f3 = virtual_site_atom_forces(vs, f, coords, boundary)
    n_atoms_vs = n_virtual_site_atoms(vs)
    fs[vs.atom_1] += f1
    if n_atoms_vs >= 2
        fs[vs.atom_2] += f2
    end
    if n_atoms_vs == 3
        fs[vs.atom_3] += f3
    end
    fs[vs.atom_ind] = zero(f)
    return nothing
end

#=
On GPUs the forces are distributed in place with atomics, since sites defined by the same
atom are distributed at the same time. `fs` is reinterpreted as numbers in the force units,
since atomics do not work on unitful static vectors.
=#
function distribute_virtual_site_forces!(fs::AbstractGPUArray{SVector{D, F}}, coords,
                                boundary, virtual_sites::AbstractGPUVector{<:VirtualSite},
                                buffers, n_threads) where {D, F}
    backend = get_backend(fs)
    T = typeof(ustrip(zero(F)))
    fs_flat = reinterpret(T, fs)
    kernel! = distribute_forces_kernel!(backend, 128)
    launch_kernel!(kernel!, fs_flat, coords, boundary, virtual_sites, Val(unit(F)), Val(D);
                   ndrange=length(virtual_sites))
    return fs
end

@kernel inbounds=true function distribute_forces_kernel!(fs_flat, @Const(coords), boundary,
                                @Const(virtual_sites), ::Val{FU}, ::Val{D}) where {FU, D}
    i = @index(Global, Linear)
    if i <= length(virtual_sites)
        vs = virtual_sites[i]
        si = D * (vs.atom_ind - 1)
        f = SVector{D}(ntuple(dim -> fs_flat[si + dim], Val(D))) * FU
        f1, f2, f3 = virtual_site_atom_forces(vs, f, coords, boundary)
        n_atoms_vs = n_virtual_site_atoms(vs)
        atomic_add_force!(fs_flat, vs.atom_1, f1, Val(FU), Val(D))
        if n_atoms_vs >= 2
            atomic_add_force!(fs_flat, vs.atom_2, f2, Val(FU), Val(D))
        end
        if n_atoms_vs == 3
            atomic_add_force!(fs_flat, vs.atom_3, f3, Val(FU), Val(D))
        end
        for dim in 1:D
            fs_flat[si + dim] = zero(eltype(fs_flat))
        end
    end
end

@inline function atomic_add_force!(fs_flat, atom_i, f, ::Val{FU}, ::Val{D}) where {FU, D}
    ai = D * (atom_i - 1)
    for dim in 1:D
        Atomix.@atomic :monotonic fs_flat[ai + dim] += ustrip(FU, f[dim])
    end
    return nothing
end

#=
The forces on the atoms defining virtual site `vs` from the force `f` on the site, with
the third or second and third zero for sites defined by fewer atoms. The lengths are
stripped to the unit of the coordinates so that the forces are in the units of `f`.
=#
@inline function virtual_site_atom_forces(vs, f, coords, boundary)
    if vs.type == 1
        return f, zero(f), zero(f)
    elseif vs.type == 2
        return vs.weight_1 * f, vs.weight_2 * f, zero(f)
    elseif vs.type == 3
        return vs.weight_1 * f, vs.weight_2 * f, vs.weight_3 * f
    elseif vs.type == 4
        # Assumes 3D, the derivative of r1 + w12 r12 + w13 r13 + wc (r12 × r13)
        r12 = vector(coords[vs.atom_1], coords[vs.atom_2], boundary)
        r13 = vector(coords[vs.atom_1], coords[vs.atom_3], boundary)
        lu = unit(eltype(r12))
        r12_s, r13_s = ustrip.(lu, r12), ustrip.(lu, r13)
        wc = ustrip(inv(lu), vs.weight_cross)
        f2 = vs.weight_12 * f + wc * cross(r13_s, f)
        f3 = vs.weight_13 * f - wc * cross(r12_s, f)
        return f - f2 - f3, f2, f3
    else # vs.type == 5
        # Assumes 3D
        r12 = vector(coords[vs.atom_1], coords[vs.atom_2], boundary)
        r13 = vector(coords[vs.atom_1], coords[vs.atom_3], boundary)
        lu = unit(eltype(r12))
        w = vs.local_weights
        xdir = ustrip.(lu, w[5] * r12 + w[6] * r13)
        ydir = ustrip.(lu, w[8] * r12 + w[9] * r13)
        p = ustrip.(lu, vs.local_position)
        dx, dy, dz, inv_norm_x, inv_norm_z = local_axes(xdir, ydir)
        f1 = local_coords_force(f, p, w[1], w[4], w[7], xdir, ydir, dx, dy, dz,
                                inv_norm_x, inv_norm_z)
        f2 = local_coords_force(f, p, w[2], w[5], w[8], xdir, ydir, dx, dy, dz,
                                inv_norm_x, inv_norm_z)
        f3 = local_coords_force(f, p, w[3], w[6], w[9], xdir, ydir, dx, dy, dz,
                                inv_norm_x, inv_norm_z)
        return f1, f2, f3
    end
end

# The force on one of the three atoms of a LocalCoordinatesSite, the closed form derivative of
#   OpenMM (ReferenceVirtualSites.cpp), term by term; all lengths are unit-stripped
@inline function local_coords_force(f, p, wo, wx, wy, xdir, ydir, dx, dy, dz, inv_norm_x,
                                    inv_norm_z)
    wxs = wx * inv_norm_x
    t = (wx * ydir - wy * xdir) * inv_norm_z
    s = cross(dz, t)
    fp1, fp2, fp3 = p * f[1], p * f[2], p * f[3]
    f1 = SVector(
        fp1[1]*wxs*(1-dx[1]*dx[1]) + fp1[3]*(dz[1]*s[1]       ) + fp1[2]*((-dx[1]*dy[1]        )*wxs + dy[1]*s[1] - dx[2]*t[2] - dx[3]*t[3]),
        fp1[1]*wxs*( -dx[1]*dx[2]) + fp1[3]*(dz[1]*s[2] + t[3]) + fp1[2]*((-dx[2]*dy[1] - dz[3])*wxs + dy[1]*s[2] + dx[2]*t[1]),
        fp1[1]*wxs*( -dx[1]*dx[3]) + fp1[3]*(dz[1]*s[3] - t[2]) + fp1[2]*((-dx[3]*dy[1] + dz[2])*wxs + dy[1]*s[3] + dx[3]*t[1]),
    )
    f2 = SVector(
        fp2[1]*wxs*( -dx[2]*dx[1]) + fp2[3]*(dz[2]*s[1] - t[3]) - fp2[2]*(( dx[1]*dy[2] - dz[3])*wxs - dy[2]*s[1] - dx[1]*t[2]),
        fp2[1]*wxs*(1-dx[2]*dx[2]) + fp2[3]*(dz[2]*s[2]       ) - fp2[2]*(( dx[2]*dy[2]        )*wxs - dy[2]*s[2] + dx[1]*t[1] + dx[3]*t[3]),
        fp2[1]*wxs*( -dx[2]*dx[3]) + fp2[3]*(dz[2]*s[3] + t[1]) - fp2[2]*(( dx[3]*dy[2] + dz[1])*wxs - dy[2]*s[3] - dx[3]*t[2]),
    )
    f3 = SVector(
        fp3[1]*wxs*( -dx[3]*dx[1]) + fp3[3]*(dz[3]*s[1] + t[2]) + fp3[2]*((-dx[1]*dy[3] - dz[2])*wxs + dy[3]*s[1] + dx[1]*t[3]),
        fp3[1]*wxs*( -dx[3]*dx[2]) + fp3[3]*(dz[3]*s[2] - t[1]) + fp3[2]*((-dx[2]*dy[3] + dz[1])*wxs + dy[3]*s[2] + dx[2]*t[3]),
        fp3[1]*wxs*(1-dx[3]*dx[3]) + fp3[3]*(dz[3]*s[3]       ) + fp3[2]*((-dx[3]*dy[3]        )*wxs + dy[3]*s[3] - dx[1]*t[1] - dx[2]*t[2]),
    )
    return f1 + f2 + f3 + wo * f
end

function pick_non_virtual_site(sys, rng=Random.default_rng())
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

zero_vs_velocity(v, vsf) = (vsf ? zero(v) : v)
