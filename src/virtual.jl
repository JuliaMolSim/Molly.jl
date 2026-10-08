# Virtual sites

export
    OneParticleSite,
    TwoParticleAverageSite,
    ThreeParticleAverageSite,
    OutOfPlaneSite,
    LocalCoordinatesSite,
    place_virtual_sites!

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

Returns a `VirtualSite` defined by:
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

Returns a `VirtualSite` defined by:
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

Returns a `VirtualSite` defined by:
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

Returns a `VirtualSite` defined by:
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

Returns a `VirtualSite` defined by:
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
    n_vs = length(virtual_sites)
    if n_vs > 0
        backend = get_backend(sys.coords)
        n_threads_dev = 256
        kernel! = backend_kernel(place_virtual_sites_kernel!, backend, n_threads_dev)
        kernel!(sys.coords, sys.boundary, virtual_sites; ndrange=n_vs,
                workgroupsize=backend_workgroupsize(backend, n_vs, n_threads))
    end
    return sys
end

@kernel function place_virtual_sites_kernel!(coords, boundary, @Const(virtual_sites))
    i = @index(Global, Linear)
    if i <= length(virtual_sites)
        vs = virtual_sites[i]
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
    end
end

function distribute_forces!(fs, sys::System{D, <:Any, T}, buffers,
                            virtual_sites=sys.virtual_sites;
                            n_threads::Integer=Threads.nthreads()) where {D, T}
    # Assumes that each virtual site is only defined once
    n_vs = length(virtual_sites)
    if n_vs > 0
        copy_forces_to_matrix!(buffers.fs_mat, fs, Val(D))
        backend = get_backend(sys.coords)
        n_threads_dev = 128
        kernel! = backend_kernel(distribute_forces_kernel!, backend, n_threads_dev)
        kernel!(buffers.fs_mat, sys.coords, sys.boundary, virtual_sites; ndrange=n_vs,
                workgroupsize=backend_workgroupsize(backend, n_vs, n_threads))
        copy_matrix_to_forces!(fs, buffers.fs_mat, sys.force_units, Val(D), Val(T))
    end
    return fs
end

function copy_matrix_to_forces!(fs, fs_mat, force_units, ::Val{D}, ::Val{T}) where {D, T}
    fs_mat_flat = reshape(fs_mat, length(fs) * D)
    fs .= reinterpret(SVector{D, T}, fs_mat_flat) .* force_units
    return fs
end

function copy_matrix_to_forces!(fs::AbstractGPUArray, fs_mat, force_units, D::Val, T::Val)
    return apply_force_units_gpu!(fs, fs_mat, force_units, D, T)
end

function copy_forces_to_matrix!(fs_mat::AbstractMatrix{T}, fs, ::Val{D}) where {T, D}
    @inbounds for atom_i in eachindex(fs)
        f = ustrip_vec(fs[atom_i])
        for dim in 1:D
            fs_mat[dim, atom_i] = f[dim]
        end
    end
    return fs_mat
end

function copy_forces_to_matrix!(fs_mat::AbstractGPUArray{T, 2}, fs::AbstractGPUArray,
                                ::Val{D}) where {T, D}
    backend = get_backend(fs)
    n_threads_gpu = gpu_threads_copy(length(fs))
    kernel! = copy_forces_to_matrix_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, fs, Val(D); ndrange=length(fs))
    return fs_mat
end

@kernel inbounds=true function copy_forces_to_matrix_kernel!(fs_mat, @Const(fs), ::Val{D}) where D
    atom_i = @index(Global, Linear)
    if atom_i <= length(fs)
        f = ustrip_vec(fs[atom_i])
        for dim in 1:D
            fs_mat[dim, atom_i] = f[dim]
        end
    end
end

@kernel function distribute_forces_kernel!(fs_mat::AbstractMatrix{T}, @Const(coords),
                        boundary::AbstractBoundary{D}, @Const(virtual_sites)) where {T, D}
    i = @index(Global, Linear)
    if i <= length(virtual_sites)
        vs = virtual_sites[i]
        if vs.type == 1
            for dim in 1:D
                f = fs_mat[dim, vs.atom_ind]
                Atomix.@atomic fs_mat[dim, vs.atom_1] += f
            end
        elseif vs.type == 2
            for dim in 1:D
                f = fs_mat[dim, vs.atom_ind]
                Atomix.@atomic fs_mat[dim, vs.atom_1] += vs.weight_1 * f
                Atomix.@atomic fs_mat[dim, vs.atom_2] += vs.weight_2 * f
            end
        elseif vs.type == 3
            for dim in 1:D
                f = fs_mat[dim, vs.atom_ind]
                Atomix.@atomic fs_mat[dim, vs.atom_1] += vs.weight_1 * f
                Atomix.@atomic fs_mat[dim, vs.atom_2] += vs.weight_2 * f
                Atomix.@atomic fs_mat[dim, vs.atom_3] += vs.weight_3 * f
            end
        elseif vs.type == 4
            # Assumes 3D
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
        elseif vs.type == 5
            # Assumes 3D, the lengths are stripped to the unit of the coordinates so that the
            #   derivatives below are in force units
            r12 = vector(coords[vs.atom_1], coords[vs.atom_2], boundary)
            r13 = vector(coords[vs.atom_1], coords[vs.atom_3], boundary)
            lu = unit(eltype(r12))
            w = vs.local_weights
            xdir = ustrip.(lu, w[5] * r12 + w[6] * r13)
            ydir = ustrip.(lu, w[8] * r12 + w[9] * r13)
            p = ustrip.(lu, vs.local_position)
            dx, dy, dz, inv_norm_x, inv_norm_z = local_axes(xdir, ydir)
            f = SVector(fs_mat[1, vs.atom_ind], fs_mat[2, vs.atom_ind], fs_mat[3, vs.atom_ind])
            for j in 1:3
                atom_j = (j == 1 ? vs.atom_1 : (j == 2 ? vs.atom_2 : vs.atom_3))
                f_j = local_coords_force(f, p, w[j], w[3 + j], w[6 + j], xdir, ydir, dx, dy, dz,
                                         inv_norm_x, inv_norm_z)
                for dim in 1:D
                    Atomix.@atomic fs_mat[dim, atom_j] += f_j[dim]
                end
            end
        end
        # Now the virtual site force has been distributed onto the other atoms,
        #   it can be set to zero
        for dim in 1:D
            fs_mat[dim, vs.atom_ind] = zero(T)
        end
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
