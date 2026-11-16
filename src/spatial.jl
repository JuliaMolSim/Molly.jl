# Spatial calculations

export
    CubicBoundary,
    RectangularBoundary,
    TriclinicBoundary,
    volume,
    density,
    scale_boundary,
    random_coord,
    vector,
    wrap_coords,
    random_velocity,
    maxwell_boltzmann,
    random_velocities,
    random_velocities!,
    bond_angle,
    torsion_angle,
    momentum,
    remove_CM_motion!,
    pressure,
    scalar_pressure,
    scale_coords!,
    dipole_moment

@public AbstractBoundary,
    length_type,
    has_infinite_boundary,
    box_sides,
    box_center,
    vector_1D,
    wrap_coord_1D,
    unwrap_molecules

"""
Base type for boundary types.

Parameters are dimension, float type and coordinate type.
The built-in boundary types are [`CubicBoundary`](@ref), [`TriclinicBoundary`](@ref)
and [`RectangularBoundary`](@ref).
"""
abstract type AbstractBoundary{D, T, C} end

"""
    CubicBoundary(x, y, z)
    CubicBoundary(x)

Cubic 3D bounding box defined by three side lengths.

If one length is given then all three sides will have that length.
Setting one or more values to `Inf` gives no boundary in that dimension.
"""
struct CubicBoundary{D, T, C} <: AbstractBoundary{D, T, C}
    side_lengths::SVector{3, C}

    function CubicBoundary(side_lengths::SVector{3, C2}; check_positive::Bool=true) where C2
        if check_positive && any(l -> l <= zero(l), side_lengths)
            throw(DomainError("side lengths must be positive, got $side_lengths"))
        end
        T2 = typeof(ustrip(side_lengths[1]))
        new{3, T2, C2}(side_lengths)
    end
end

CubicBoundary(x, y, z; kwargs...) = CubicBoundary(SVector{3}(x, y, z); kwargs...)
CubicBoundary(x::Number; kwargs...) = CubicBoundary(SVector{3}(x, x, x); kwargs...)
CubicBoundary(m::SMatrix{3,3}; kwargs...) = CubicBoundary(SVector{3}(m[1,1], m[2,2], m[3,3]); kwargs...)
CubicBoundary(m::Matrix; kwargs...) = CubicBoundary(SVector{3}(m[1,1], m[2,2], m[3,3]); kwargs...)

function box_matrix(b::CubicBoundary)
    return SMatrix{3, 3}(
        ustrip(b.side_lengths[1]), 0, 0,
        0, ustrip(b.side_lengths[2]), 0,
        0, 0, ustrip(b.side_lengths[3]),
    ) * unit(b.side_lengths[1])
end

Base.getindex(b::CubicBoundary, i::Integer) = b.side_lengths[i]
Base.firstindex(b::CubicBoundary) = 1
Base.lastindex(b::CubicBoundary) = 3

function Base.zero(b::CubicBoundary{3, <:Any, C}) where C
    return CubicBoundary(zero(SVector{3, C}); check_positive=false)
end

function Base.:+(b1::CubicBoundary, b2::CubicBoundary)
    return CubicBoundary(b1.side_lengths .+ b2.side_lengths; check_positive=false)
end

function Chemfiles.UnitCell(b::CubicBoundary)
    if unit(eltype(b.side_lengths)) == NoUnits
        # Float64 required for Chemfiles
        Chemfiles.UnitCell(Float64.(Array(b.side_lengths)) .* 10) # Assume nm
    else
        Chemfiles.UnitCell(Float64.(Array(ustrip.(u"Å", b.side_lengths))))
    end
end

"""
    RectangularBoundary(x, y)
    RectangularBoundary(x)

Rectangular 2D bounding box defined by two side lengths.

If one length is given then both sides will have that length.
Setting one or more values to `Inf` gives no boundary in that dimension.
"""
struct RectangularBoundary{D, T, C} <: AbstractBoundary{D, T, C}
    side_lengths::SVector{2, C}

    function RectangularBoundary(side_lengths::SVector{2, C2}; check_positive::Bool=true) where C2
        if check_positive && any(l -> l <= zero(l), side_lengths)
            throw(DomainError("side lengths must be positive, got $side_lengths"))
        end
        T2 = typeof(ustrip(side_lengths[1]))
        new{2, T2, C2}(side_lengths)
    end
end

RectangularBoundary(x, y; kwargs...) = RectangularBoundary(SVector{2}(x, y); kwargs...)
RectangularBoundary(x::Number; kwargs...) = RectangularBoundary(SVector{2}(x, x); kwargs...)
RectangularBoundary(m::SMatrix{3,3}; kwargs...) = RectangularBoundary(SVector{2}(m[1,1], m[2,2]); kwargs...)

function box_matrix(b::RectangularBoundary)
    return SMatrix{3,3}(
        b.side_lengths[1], 0, 0,
        0, b.side_lengths[2], 0,
        0, 0, one(eltype(b.side_lengths)),
    )
end

Base.getindex(b::RectangularBoundary, i::Integer) = b.side_lengths[i]
Base.firstindex(b::RectangularBoundary) = 1
Base.lastindex(b::RectangularBoundary) = 2

function Base.zero(b::RectangularBoundary{2, <:Any, C}) where C
    return RectangularBoundary(zero(SVector{2, C}); check_positive=false)
end

function Base.:+(b1::RectangularBoundary, b2::RectangularBoundary)
    return RectangularBoundary(b1.side_lengths .+ b2.side_lengths; check_positive=false)
end

"""
    TriclinicBoundary(v1, v2, v3; approx_images=true)
    TriclinicBoundary(SVector(v1, v2, v3); approx_images=true)
    TriclinicBoundary(SVector(l1, l2, l3), SVector(α, β, γ); approx_images=true)
    TriclinicBoundary(arr; approx_images=true)

Triclinic 3D bounding box defined by 3 `SVector{3}` basis vectors or basis vector
lengths and angles α/β/γ in radians.

The first basis vector must point along the x-axis and the second must lie in the
xy plane.
An approximation is used to find the closest periodic image when using the
minimum image convention.
The approximation is correct for distances shorter than half the shortest box
height/width.
Setting the keyword argument `approx_images` to `false` means the exact closest
image is found, which is slower.

Not compatible with infinite boundaries.
"""
struct TriclinicBoundary{D, T, C, A, I} <: AbstractBoundary{D, T, C}
    basis_vectors::SVector{3, SVector{3, C}}
    α::T
    β::T
    γ::T
    reciprocal_size::SVector{3, I}
    cot_bprojyz_cprojyz::T
    cprojxy_x_over_z::T
    cprojxy_y_over_z::T
    cot_a_b::T
end

ispositive(x) = x > zero(x)

function TriclinicBoundary(bv::Union{SVector{3,<:SVector{3}}, SMatrix{3,3}}; approx_images::Bool=true)
    # Normalize input, if box matrix is passed turn into SVector{3, SVector{3, T}}
    if bv isa SMatrix
        bv = SVector(bv[:,1], bv[:,2], bv[:,3])
    end

    NT  = typeof(ustrip(bv[1][1])) # Underlying floating point type

    if !ispositive(bv[1][1]) || !iszero_value(bv[1][2]) || !iszero_value(bv[1][3])
        throw(ArgumentError("first basis vector must be along the x-axis (no y or z component) " *
                            "and have a positive x component " *
                            "when constructing a TriclinicBoundary, got $(bv[1])"))
    end
    if !ispositive(bv[2][2]) || !iszero_value(bv[2][3])
        throw(ArgumentError("second basis vector must be in the xy plane (no z component) " *
                            "and have a positive y component " *
                            "when constructing a TriclinicBoundary, got $(bv[2])"))
    end
    if !ispositive(bv[3][3])
        throw(ArgumentError("third basis vector must have a positive z component " *
                            "when constructing a TriclinicBoundary, got $(bv[3])"))
    end
    reciprocal_size = SVector{3}(inv(bv[1][1]), inv(bv[2][2]), inv(bv[3][3]))

    # Angles (dimensionless, in radians)
    α = NT(ustrip(bond_angle(bv[2], bv[3])))
    β = NT(ustrip(bond_angle(bv[1], bv[3])))
    γ = NT(ustrip(bond_angle(bv[1], bv[2])))

    # Cotangent of the angle between b_yz and c_yz.
    # This is negative when c has a negative y component, and the sign is needed to wrap
    #   coordinates in that case. cross_yz is positive due to the checks above.
    by, bz = bv[2][2], bv[2][3]
    cy, cz = bv[3][2], bv[3][3]
    cross_yz = ustrip(by * cz - bz * cy)
    dot_yz = ustrip(by * cy + bz * cz)
    cot_bprojyz_cprojyz = NT(dot_yz / cross_yz)

    # Direct ratios used to project along c into the xy plane.
    cx, cy2, cz2 = bv[3][1], bv[3][2], bv[3][3]
    cprojxy_x_over_z = NT(ustrip(cx / abs(cz2)))
    cprojxy_y_over_z = NT(ustrip(cy2 / abs(cz2)))

    # Cotangent of the angle between a and b.
    bx, by2 = bv[2][1], bv[2][2]
    cot_a_b = NT(ustrip(bx / by2))

    return TriclinicBoundary{3, NT, eltype(eltype(bv)), approx_images, eltype(reciprocal_size)}(
        bv, α, β, γ, reciprocal_size, cot_bprojyz_cprojyz,
        cprojxy_x_over_z, cprojxy_y_over_z, cot_a_b,
    )
end

function TriclinicBoundary(bv_lengths, angles; kwargs...)
    if any(!ispositive, bv_lengths)
        throw(ArgumentError("basis vector lengths must be positive, got $bv_lengths"))
    end
    if !all(a -> 0 < ustrip(a) < π, angles)
        throw(ArgumentError("angles must be in (0, π), got $angles"))
    end

    NT  = typeof(ustrip(bv_lengths[1]))
    has_units = bv_lengths[1] isa Unitful.Quantity
    uL  = has_units ? unit(bv_lengths[1]) : one(NT)
    tol0 = sqrt(eps(NT))

    # Compute in Float64 for stability
    Tw = Float64
    L1, L2, L3 = Tw.(ustrip.(bv_lengths))
    α,  β,  γ  = Tw.(ustrip.(angles))
    sγ, cγ = sincos(γ)
    cα, cβ = cos(α), cos(β)

    v1x, v1y, v1z = L1, 0.0, 0.0
    v2x, v2y, v2z = L2*cγ, L2*sγ, 0.0
    v3x = L3*cβ
    v3y = L3*(cα - cβ*cγ)/sγ
    v3z = sqrt(max(0.0, L3^2 - v3x^2 - v3y^2))

    # Convert back to input length type, zero-out tiny components
    toL(x::Real) = ((abs(x) < tol0) ? zero(NT) : NT(x)) * uL
    v1 = SVector(toL(v1x), toL(v1y), toL(v1z))
    v2 = SVector(toL(v2x), toL(v2y), toL(v2z))
    v3 = SVector(toL(v3x), toL(v3y), toL(v3z))

    return TriclinicBoundary(SVector(v1, v2, v3); kwargs...)
end

TriclinicBoundary(v1, v2, v3; kwargs...) = TriclinicBoundary(SVector{3}(v1, v2, v3); kwargs...)
TriclinicBoundary(arr; kwargs...)        = TriclinicBoundary(SVector{3}(arr); kwargs...)

box_matrix(b::TriclinicBoundary) = SMatrix{3, 3}(hcat(b.basis_vectors...))

Base.getindex(b::TriclinicBoundary, i::Integer) = b.basis_vectors[i]
Base.firstindex(b::TriclinicBoundary) = 1
Base.lastindex(b::TriclinicBoundary) = 3

# Chemfiles.jl passes the matrix memory to the row-major C library, which takes the
#   columns there as the basis vectors, so the basis vectors are the rows here
function Chemfiles.UnitCell(b::TriclinicBoundary)
    bv_rows = permutedims(hcat(b.basis_vectors...))
    if unit(eltype(eltype(b.basis_vectors))) == NoUnits
        Chemfiles.UnitCell(Float64.(Array(bv_rows)) .* 10) # Assume nm
    else
        Chemfiles.UnitCell(Float64.(Array(ustrip.(u"Å", bv_rows))))
    end
end

function boundary_from_chemfiles(unit_cell, T=Float64, units=u"nm")
    shape = Chemfiles.shape(unit_cell)
    if shape == Chemfiles.Infinite
        return CubicBoundary(T(Inf) * units)
    elseif shape == Chemfiles.Orthorhombic
        side_lengths = SVector{3}(T.(Chemfiles.lengths(unit_cell) * u"Å"))
        if units == NoUnits
            return CubicBoundary(ustrip.(u"nm", side_lengths)) # Assume nm
        else
            return CubicBoundary(uconvert.(units, side_lengths))
        end
    elseif shape == Chemfiles.Triclinic
        side_lengths = SVector{3}(T.(Chemfiles.lengths(unit_cell) * u"Å"))
        angles = SVector{3}(deg2rad.(T.(Chemfiles.angles(unit_cell))))
        if units == NoUnits
            return TriclinicBoundary(ustrip.(u"nm", side_lengths), angles) # Assume nm
        else
            return TriclinicBoundary(uconvert.(units, side_lengths), angles)
        end
    else
        error("unrecognised Chemfiles cell shape $shape")
    end
end

Base.broadcastable(b::Union{CubicBoundary, RectangularBoundary}) = b.side_lengths

AtomsBase.n_dimensions(::AbstractBoundary{D}) where {D} = D
AtomsBase.n_dimensions(::Union{System{D}, ReplicaSystem{D}}) where {D} = D

float_type(::AbstractBoundary{<:Any, T}) where {T} = T

"""
    length_type(sys)
    length_type(boundary)

The length type a [`System`](@ref), [`ReplicaSystem`](@ref) or bounding box uses,
such as `typeof(1.0f0u"nm")`.
"""
length_type(::AbstractBoundary{<:Any, <:Any, C}) where {C} = C
length_type(sys::Union{System, ReplicaSystem}) = length_type(sys.boundary)

Unitful.ustrip(b::CubicBoundary) = CubicBoundary(ustrip.(b.side_lengths))
Unitful.ustrip(u::Unitful.Units, b::CubicBoundary) = CubicBoundary(ustrip.(u, b.side_lengths))
Unitful.ustrip(b::RectangularBoundary) = RectangularBoundary(ustrip.(b.side_lengths))
Unitful.ustrip(u::Unitful.Units, b::RectangularBoundary) = RectangularBoundary(ustrip.(u, b.side_lengths))

function Unitful.ustrip(b::TriclinicBoundary{<:Any, <:Any, <:Any, A}) where A
    return TriclinicBoundary(map(bv -> ustrip.(bv), b.basis_vectors); approx_images=A)
end

function Unitful.ustrip(u::Unitful.Units, b::TriclinicBoundary{<:Any, <:Any, <:Any, A}) where A
    return TriclinicBoundary(map(bv -> ustrip.(u, bv), b.basis_vectors); approx_images=A)
end

function AtomsBase.cell_vectors(b::CubicBoundary{3, <:Any, C}) where C
    z = zero(C)
    bb = (
        SVector(b[1], z   , z   ),
        SVector(z   , b[2], z   ),
        SVector(z   , z   , b[3]),
    )
    return (unit(C) == NoUnits ? (bb .* u"nm") : bb) # Assume nm without other information
end

function AtomsBase.cell_vectors(b::RectangularBoundary{2, <:Any, C}) where C
    z = zero(C)
    bb = (
        SVector(b[1], z   ),
        SVector(z   , b[2]),
    )
    return (unit(C) == NoUnits ? (bb .* u"nm") : bb)
end

function AtomsBase.cell_vectors(b::TriclinicBoundary{3, <:Any, C}) where C
    bb = (b.basis_vectors[1], b.basis_vectors[2], b.basis_vectors[3])
    return (unit(C) == NoUnits ? (bb .* u"nm") : bb)
end

function invert_box_vectors(boundary::CubicBoundary)
    sl = boundary.side_lengths
    z = zero(inv(sl[1]))
    recip_box = SVector(
        SVector(inv(sl[1]), z, z),
        SVector(z, inv(sl[2]), z),
        SVector(z, z, inv(sl[3])),
    )
    return recip_box
end

function invert_box_vectors(boundary::TriclinicBoundary)
    bv = boundary.basis_vectors
    z = zero(bv[1][1]^2)
    recip_box = SVector(
        SVector(bv[2][2]*bv[3][3], z, z),
        SVector(-bv[2][1]*bv[3][3], bv[1][1]*bv[3][3], z),
        SVector(bv[2][1]*bv[3][2] - bv[2][2]*bv[3][1], -bv[1][1]*bv[3][2], bv[1][1]*bv[2][2]),
    )
    return recip_box ./ volume(boundary)
end

"""
    has_infinite_boundary(sys)
    has_infinite_boundary(boundary)

Whether a boundary has one or more infinite (non-periodic) dimensions.
"""
has_infinite_boundary(b::Union{CubicBoundary, RectangularBoundary}) = any(isinf, b.side_lengths)
has_infinite_boundary(b::TriclinicBoundary) = false # Not currently supported
has_infinite_boundary(sys::System) = has_infinite_boundary(sys.boundary)

"""
    box_sides(boundary)
    box_sides(boundary, i)

The side lengths of a bounding box, or the side length in dimension `i`.

For a [`TriclinicBoundary`](@ref) the diagonal components of the basis vectors are returned.
"""
@inline box_sides(b::Union{CubicBoundary, RectangularBoundary}) = b.side_lengths
@inline box_sides(b::Union{CubicBoundary, RectangularBoundary}, i) = b.side_lengths[i]
@inline box_sides(b::TriclinicBoundary) = SVector(b[1][1], b[2][2], b[3][3])
@inline box_sides(b::TriclinicBoundary, i) = b[i][i]

"""
    volume(sys)
    volume(boundary)

Calculate the volume (3D) or area (2D) of a [`System`](@ref) or bounding box.

Returns infinite volume for infinite boundaries.
"""
volume(sys) = volume(sys.boundary)
volume(b::AbstractBoundary) = prod(box_sides(b))

"""
    density(sys)

The density of a [`System`](@ref).

Returns zero density for infinite boundaries.
"""
function density(sys)
    m = sys.total_mass
    if dimension(m) == u"𝐌 * 𝐍^-1"
        m_no_mol = m / Unitful.Na
    else
        m_no_mol = m
    end
    d = m_no_mol / volume(sys)
    if unit(d) == NoUnits
        return d
    else
        return uconvert(u"kg * m^-3", d)
    end
end

"""
    box_center(boundary)

Calculate the center of a bounding box.

Dimensions with infinite length return zero.
"""
function box_center(b::Union{CubicBoundary, RectangularBoundary})
    return map(x -> isinf(x) ? zero(x) : x / 2, b.side_lengths)
end

box_center(b::TriclinicBoundary) = sum(b.basis_vectors) / 2

"""
    scale_boundary(boundary, scale_factor)

Scale the sides of a bounding box by a scaling factor.

The scaling factor can be a single number or a `SVector` of the appropriate number
of dimensions corresponding to the scaling factor for each axis.
For a 3D bounding box the volume scales as the cube of the scaling factor.
"""
scale_boundary(b::CubicBoundary, scale) = CubicBoundary(b.side_lengths .* scale)
scale_boundary(b::RectangularBoundary, scale) = RectangularBoundary(b.side_lengths .* scale)

function scale_boundary(b::TriclinicBoundary{<:Any, <:Any, <:Any, A}, scale) where A
    return TriclinicBoundary(b[1] .* scale, b[2] .* scale, b[3] .* scale; approx_images=A)
end

# The upper and lower corners of the minimum cubic box surrounding the bounding box, used
#   for visualization
# The lower corner is below the origin for a triclinic box with negative off-diagonal
#   components, which has coordinates below zero
cubic_bounding_box(b::Union{CubicBoundary, RectangularBoundary}) = b.side_lengths
cubic_bounding_box(b::TriclinicBoundary) = max.(triclinic_box_corners(b)...)

cubic_bounding_box_min(b::Union{CubicBoundary, RectangularBoundary}) = zero(b.side_lengths)
cubic_bounding_box_min(b::TriclinicBoundary) = min.(triclinic_box_corners(b)...)

function triclinic_box_corners(b::TriclinicBoundary)
    v1, v2, v3 = b.basis_vectors
    return (zero(v1), v1, v2, v3, v1 + v2, v1 + v3, v2 + v3, v1 + v2 + v3)
end

# Coordinates for visualizing bounding box
function bounding_box_lines(boundary::CubicBoundary, dist_unit)
    sl = ustrip.(dist_unit, boundary.side_lengths)
    z = zero(sl[1])
    p1 = SVector(z    , z    , z    )
    p2 = SVector(sl[1], z    , z    )
    p3 = SVector(z    , sl[2], z    )
    p4 = SVector(z    , z    , sl[3])
    p5 = SVector(sl[1], sl[2], z    )
    p6 = SVector(sl[1], z    , sl[3])
    p7 = SVector(z    , sl[2], sl[3])
    p8 = SVector(sl[1], sl[2], sl[3])
    seq = [p1, p4, p7, p3, p1, p2, p5, p3, p7, p8, p6, p4, p6, p2, p5, p8]
    return getindex.(seq, 1), getindex.(seq, 2), getindex.(seq, 3)
end

function bounding_box_lines(boundary::RectangularBoundary, dist_unit)
    sl = ustrip.(dist_unit, boundary.side_lengths)
    xs = [0.0, 0.0, sl[1], sl[1], 0.0]
    ys = [0.0, sl[2], sl[2], 0.0, 0.0]
    return xs, ys
end

function bounding_box_lines(boundary::TriclinicBoundary, dist_unit)
    bv = ustrip_vec.(dist_unit, boundary.basis_vectors)
    p1 = zero(bv[1])
    p2 = bv[1]
    p3 = bv[2]
    p4 = bv[3]
    p5 = bv[1] + bv[2]
    p6 = bv[1] + bv[3]
    p7 = bv[2] + bv[3]
    p8 = bv[1] + bv[2] + bv[3]
    seq = [p1, p4, p7, p3, p1, p2, p5, p3, p7, p8, p6, p4, p6, p2, p5, p8]
    return getindex.(seq, 1), getindex.(seq, 2), getindex.(seq, 3)
end

"""
    random_coord(boundary; rng=Random.default_rng())

Generate a random coordinate uniformly distributed within a bounding box.
"""
function random_coord(boundary::CubicBoundary{3, T}; rng=Random.default_rng()) where T
    return rand(rng, SVector{3, T}) .* boundary
end

function random_coord(boundary::RectangularBoundary{2, T}; rng=Random.default_rng()) where T
    return rand(rng, SVector{2, T}) .* boundary
end

function random_coord(boundary::TriclinicBoundary{3, T}; rng=Random.default_rng()) where T
    return sum(rand(rng, SVector{3, T}) .* boundary.basis_vectors)
end

"""
    vector_1D(c1, c2, side_length)

Displacement between two 1D coordinate values from c1 to c2, accounting for
periodic boundary conditions in a [`CubicBoundary`](@ref) or [`RectangularBoundary`](@ref).

The minimum image convention is used, so the displacement is to the closest
version of the coordinate accounting for the periodic boundaries.
"""
function vector_1D(c1, c2, side_length)
    v12 = c2 - c1
    half_sl = side_length / 2
    return ifelse(
        v12 >  half_sl, v12 - side_length,
        ifelse(v12 < -half_sl, v12 + side_length, v12),
    )
end

"""
    vector(c1, c2, boundary)

Displacement between two coordinate values from c1 to c2, accounting for
periodic boundary conditions.

The minimum image convention is used, so the displacement is to the closest
version of the coordinates accounting for the periodic boundaries.
For the [`TriclinicBoundary`](@ref) an approximation is used to find the closest
version by default.
"""
function vector(c1, c2, boundary::CubicBoundary)
    return @inbounds SVector(
        vector_1D(c1[1], c2[1], boundary.side_lengths[1]),
        vector_1D(c1[2], c2[2], boundary.side_lengths[2]),
        vector_1D(c1[3], c2[3], boundary.side_lengths[3]),
    )
end

function vector(c1, c2, boundary::RectangularBoundary)
    return @inbounds SVector(
        vector_1D(c1[1], c2[1], boundary.side_lengths[1]),
        vector_1D(c1[2], c2[2], boundary.side_lengths[2]),
    )
end

# The components of the basis vectors that are always zero are skipped
function vector(c1, c2, boundary::TriclinicBoundary{3, T, <:Any, true}) where T
    a, b, c = boundary.basis_vectors
    rs = boundary.reciprocal_size
    dx, dy, dz = c2 - c1
    n3 = floor(dz * rs[3] + T(0.5))
    dx, dy, dz = dx - c[1] * n3, dy - c[2] * n3, dz - c[3] * n3
    n2 = floor(dy * rs[2] + T(0.5))
    dx, dy = dx - b[1] * n2, dy - b[2] * n2
    n1 = floor(dx * rs[1] + T(0.5))
    dx -= a[1] * n1
    return SVector(dx, dy, dz)
end

function vector(c1, c2, boundary::TriclinicBoundary{3, T, <:Any, false}) where T
    bv = boundary.basis_vectors
    offsets = (-1, 0, 1)
    min_sqdist = typemax(c1[1] ^ 2)
    min_dr = zero(c1)
    for ox in offsets, oy in offsets, oz in offsets
        c2_offset = c2 + ox * bv[1] + oy * bv[2] + oz * bv[3]
        dr = c2_offset - c1
        sqdist = dot(dr, dr)
        if sqdist < min_sqdist
            min_dr = dr
            min_sqdist = sqdist
        end
    end
    return min_dr
end

# Pad a vector to 3D to allow operations such as the cross product
function vector_pad3D(c1::SVector{2, T}, c2::SVector{2, T}, boundary::RectangularBoundary{2}) where T
    SVector{3, T}(
        vector_1D(c1[1], c2[1], boundary[1]),
        vector_1D(c1[2], c2[2], boundary[2]),
        zero(T),
    )
end

vector_pad3D(c1::SVector{3}, c2::SVector{3}, boundary) = vector(c1, c2, boundary)

# Trim a vector back to 2D if required
trim3D(v::SVector{3, T}, boundary::RectangularBoundary{2}) where T = SVector{2, T}(v[1], v[2])
trim3D(v::SVector{3}, boundary) = v

"""
    wrap_coord_1D(c, side_length)

Ensure a 1D coordinate is within the bounding box and return the coordinate.
"""
function wrap_coord_1D(c, side_length)
    if isinf(side_length)
        return c
    else
        return c - floor(c / side_length) * side_length
    end
end

"""
    wrap_coords(c, boundary)

Ensure a coordinate is within the bounding box and return the coordinate.
"""
wrap_coords(v, boundary::Union{CubicBoundary, RectangularBoundary}) = wrap_coord_1D.(v, boundary)

function wrap_coords(v, boundary::TriclinicBoundary)
    bv, rs = boundary.basis_vectors, boundary.reciprocal_size
    v_wrap = v
    # Bound in z-axis
    v_wrap -= bv[3] * floor(v_wrap[3] * rs[3])
    # Bound in y-axis
    v_wrap -= bv[2] * floor((v_wrap[2] - v_wrap[3] * boundary.cot_bprojyz_cprojyz) * rs[2])
    dx = v_wrap[3] * boundary.cprojxy_x_over_z
    dy = v_wrap[3] * boundary.cprojxy_y_over_z
    # Bound in x-axis
    v_wrap -= bv[1] * floor(
        (v_wrap[1] - dx - (v_wrap[2] - dy) * boundary.cot_a_b) * rs[1],
    )
    return v_wrap
end

"""
    unwrap_molecules(sys)
    unwrap_molecules(coords, boundary, topology)

Unwrap coordinates so that molecules are not split over the boundary.

Returns coordinates on CPU.
Ensures that the center of geometry of bonded clusters of atoms is
always inside the simulation box.
"""
unwrap_molecules(sys) = unwrap_molecules(sys.coords, sys.boundary, sys.topology)

function unwrap_molecules(coords::AbstractVector{<:SVector{D}}, boundary, topology) where D
    coords_cpu = from_device(coords)
    if isnothing(topology)
        return coords_cpu
    end

    # Fractional to Cartesian conversion
    if hasproperty(boundary, :basis_vectors)
        if D != 3
            error("Triclinic boundary only defined for 3-dimensions")
        end
        Bm = reduce(hcat, boundary.basis_vectors)
        B  = SMatrix{3, 3}(Bm)
        to_frac = (r::SVector{3}) -> ustrip.(B) \ ustrip.(unit(eltype(B)), r) # Dimensionless
        to_cart = (f::SVector{3}) -> B * f # Length units
    else
        sl = boundary.side_lengths
        to_frac = (r::SVector{D}) -> r ./ sl # Dimensionless
        to_cart = (f::SVector{D}) -> f .* sl # Length units
    end
    wrap01(v) = v .- floor.(v .+ eps(eltype(v))) # Keep in [0,1)

    # Wrapped fractional coords
    N  = length(coords_cpu)
    f1 = to_frac(coords_cpu[1])
    f  = Vector{typeof(f1)}(undef, N) # Dimensionless
    @inbounds for i in 1:N
        f[i] = wrap01(to_frac(coords_cpu[i]))
    end

    # Bond adjacency
    adj = [Int[] for _ in 1:N]
    @inbounds for (i32, j32) in topology.bonded_atoms
        i, j = Int(i32), Int(j32)
        push!(adj[i], j)
        push!(adj[j], i)
    end

    # Unwrap each bonded connected component.
    #
    # For each cluster:
    # 1. Anchor the seed atom in the primary box.
    # 2. Reconstruct bonded neighbours using nearest-image fractional offsets.
    # 3. Translate the entire cluster by a lattice vector so that COG falls inside box.
    u = similar(f) # Dimensionless
    visited = falses(N)
    @inbounds for seed in 1:N
        visited[seed] && continue
        u[seed] = f[seed]
        visited[seed] = true
        stack = Int[seed]
        cluster = Int[seed]
        while !isempty(stack)
            i = pop!(stack)
            fi, ui = f[i], u[i]
            for j in adj[i]
                visited[j] && continue
                df = f[j] - fi
                df -= round.(df) # Shift to nearest image, component-wise
                u[j] = ui + df
                visited[j] = true
                push!(stack, j)
                push!(cluster, j)
            end
        end

        # Move the whole bonded cluster so its center of geometry is inside [0, 1).
        #
        # This preserves the unwrapped internal geometry because the same integer
        # lattice shift is applied to every atom in the cluster.
        cog = zero(u[seed])
        for i in cluster
            cog += u[i]
        end
        cog /= length(cluster)

        shift = floor.(cog)

        for i in cluster
            u[i] -= shift
        end
    end

    # Back to Cartesian with units
    out = Vector{typeof(coords_cpu[1])}(undef, N)
    @inbounds for i in 1:N
        out[i] = to_cart(u[i])
    end
    return out
end

function check_correction_arg(correction)
    if !(correction in (:pbc, :wrap))
        throw(ArgumentError("correction argument must be :pbc or :wrap, found $correction"))
    end
end

@inline function random_velocity_svector(::Val{D}, args...; rng=Random.default_rng()) where D
    return SVector(ntuple(_ -> maxwell_boltzmann(args...; rng=rng), Val(D)))
end

"""
    random_velocity(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature;
                    dims=3, rng=Random.default_rng())
    random_velocity(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature,
                    k::Union{BoltzmannConstUnits, MolarBoltzmannConstUnits};
                    dims=3, rng=Random.default_rng())
    random_velocity(atom_mass::Real, temp::Real,
                    k::Real=ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k);
                    dims=3, rng=Random.default_rng())

Generate a random velocity from the Maxwell-Boltzmann distribution, with
optional custom Boltzmann constant.
"""
function random_velocity(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature;
                         dims::Integer=3, rng=Random.default_rng())
    return random_velocity_svector(Val(dims), atom_mass, temp; rng=rng)
end

function random_velocity(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature,
                         k::Union{BoltzmannConstUnits, MolarBoltzmannConstUnits};
                         dims::Integer=3, rng=Random.default_rng())
    return random_velocity_svector(Val(dims), atom_mass, temp, k; rng=rng)
end

function random_velocity(atom_mass::Real, temp::Real,
                         k::Real=ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k);
                         dims::Integer=3, rng=Random.default_rng())
    return random_velocity_svector(Val(dims), atom_mass, temp, k; rng=rng)
end

"""
    maxwell_boltzmann(atom_mass::Unitful.Mass, temp::Unitful.Temperature,
                      k::BoltzmannConstUnits=Unitful.k; rng=Random.default_rng())
    maxwell_boltzmann(atom_mass::MolarMass, temp::Unitful.Temperature,
                      k_molar::MolarBoltzmannConstUnits=(Unitful.k * Unitful.Na);
                      rng=Random.default_rng())
    maxwell_boltzmann(atom_mass::Real, temperature::Real,
                      k::Real=ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k);
                      rng=Random.default_rng())

Generate a random velocity along one dimension from the Maxwell-Boltzmann
distribution, with optional custom Boltzmann constant.
"""
function maxwell_boltzmann(atom_mass::Unitful.Mass, temp::Unitful.Temperature,
                           k::BoltzmannConstUnits=uconvert(u"g * nm^2 * ps^-2 * K^-1", Unitful.k);
                           rng=Random.default_rng())
    T = typeof(convert(AbstractFloat, ustrip(temp)))
    σ = sqrt(T(k) * temp / atom_mass)
    return rand(rng, Normal(zero(T), T(ustrip(σ)))) * unit(σ)
end

function maxwell_boltzmann(atom_mass::MolarMass, temp::Unitful.Temperature,
                           k_molar::MolarBoltzmannConstUnits=uconvert(u"g * mol^-1 * nm^2 * ps^-2 * K^-1", Unitful.k * Unitful.Na);
                           rng=Random.default_rng())
    T = typeof(convert(AbstractFloat, ustrip(temp)))
    σ = sqrt(T(k_molar) * temp / atom_mass)
    return rand(rng, Normal(zero(T), T(ustrip(σ)))) * unit(σ)
end

function maxwell_boltzmann(atom_mass::Real, temp::Real,
                           k::Real=ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k);
                           rng=Random.default_rng())
    T = typeof(convert(AbstractFloat, temp))
    σ = sqrt(T(k) * temp / atom_mass)
    return rand(rng, Normal(zero(T), T(σ)))
end

"""
    random_velocities(sys, temp; rng=Random.default_rng(), remove_CM_motion=false)

Generate random velocities from the Maxwell-Boltzmann distribution
for a [`System`](@ref).

Virtual sites are given a velocity of zero.
Setting `remove_CM_motion=true` removes the center of mass motion from the
generated velocities with [`remove_CM_motion!`](@ref).
"""
function random_velocities(sys::System, temp; rng=Random.default_rng(),
                           remove_CM_motion::Bool=false)
    random_velocities!(similar(sys.velocities), sys, temp; rng=rng,
                       remove_CM_motion=remove_CM_motion)
end

"""
    random_velocities!(sys, temp; rng=Random.default_rng(), remove_CM_motion=false)
    random_velocities!(vels, sys, temp; rng=Random.default_rng(), remove_CM_motion=false)

Set the velocities of a [`System`](@ref), or a vector, to random velocities
generated from the Maxwell-Boltzmann distribution.

Virtual sites are given a velocity of zero.
Setting `remove_CM_motion=true` removes the center of mass motion from the
generated velocities with [`remove_CM_motion!`](@ref).
"""
function random_velocities!(sys, temp; rng=Random.default_rng(), remove_CM_motion::Bool=false)
    random_velocities!(sys.velocities, sys, temp; rng=rng, remove_CM_motion=remove_CM_motion)
    return sys
end

function random_velocities!(vels::AbstractVector{SVector{D, C}}, sys::System{<:Any, <:Any, T},
                            temp; rng=Random.default_rng(),
                            remove_CM_motion::Bool=false) where {D, C, T}
    ms = from_device(masses(sys))
    vsf = from_device(sys.virtual_site_flags)
    kT = sys.k * temp
    ctr1 = rand(rng, UInt64)
    key = rand(rng, UInt64)
    natoms = UInt64(length(ms))
    @inbounds @simd ivdep for i in eachindex(vels, vsf, ms)
        scale = ifelse(vsf[i], zero(C), C(Base.FastMath.sqrt_fast(kT/ms[i])))
        vels[i] = randn_svec(SVector{D, T}, i%UInt64, ctr1, key, natoms) * scale
    end
    if remove_CM_motion
        remove_CM_motion!(vels, sys)
    end
    return vels
end

function random_velocities!(vels::AbstractGPUArray, sys::System{<:Any, <:Any, T},
                            temp; rng=Random.default_rng(),
                            remove_CM_motion::Bool=false) where T
    AT = array_type(vels)
    ms = to_device(sys.masses, AT)
    vsf = to_device(sys.virtual_site_flags, AT)
    kT = sys.k * temp
    ctr1 = rand(rng, UInt64)
    key = rand(rng, UInt64)
    backend = get_backend(vels)
    kernel! = random_velocities_kernel!(backend)
    kernel!(vels, ms, kT, vsf, ctr1, key, Val(T); ndrange=length(vels))
    if remove_CM_motion
        remove_CM_motion!(vels, sys)
    end
    return vels
end

# Sometimes domain error occurs for acos if the value is > 1.0 or < -1.0
acos_bound(x::Real) = acos(clamp(x, -1, 1))

"""
    bond_angle(coord_i, coord_j, coord_k, boundary)
    bond_angle(vec_ji, vec_jk)

Calculate the bond or pseudo-bond angle in radians between three
coordinates or two vectors.

The angle between j→i and j→k is returned in the range 0 to π.
"""
function bond_angle(coords_i, coords_j, coords_k, boundary)
    vec_ji = vector(coords_j, coords_i, boundary)
    vec_jk = vector(coords_j, coords_k, boundary)
    return bond_angle(vec_ji, vec_jk)
end

function bond_angle(vec_ji, vec_jk)
    acos_bound(dot(vec_ji, vec_jk) / (norm(vec_ji) * norm(vec_jk)))
end

"""
    torsion_angle(coord_i, coord_j, coord_k, coord_l, boundary)
    torsion_angle(vec_ij, vec_jk, vec_kl)

Calculate the torsion angle in radians defined by four coordinates or
three vectors.

The angle between the planes defined by atoms (i, j, k) and (j, k, l) is
returned in the range -π to π.
"""
function torsion_angle(coords_i, coords_j, coords_k, coords_l, boundary)
    vec_ij = vector(coords_i, coords_j, boundary)
    vec_jk = vector(coords_j, coords_k, boundary)
    vec_kl = vector(coords_k, coords_l, boundary)
    return torsion_angle(vec_ij, vec_jk, vec_kl)
end

function torsion_angle(vec_ij, vec_jk, vec_kl)
    cross_ij_jk = vec_ij × vec_jk
    cross_jk_kl = vec_jk × vec_kl
    θ = atan(
        ustrip(dot(cross_ij_jk × cross_jk_kl, normalize(vec_jk))),
        ustrip(dot(cross_ij_jk, cross_jk_kl)),
    )
    return θ
end

function torsion_vectors(coords_i, coords_j, coords_k, coords_l, boundary)
    ab = vector(coords_i, coords_j, boundary)
    bc = vector(coords_j, coords_k, boundary)
    cd = vector(coords_k, coords_l, boundary)
    cross_ab_bc = ab × bc
    cross_bc_cd = bc × cd
    bc_norm = norm(bc)
    θ = atan(
        ustrip(dot(cross_ab_bc × cross_bc_cd, bc / bc_norm)),
        ustrip(dot(cross_ab_bc, cross_bc_cd)),
    )
    return ab, bc, cd, cross_ab_bc, cross_bc_cd, bc_norm, θ
end

"""
    momentum(system)

The total momentum of a [`System`](@ref).

This is the sum over atoms of the velocity multiplied by the mass, and is close
to zero when the center of mass motion is removed with [`remove_CM_motion!`](@ref).
"""
momentum(sys) = sum(sys.velocities .* masses(sys))

"""
    remove_CM_motion!(system)

Remove the center of mass motion from a [`System`](@ref).

This assumes a periodic system and only removes the translational component
of the motion.
Systems with infinite boundaries should also remove the rotation around the
center of mass, which is not currently done.

Simulators should in general call this function regularly, since the number
of degrees of freedom that is used to calculate temperature assumes that the
center of mass motion is removed.
"""
function remove_CM_motion!(sys)
    remove_CM_motion!(sys.velocities, sys)
    return sys
end

function remove_CM_motion!(vels::AbstractVector, sys)
    masses_cpu = masses(sys)
    cm_momentum = zero(eltype(vels)) .* zero(eltype(masses_cpu))
    for i in eachindex(vels)
        cm_momentum += vels[i] * masses_cpu[i]
    end
    cm_velocity = cm_momentum / sys.total_mass
    for i in eachindex(vels)
        if !sys.virtual_site_flags[i]
            vels[i] -= cm_velocity
        end
    end
    return vels
end

update_vel(v, cm_v, vsf) = (vsf ? zero(v) : v - cm_v)

# The CUDA extension has a fast 3D CUDA path for `remove_CM_motion!(sys)`
function remove_CM_motion!(vels::AbstractGPUArray, sys)
    cm_momentum = mapreduce((v, m) -> v .* m, +, vels, masses(sys))
    cm_velocity = cm_momentum / sys.total_mass
    if isempty(sys.virtual_sites)
        vels .-= (cm_velocity,)
    else
        vels .= update_vel.(vels, (cm_velocity,), sys.virtual_site_flags)
    end
    return vels
end

@doc raw"""
    pressure(system, neighbors=find_neighbors(system), step_n=0, buffers=nothing;
             recompute=true, n_threads=Threads.nthreads(),
             pairwise_inters=system.pairwise_inters,
             specific_inter_lists=system.specific_inter_lists,
             general_inters=system.general_inters, strictness=:warn)

Calculate the pressure tensor of the system.

The pressure is defined as
```math
\bf{P} = \frac{ 2 \cdot \bf{K} + \bf{W} }{V} \quad\textrm{i.e.}\quad
P_{\alpha\beta} = \frac{1}{V} \left( \sum_i m_i v_{i,\alpha} v_{i,\beta}
+ \sum_i r_{i,\alpha} f_{i,\beta} \right)
```
where ``V`` is the system volume (area in 2D), ``\bf{K}`` is the kinetic energy tensor
(see [`kinetic_energy`](@ref)) and ``\bf{W}`` is the [`virial`](@ref) tensor.
``\bf{K}`` is calculated from the velocities stored in the system, including any center
of mass motion, and does not depend on the number of degrees of freedom used by
[`temperature`](@ref).
For the leapfrog-like integrator these velocities are half a time step
behind the coordinates.
For systems with units the default pressure is in bar in 3D and in bar nm in 2D.

This is the [pressure tensor of LAMMPS](https://docs.lammps.org/compute_pressure.html),
and is the negative of the sum of the
[per-atom stress of LAMMPS](https://docs.lammps.org/compute_stress_atom.html) divided by
the volume, apart from tail corrections.
LAMMPS can subtract a velocity bias in the kinetic term via the temperature compute,
which Molly does not do.
It is also the pressure tensor of GROMACS,
``\bf{P} = \frac{2}{V} (\bf{E}_\textrm{kin} - \bf{\Xi})`` with
``\bf{\Xi} = -\bf{W} / 2``, though for the leapfrog integrator GROMACS uses the average
of the kinetic energy tensors from the two half steps around the coordinates.

For constrained systems, direct recomputation approximates constraint
contributions using a deterministic small-step constraint preview. To reuse a
total virial from a simulation step, pass valid simulator buffers with
`recompute=false`.

To calculate the scalar pressure, see [`scalar_pressure`](@ref).

Not compatible with infinite boundaries.
"""
function pressure(sys; n_threads::Integer=Threads.nthreads(), kwargs...)
    return pressure(sys, find_neighbors(sys; n_threads=n_threads), 0, nothing;
                    n_threads=n_threads, kwargs...)
end

# The pressure units of a system with units, bar in 3 dimensions and bar * nm, which has
#   dimensions of force per length, in 2 dimensions
pressure_units(sys::System{D}) where {D} = (sys.energy_units == NoUnits ? NoUnits :
                                            (D == 2 ? u"bar * nm" : u"bar"))

function pressure_from_tensors!(pres_tensor, sys::System{D}, kin_tensor, virial, vol) where D
    if sys.energy_units == NoUnits
        # If implied energy units are (u * nm^2 * ps^-2) and everything is
        #   consistent then this has implied units of (u * nm^-1 * ps^-2)
        #   for 3 dimensions and (u * ps^-2) for 2 dimensions
        @inbounds for col in 1:D
            for row in 1:D
                K = energy_remove_mol(kin_tensor[row, col]) # (1/2) Σ m v⊗v
                W = energy_remove_mol(virial[row, col]) # Σ r⊗f
                pres_tensor[row, col] = (2 * K + W) / vol
            end
        end
    else
        P_units = pressure_units(sys)
        @inbounds for col in 1:D
            for row in 1:D
                K = energy_remove_mol(kin_tensor[row, col])
                W = energy_remove_mol(virial[row, col])
                pres_tensor[row, col] = uconvert(P_units, (2 * K + W) / vol) # Σ r⊗f
            end
        end
    end
    return pres_tensor
end

function pressure(sys::System{D}, neighbors, step_n::Integer=0, buffers_in=nothing;
                  recompute::Bool=true, n_threads::Integer=Threads.nthreads(),
                  kin_tensor=nothing, kwargs...) where D
    if isnothing(buffers_in)
        buffers = init_buffers!(sys, n_threads)
    else
        buffers = buffers_in
    end
    if recompute && length(sys.constraints) > 0
        if isnothing(buffers_in) || !has_total_virial(buffers, step_n)
            compute_initial_total_virial!(buffers, sys, neighbors, step_n;
                                          n_threads=n_threads, kwargs...)
        end
    elseif recompute
        forces!(zero_forces(sys), sys, neighbors, step_n, buffers, Val(true);
                n_threads=n_threads, kwargs...)
    elseif length(sys.constraints) > 0 && !has_total_virial(buffers, step_n)
        error("cannot calculate pressure for constrained systems without a valid total virial")
    end

    if isnothing(kin_tensor)
        # Always evaluate K in case velocities were rescaled by a thermostat
        kinetic_energy_tensor!(buffers.kin_tensor, sys)
    else
        buffers.kin_tensor .= kin_tensor
    end
    if has_infinite_boundary(sys.boundary)
        error("pressure calculation not compatible with infinite boundaries")
    end

    pressure_from_tensors!(buffers.pres_tensor, sys, buffers.kin_tensor, buffers.virial,
                           volume(sys.boundary))
    mark_pressure!(buffers.validity, step_n)
    return buffers.pres_tensor
end

@doc raw"""
    scalar_pressure(system, neighbors=find_neighbors(system), step_n=0, buffers=nothing;
                    recompute=true, n_threads=Threads.nthreads(),
                    pairwise_inters=system.pairwise_inters,
                    specific_inter_lists=system.specific_inter_lists,
                    general_inters=system.general_inters, strictness=:warn)

Calculate the pressure of the system as a scalar.

This is the trace of the [`pressure`](@ref) tensor divided by the number of dimensions
``d``:
```math
P = \frac{\rm{Tr}\left[ \bf{P} \right]}{d}
  = \frac{1}{d V} \left( 2 K + \sum_i \bf{r_i} \cdot \bf{f_i} \right)
```
where ``K`` is the kinetic energy.
This matches the scalar pressure of GROMACS.
The scalar pressure of LAMMPS uses ``N_\textrm{df} k_B T / d`` for the kinetic term, which
is the same as ``2 K / d`` when the temperature includes all velocities.
For constrained systems, the same deterministic preview convention as
[`pressure`](@ref) applies.
"""
function scalar_pressure(sys; n_threads::Integer=Threads.nthreads(), kwargs...)
    return scalar_pressure(sys, find_neighbors(sys; n_threads=n_threads), 0, nothing;
                           n_threads=n_threads, kwargs...)
end

function scalar_pressure(sys::System{D}, neighbors, step_n::Integer=0, buffers=nothing;
                         recompute::Bool=true, n_threads::Integer=Threads.nthreads(),
                         kin_tensor=nothing, kwargs...) where D
    P = pressure(sys, neighbors, step_n, buffers; recompute=recompute,
                 n_threads=n_threads, kin_tensor=kin_tensor, kwargs...)
    return tr(P) / D
end

# Center of geometry per molecule using unwrapped fractional coordinates
function molecule_centers(coords::AbstractArray{SVector{D,C}}, boundary, topology) where {D, C}
    if isnothing(topology)
        return coords
    end

    wrap01(v) = v .- floor.(v)

    is_triclinic = hasproperty(boundary, :basis_vectors)
    if is_triclinic && D != 3
        error("triclinic boundary only defined for 3-dimensions")
    end

    # Build frac<->cart transforms
    if is_triclinic
        B = box_matrix(boundary)
        to_frac = (r::SVector{3}) -> B \ r
        to_cart = (f::SVector{3}) -> B * f
    else
        sl = boundary.side_lengths
        to_frac = (r::SVector{D}) -> r ./ sl
        to_cart = (f::SVector{D}) -> f .* sl
    end

    # Flatten coords
    x = vec(coords)
    N = length(x)

    # Fractional, wrapped
    y1 = wrap01(to_frac(x[1]))
    f  = Vector{typeof(y1)}(undef, N)
    f[1] = y1
    @inbounds for i in 2:N
        f[i] = wrap01(to_frac(x[i]))
    end

    atom_mol = topology.atom_molecule_inds
    n_mol    = length(topology.molecule_atom_counts)
    bonds    = topology.bonded_atoms

    # Build per-atom neighbor list (bonds)
    nbrs = [Int[] for _ in 1:N]
    @inbounds for b in bonds
        i, j = b[1], b[2]
        push!(nbrs[i], j)
        push!(nbrs[j], i)
    end

    # Group atoms by molecule
    atoms_by_mol = [Int[] for _ in 1:n_mol]
    @inbounds for i in 1:N
        push!(atoms_by_mol[atom_mol[i]], i)
    end

    # Unwrapped fractional coords per atom (filled per molecule)
    u = Vector{eltype(f)}(undef, N)

    centers = Vector{SVector{D, eltype(x[1][1])}}(undef, n_mol)
    visited = falses(N)
    stack = Vector{Int}(undef, N)
    for m in 1:n_mol
        atoms = atoms_by_mol[m]
        if isempty(atoms)
            centers[m] = to_cart(zero(f[1]))
            continue
        end

        # Search over each connected component within the molecule
        for seed in atoms
            visited[seed] && continue
            u[seed] = f[seed]
            visited[seed] = true
            stack_len = 1
            stack[stack_len] = seed
            while stack_len > 0
                i = stack[stack_len]
                stack_len -= 1
                @inbounds for j in nbrs[i]
                    # stay within molecule
                    if atom_mol[j] != m || visited[j]
                        continue
                    end
                    Δ = f[j] - f[i] - round.(f[j] - f[i])
                    u[j] = u[i] + Δ
                    visited[j] = true
                    stack_len += 1
                    stack[stack_len] = j
                end
            end
        end

        # Any isolated atoms that had no bonds
        @inbounds for i in atoms
            if !visited[i]
                u[i] = f[i]
                visited[i] = true
            end
        end

        # Mean in unwrapped fractional space
        s = zero(u[atoms[1]])
        @inbounds for i in atoms
            s += u[i]
        end
        ū = s / length(atoms)

        # Wrap back and convert to Cartesian
        centers[m] = to_cart(wrap01(ū))
    end

    return centers
end

#=
Moving molecules as units on the GPU

The center of a molecule is the center of geometry of the molecule made whole, which
`molecule_centers` finds on the CPU by walking the bonds. On the GPU the bonds are walked
in parallel. A spanning forest of the bond graph of each molecule is found once on the
CPU and kept with the topology, as the parent of each atom and the depth of the deepest
tree. Each atom then finds the whole number of box lengths between its wrapped fractional
coordinates and those of its parent, and pointer jumping sums these along the path to the
root of its tree in a number of rounds that grows with the logarithm of the depth, so a
protein of a thousand bonded atoms in a chain takes ten rounds. The unwrapped fractional
coordinates of an atom are then its wrapped coordinates less the sum, which gives the same
centers as the walk on the CPU since the sum along a path does not depend on the path
unless the molecule is bonded to its own image, where neither is meaningful.
=#
struct MoleculeUnwrapPlan{AI, AV, AS, AC}
    parents::AI # Parent of each atom in the spanning forest, 0 for a root
    atom_molecule_inds::AI
    molecule_atom_counts::AI
    n_jumps::Int # Rounds of pointer jumping that reach the root of every tree
    frac::AV # Wrapped fractional coordinates of each atom
    shifts_1::AS # Box lengths from each atom to its current ancestor, double buffered
    shifts_2::AS
    ancestors_1::AI
    ancestors_2::AI
    center_sums::AC # Sum of the unwrapped fractional coordinates of the atoms of a molecule
end

# The spanning forest of the bond graph within each molecule, as the parent of each atom
#   with 0 for the roots, and the number of rounds of pointer jumping needed to reach them
function molecule_spanning_forest(topology::MolecularTopology, n_atoms::Integer)
    atom_mol = topology.atom_molecule_inds
    bonds = topology.bonded_atoms
    # Adjacency in compressed sparse row form
    degree = zeros(Int32, n_atoms)
    for (i, j) in bonds
        degree[i] += one(Int32)
        degree[j] += one(Int32)
    end
    starts = Vector{Int32}(undef, n_atoms + 1)
    starts[1] = one(Int32)
    for i in 1:n_atoms
        starts[i + 1] = starts[i] + degree[i]
    end
    adjacent = Vector{Int32}(undef, starts[end] - 1)
    fill!(degree, zero(Int32))
    for (i, j) in bonds
        adjacent[starts[i] + degree[i]] = j
        degree[i] += one(Int32)
        adjacent[starts[j] + degree[j]] = i
        degree[j] += one(Int32)
    end

    # Breadth-first search from the lowest atom of each connected component, which gives
    #   the shallowest trees
    parents = zeros(Int32, n_atoms)
    depths = zeros(Int32, n_atoms)
    visited = falses(n_atoms)
    queue = Vector{Int32}(undef, n_atoms)
    max_depth = zero(Int32)
    for seed in 1:n_atoms
        visited[seed] && continue
        visited[seed] = true
        queue_head, queue_tail = 1, 1
        queue[1] = seed
        while queue_head <= queue_tail
            i = queue[queue_head]
            queue_head += 1
            for k in starts[i]:(starts[i + 1] - 1)
                j = adjacent[k]
                (visited[j] || atom_mol[j] != atom_mol[i]) && continue
                visited[j] = true
                parents[j] = i
                depths[j] = depths[i] + one(Int32)
                max_depth = max(max_depth, depths[j])
                queue_tail += 1
                queue[queue_tail] = j
            end
        end
    end
    # After k rounds the shift of each atom covers 2^k bonds towards the root
    n_jumps = (max_depth <= 1 ? 0 : ceil(Int, log2(max_depth)))
    return parents, n_jumps
end

#=
The plan for unwrapping the molecules of `coords` on the device, see `MoleculeUnwrapPlan`,
from the cache of `topology` or built and cached.
=#
function molecule_unwrap_plan(topology::MolecularTopology,
                              coords::AbstractGPUArray{SVector{D, C}}) where {D, C}
    T = typeof(ustrip(zero(C)))
    N = length(coords)
    cache = topology.unwrap_cache[]
    if cache isa MoleculeUnwrapPlan && eltype(cache.frac) == SVector{D, T} &&
            length(cache.frac) == N && array_type(cache.frac) == array_type(coords)
        return cache
    end
    if length(topology.atom_molecule_inds) != N
        throw(ArgumentError("the topology is for $(length(topology.atom_molecule_inds)) " *
                            "atoms but there are $N coordinates"))
    end
    AT = array_type(coords)
    TH = default_float_type_high(AT, T)
    backend = get_backend(coords)
    parents, n_jumps = molecule_spanning_forest(topology, N)
    n_mol = length(topology.molecule_atom_counts)
    plan = MoleculeUnwrapPlan(
        to_device(parents, AT),
        to_device(topology.atom_molecule_inds, AT),
        to_device(topology.molecule_atom_counts, AT),
        n_jumps,
        similar(coords, SVector{D, T}),
        similar(coords, SVector{D, Int32}),
        similar(coords, SVector{D, Int32}),
        similar(coords, Int32),
        similar(coords, Int32),
        KernelAbstractions.zeros(backend, TH, D, n_mol),
    )
    topology.unwrap_cache[] = plan
    return plan
end

@inline wrap_frac01(v) = v .- floor.(v)

# The box matrix without units, in the length unit of the coordinates
function box_matrix_nounits(boundary::Union{CubicBoundary, RectangularBoundary}, coord_unit,
                            ::Type{T}) where T
    sl = boundary.side_lengths
    D = length(sl)
    vals = ntuple(Val(D * D)) do k
        row, col = (k - 1) % D + 1, (k - 1) ÷ D + 1
        (row == col ? T(ustrip(coord_unit, sl[row])) : zero(T))
    end
    return SMatrix{D, D, T}(vals)
end

function box_matrix_nounits(boundary::TriclinicBoundary, coord_unit, ::Type{T}) where T
    return SMatrix{3, 3, T}(ustrip.(coord_unit, box_matrix(boundary)))
end

#=
Find the wrapped fractional coordinates, the shifts to the roots and the sums of the
unwrapped fractional coordinates of each molecule in `plan`, returning the array holding
the shifts.
=#
function unwrap_molecules_gpu!(plan::MoleculeUnwrapPlan, coords, inv_box)
    backend = get_backend(coords)
    N = length(coords)
    launch_kernel!(unwrap_init_kernel!(backend, 256), plan.frac, plan.shifts_1,
                   plan.ancestors_1, coords, plan.parents, inv_box; ndrange=N)
    shifts_in, shifts_out = plan.shifts_1, plan.shifts_2
    ancestors_in, ancestors_out = plan.ancestors_1, plan.ancestors_2
    for _ in 1:plan.n_jumps
        launch_kernel!(unwrap_jump_kernel!(backend, 256), shifts_out, ancestors_out,
                       shifts_in, ancestors_in; ndrange=N)
        shifts_in, shifts_out = shifts_out, shifts_in
        ancestors_in, ancestors_out = ancestors_out, ancestors_in
    end
    fill!(plan.center_sums, zero(eltype(plan.center_sums)))
    launch_kernel!(molecule_center_sum_kernel!(backend, 256), plan.center_sums, plan.frac,
                   shifts_in, plan.atom_molecule_inds; ndrange=N)
    return shifts_in
end

@kernel inbounds=true function unwrap_init_kernel!(frac, shifts, ancestors, @Const(coords),
                                                   @Const(parents), inv_box)
    i = @index(Global, Linear)
    if i <= length(coords)
        f = wrap_frac01(inv_box * ustrip_vec(coords[i]))
        frac[i] = f
        p = parents[i]
        if iszero(p)
            shifts[i] = zero(eltype(shifts))
            ancestors[i] = zero(eltype(ancestors))
        else
            f_parent = wrap_frac01(inv_box * ustrip_vec(coords[p]))
            shifts[i] = round.(Int32, f - f_parent)
            ancestors[i] = p
        end
    end
end

@kernel inbounds=true function unwrap_jump_kernel!(shifts_out, ancestors_out,
                                                   @Const(shifts_in), @Const(ancestors_in))
    i = @index(Global, Linear)
    if i <= length(ancestors_in)
        a = ancestors_in[i]
        if iszero(a)
            shifts_out[i] = shifts_in[i]
            ancestors_out[i] = a
        else
            shifts_out[i] = shifts_in[i] + shifts_in[a]
            ancestors_out[i] = ancestors_in[a]
        end
    end
end

@kernel inbounds=true function molecule_center_sum_kernel!(center_sums, @Const(frac),
                                                @Const(shifts), @Const(atom_molecule_inds))
    i = @index(Global, Linear)
    if i <= length(frac)
        u = frac[i] - shifts[i]
        m = atom_molecule_inds[i]
        for d in eachindex(u)
            Atomix.@atomic center_sums[d, m] += eltype(center_sums)(u[d])
        end
    end
end

# The mean of the unwrapped fractional coordinates of the atoms of molecule `m`
@inline function molecule_mean_frac(center_sums, counts, m, ::Val{D}, ::Val{T}) where {D, T}
    n = counts[m]
    return SVector{D, T}(ntuple(d -> T(center_sums[d, m] / n), Val(D)))
end

@kernel inbounds=true function molecule_centers_kernel!(centers, @Const(center_sums),
                                        @Const(counts), box, coord_unit, ::Val{D},
                                        ::Val{T}) where {D, T}
    m = @index(Global, Linear)
    if m <= length(counts)
        f = wrap_frac01(molecule_mean_frac(center_sums, counts, m, Val(D), Val(T)))
        centers[m] = (box * f) * coord_unit
    end
end

function molecule_centers(coords::AbstractGPUArray{SVector{D, C}}, boundary,
                          topology) where {D, C}
    if isnothing(topology)
        return coords
    end
    T = typeof(ustrip(zero(C)))
    coord_unit = unit(C)
    plan = molecule_unwrap_plan(topology, coords)
    box = box_matrix_nounits(boundary, coord_unit, T)
    unwrap_molecules_gpu!(plan, coords, inv(box))
    n_mol = length(plan.molecule_atom_counts)
    centers = similar(coords, n_mol)
    launch_kernel!(molecule_centers_kernel!(get_backend(coords), 256), centers,
                   plan.center_sums, plan.molecule_atom_counts, box, coord_unit, Val(D),
                   Val(T); ndrange=n_mol)
    return centers
end

#=
Place the atoms of the scaled system. The displacement of an atom from the center of its
molecule is found from the unwrapped fractional coordinates, so the molecule is kept whole,
and is rotated by `R` before being added to the scaled center.
=#
@kernel inbounds=true function scale_place_kernel!(coords, velocities, @Const(frac),
                                        @Const(shifts), @Const(center_sums), @Const(counts),
                                        @Const(atom_molecule_inds), box_old, μ, R, μinv,
                                        boundary_new, coord_unit, ::Val{D}, ::Val{T},
                                        ::Val{scale_vel}) where {D, T, scale_vel}
    i = @index(Global, Linear)
    if i <= length(coords)
        m = atom_molecule_inds[i]
        mean_frac = molecule_mean_frac(center_sums, counts, m, Val(D), Val(T))
        center_old = box_old * wrap_frac01(mean_frac)
        u = frac[i] - shifts[i]
        δ = box_old * (u - mean_frac)
        r_new = R * δ + μ * center_old
        coords[i] = wrap_coords(r_new * coord_unit, boundary_new)
        if scale_vel
            velocities[i] = μinv * velocities[i]
        end
    end
end

rebuild_boundary(b::CubicBoundary,       box) = CubicBoundary(box)
rebuild_boundary(b::RectangularBoundary, box) = RectangularBoundary(box)

function rebuild_boundary(b::TriclinicBoundary{<:Any, <:Any, <:Any, A}, box) where A
    return TriclinicBoundary(box; approx_images=A)
end

# Check that the interaction cutoff still fits in the box after the boundary is changed,
#   for example by a barostat
function check_boundary_change(sys, strictness)
    max_sqdist = max_zero_beyond(sys.pairwise_inters)
    if !isnothing(max_sqdist) && !iszero(max_sqdist)
        check_cutoff_box_size(sqrt(max_sqdist), sys.boundary, strictness; maxlog=1)
    end
    return nothing
end

# The rotation of the right polar decomposition of `μ`, made right-handed
function rotation_part(μ::SMatrix{D, D}) where D
    sv = svd(Matrix(μ))
    R = SMatrix{D, D}(sv.U * sv.Vt)
    if det(R) < 0
        idx = argmin(sv.S)
        Ufix = Matrix(sv.U)
        Ufix[:, idx] .*= -1
        R = SMatrix{D, D}(Ufix * sv.Vt)
    end
    return R
end

@doc raw"""
    scale_coords!(sys::System{<:Any, AT}, μ::SMatrix{D,D};
                  rotate::Bool=true,
                  ignore_molecules::Bool=false,
                  scale_velocities::Bool=false,
                  strictness=:warn)

Scale the bounding box and coordinates of a system by the matrix ``\mu``, as done by
barostats such as [`CRescaleBarostat`](@ref).

```math
B' = \mu B, \quad \mathbf{r}' = \mu \mathbf{r}, \quad \mathbf{v}' = \mu^{-1} \mathbf{v}
```
where ``B`` is the box matrix and the velocities are only scaled when
`scale_velocities=true`.
If the system has a topology the molecules are moved as rigid units: the center of each
molecule is scaled, and with `rotate=true` the molecule is rotated by the rotational part
of ``\mu``. Setting `ignore_molecules=true` scales every atom instead.

A warning is given at most once if the new box is too small for the interaction
cutoff distance.
"""
function scale_coords!(sys::System{<:Any, AT, T},
                       μ_in::SMatrix{D, D};
                       rotate::Bool=true,
                       ignore_molecules::Bool=false,
                       scale_velocities::Bool=false,
                       strictness=default_strictness()) where {AT, T, D}
    # This function assumes that constrained atoms, and virtual sites and the atoms that
    #   define them, are in the same molecule, meaning that they are scaled appropriately
    if has_infinite_boundary(sys.boundary)
        throw(ArgumentError("infinite boundary not supported for scale_coords!"))
    end

    μ = SMatrix{D, D, T}(μ_in) # Convert scaling matrix to float type of the system
    μinv = inv(μ)

    if ignore_molecules || isnothing(sys.topology)
        # box
        B  = SMatrix{D,D}(ustrip.(box_matrix(sys.boundary)))
        Bu = unit(eltype(eltype(sys.coords)))
        B′ = μ * B
        sys.boundary = rebuild_boundary(sys.boundary, B′ .* Bu)
        # coords
        sys.coords .= to_device([μ * c for c in from_device(sys.coords)], AT)  # keeps units
        # velocities
        if scale_velocities
            sys.velocities .= to_device([μinv * v for v in from_device(sys.velocities)], AT)
        end
        check_boundary_change(sys, strictness)
        return sys
    else
        # units and host copies
        coord_u = unit(eltype(eltype(sys.coords)))
        coords  = from_device(ustrip_vec.(sys.coords))
        b_old_u = sys.boundary
        b_old   = ustrip(coord_u, b_old_u)

        # cell matrices
        B  = SMatrix{D,D}(ustrip.(box_matrix(b_old_u)))
        B′ = μ * B

        # topology
        topo    = sys.topology
        mol_of  = topo.atom_molecule_inds
        centers = molecule_centers(coords, b_old, topo)
        c_box   = box_center(b_old)

        # center molecules into same image
        @inbounds for i in eachindex(coords)
            coords[i] = wrap_coords(coords[i] + c_box - centers[mol_of[i]], b_old)
        end

        # new COMs, stored back in centers
        invB = inv(B)
        @inbounds for m in eachindex(centers)
            s    = invB * centers[m]
            centers[m] = B′ * s # = μ * centers[m]
        end

        # rotation from right polar decomposition
        if rotate
            R = rotation_part(μ)
        else
            R = SMatrix{D,D}(I)
        end

        # new boundary
        b_new_u = rebuild_boundary(b_old_u, B′ .* coord_u)
        b_new   = ustrip(coord_u, b_new_u)

        # place atoms
        if rotate
            @inbounds for i in eachindex(coords)
                m  = mol_of[i]
                δ  = coords[i] - c_box
                δ′ = R * δ
                r′ = δ′ + centers[m]
                coords[i] = wrap_coords(r′, b_new)
            end
        else
            @inbounds for i in eachindex(coords)
                m = mol_of[i]
                coords[i] = wrap_coords(coords[i] - c_box + centers[m], b_new)
            end
        end

        # write back
        sys.coords .= to_device(coords .* coord_u, AT)
        sys.boundary = b_new_u

        # velocities
        if scale_velocities
            vels = from_device(sys.velocities)
            @inbounds for i in eachindex(vels)
                vels[i] = μinv * vels[i]
            end
            sys.velocities .= to_device(vels, AT)
        end

        check_boundary_change(sys, strictness)
        return sys
    end
end

function scale_coords!(sys::System{<:Any, AT, T},
                       μ_in::SMatrix{D, D};
                       rotate::Bool=true,
                       ignore_molecules::Bool=false,
                       scale_velocities::Bool=false,
                       strictness=default_strictness()) where {AT <: AbstractGPUArray, T, D}
    if has_infinite_boundary(sys.boundary)
        throw(ArgumentError("infinite boundary not supported for scale_coords!"))
    end
    μ = SMatrix{D, D, T}(μ_in)
    μinv = inv(μ)
    coord_unit = unit(eltype(eltype(sys.coords)))
    B = box_matrix_nounits(sys.boundary, coord_unit, T)
    B′ = μ * B
    boundary_new = rebuild_boundary(sys.boundary, B′ .* coord_unit)

    if ignore_molecules || isnothing(sys.topology)
        sys.coords .= (μ,) .* sys.coords
        if scale_velocities
            sys.velocities .= (μinv,) .* sys.velocities
        end
    else
        plan = molecule_unwrap_plan(sys.topology, sys.coords)
        shifts = unwrap_molecules_gpu!(plan, sys.coords, inv(B))
        R = (rotate ? SMatrix{D, D, T}(rotation_part(μ)) : SMatrix{D, D, T}(I))
        launch_kernel!(scale_place_kernel!(get_backend(sys.coords), 256), sys.coords,
                       sys.velocities, plan.frac, shifts, plan.center_sums,
                       plan.molecule_atom_counts, plan.atom_molecule_inds, B, μ, R, μinv,
                       boundary_new, coord_unit, Val(D), Val(T), Val(scale_velocities);
                       ndrange=length(sys.coords))
    end
    sys.boundary = boundary_new
    check_boundary_change(sys, strictness)
    return sys
end

@doc raw"""
    dipole_moment(sys)

The dipole moment of a system.

Calculated as the sum of the coordinates multiplied by the partial charges,
``\boldsymbol{\mu} = \sum_i q_i \mathbf{r}_i``.
Requires the charges on the atoms to be set.
"""
dipole_moment(sys) = sum(sys.coords .* charges(sys))

# Conversion to another float type, units or device, see `System(sys; ...)`
function convert_component(c::SystemConversion, b::TriclinicBoundary{<:Any, <:Any, <:Any, A}) where A
    return TriclinicBoundary(convert_component(c, b.basis_vectors); approx_images=A)
end
