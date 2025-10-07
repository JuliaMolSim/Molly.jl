# Spatial calculations

export
    CubicBoundary,
    RectangularBoundary,
    TriclinicBoundary,
    volume,
    density,
    box_center,
    scale_boundary,
    random_coord,
    vector_1D,
    vector,
    wrap_coord_1D,
    wrap_coords,
    unwrap_molecules,
    random_velocity,
    maxwell_boltzmann,
    random_velocities,
    random_velocities!,
    bond_angle,
    torsion_angle,
    remove_CM_motion!,
    pressure,
    scalar_pressure,
    molecule_centers,
    scale_coords!,
    dipole_moment

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

boxmatrix(b::CubicBoundary) = SMatrix{3,3}(ustrip(b.side_lengths[1]), 0, 0, 0, ustrip(b.side_lengths[2]), 0, 0, 0, ustrip(b.side_lengths[3])) * unit(b.side_lengths[1])

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

boxmatrix(b::RectangularBoundary) = SMatrix{3,3}(b.side_lengths[1],0,0, 0,b.side_lengths[2],0, 0,0,one(eltype(b.side_lengths)))

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

Not currently compatible with infinite boundaries.
"""
struct TriclinicBoundary{D, T, C, A, I} <: AbstractBoundary{D, T, C}
    basis_vectors::SVector{3, SVector{3, C}}
    α::T
    β::T
    γ::T
    reciprocal_size::SVector{3, I}
    tan_bprojyz_cprojyz::T
    tan_c_cprojxy::T
    cos_a_cprojxy::T
    sin_a_cprojxy::T
    tan_a_b::T
end

ispositive(x) = x > zero(x)

function TriclinicBoundary(bv::Union{SVector{3,<:SVector{3}}, SMatrix{3,3}}; approx_images::Bool=true)

    # Normalize input, if box matrix is passed turn into SVector{3, SVector{3, T}}
    if bv isa SMatrix
        bv = SVector(bv[:,1], bv[:,2], bv[:,3])
    end

    # numeric and unit types
    NT  = typeof(ustrip(bv[1][1]))              # underlying floating type
    uL  = (bv[1][1] isa Unitful.Quantity) ? unit(bv[1][1]) : one(NT)
    tolL = sqrt(eps(NT)) * uL                   # length tolerance (with units)
    tol0 = sqrt(eps(NT))                        # unitless tolerance

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

    # tan(angle between b_yz and c_yz)
    by, bz = bv[2][2], bv[2][3]
    cy, cz = bv[3][2], bv[3][3]
    n_byz = hypot(by, bz)
    n_cyz = hypot(cy, cz)
    tan_bprojyz_cprojyz =
        (n_byz ≤ tolL || n_cyz ≤ tolL) ? zero(NT) :
        let num = ustrip(abs(by*cz - bz*cy)), den = ustrip(by*cy + bz*cz)
            abs(den) ≤ tol0 ? NT(Inf) : NT(abs(num/den))
        end

    # tan(angle between c and its xy-projection), and a vs c_xy direction
    cx, cxy, cz3 = bv[3][1], bv[3][2], bv[3][3]
    n_cxy = hypot(cx, cxy)
    if n_cxy ≤ tolL
        tan_c_cprojxy = NT(Inf)          # c ⟂ xy-plane
        cos_a_cprojxy = one(NT)          # define dir(c_xy) = +x
        sin_a_cprojxy = zero(NT)
    else
        tan_c_cprojxy = NT(ustrip(abs(cz3) / n_cxy))
        cos_a_cprojxy = NT(ustrip(cx / n_cxy))
        sin_a_cprojxy = NT(ustrip(cxy / n_cxy))
    end

    # tan(angle between a and b) = b_y / b_x
    bx = bv[2][1]
    tan_a_b = (abs(bx) ≤ tolL) ? NT(Inf) : NT(ustrip(bv[2][2] / bx))

    return TriclinicBoundary{3, NT, eltype(eltype(bv)), approx_images, eltype(reciprocal_size)}(
                bv, α, β, γ, reciprocal_size,
                tan_bprojyz_cprojyz, tan_c_cprojxy,
                cos_a_cprojxy, sin_a_cprojxy, tan_a_b
            )
end

function TriclinicBoundary(bv_lengths, angles; kwargs...)
    if any(!ispositive, bv_lengths)
        throw(ArgumentError("basis vector lengths must be positive, got $bv_lengths"))
    end
    if !all(a -> 0 < ustrip(a) < π, angles)
        throw(ArgumentError("angles must be in (0, π), got $angles"))
    end

    # underlying numeric type and unit for lengths
    NT  = typeof(ustrip(bv_lengths[1]))
    has_units = bv_lengths[1] isa Unitful.Quantity
    uL  = has_units ? unit(bv_lengths[1]) : one(NT)
    tol0 = sqrt(eps(NT))

    # compute in Float64 for stability
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

    # convert back to input length type, zero-out tiny components
    toL(x::Real) = ((abs(x) < tol0) ? zero(NT) : NT(x)) * uL
    v1 = SVector(toL(v1x), toL(v1y), toL(v1z))
    v2 = SVector(toL(v2x), toL(v2y), toL(v2z))
    v3 = SVector(toL(v3x), toL(v3y), toL(v3z))

    return TriclinicBoundary(SVector(v1, v2, v3); kwargs...)
end

TriclinicBoundary(v1, v2, v3; kwargs...) = TriclinicBoundary(SVector{3}(v1, v2, v3); kwargs...)
TriclinicBoundary(arr; kwargs...)        = TriclinicBoundary(SVector{3}(arr); kwargs...)

boxmatrix(b::TriclinicBoundary)   = SMatrix{3,3}(hcat(b.basis_vectors...) )

Base.getindex(b::TriclinicBoundary, i::Integer) = b.basis_vectors[i]
Base.firstindex(b::TriclinicBoundary) = 1
Base.lastindex(b::TriclinicBoundary) = 3

function Chemfiles.UnitCell(b::TriclinicBoundary)
    if unit(eltype(eltype(b.basis_vectors))) == NoUnits
        Chemfiles.UnitCell(Float64.(Array(hcat(b.basis_vectors...))) .* 10) # Assume nm
    else
        Chemfiles.UnitCell(Float64.(Array(ustrip.(u"Å", hcat(b.basis_vectors...)))))
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
float_type(::AbstractBoundary{<:Any, T}) where {T} = T
length_type(b::AbstractBoundary{<:Any, <:Any, C}) where {C} = C

Unitful.ustrip(b::CubicBoundary) = CubicBoundary(ustrip.(b.side_lengths))
Unitful.ustrip(u::Unitful.Units, b::CubicBoundary) = CubicBoundary(ustrip.(u, b.side_lengths))
Unitful.ustrip(b::RectangularBoundary) = RectangularBoundary(ustrip.(b.side_lengths))
Unitful.ustrip(u::Unitful.Units, b::RectangularBoundary) = RectangularBoundary(ustrip.(u, b.side_lengths))

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

has_infinite_boundary(b::Union{CubicBoundary, RectangularBoundary}) = any(isinf, b.side_lengths)
has_infinite_boundary(b::TriclinicBoundary) = false # Not currently supported
has_infinite_boundary(sys::System) = has_infinite_boundary(sys.boundary)

n_infinite_dims(b::Union{CubicBoundary, RectangularBoundary}) = sum(isinf, b.side_lengths)
n_infinite_dims(b::TriclinicBoundary) = 0
n_infinite_dims(sys::System) = n_infinite_dims(sys.boundary)

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
    m = sum(mass, sys.atoms)
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

function scale_boundary(b::TriclinicBoundary, scale)
    return TriclinicBoundary(b[1] .* scale, b[2] .* scale, b[3] .* scale)
end

# The minimum cubic box surrounding the bounding box, used for visualization
cubic_bounding_box(b::Union{CubicBoundary, RectangularBoundary}) = b.side_lengths
cubic_bounding_box(b::TriclinicBoundary) = sum(b.basis_vectors)

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
    v12_p_sl = v12 + side_length
    v12_m_sl = v12 - side_length
    return ifelse(
        v12 > zero(c1),
        ifelse( v12 < -v12_m_sl, v12, v12_m_sl),
        ifelse(-v12 <  v12_p_sl, v12, v12_p_sl),
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

function vector(c1, c2, boundary::TriclinicBoundary{3, T, <:Any, true}) where T
    dr = c2 - c1
    dr -= boundary.basis_vectors[3] * floor(dr[3] * boundary.reciprocal_size[3] + T(0.5))
    dr -= boundary.basis_vectors[2] * floor(dr[2] * boundary.reciprocal_size[2] + T(0.5))
    dr -= boundary.basis_vectors[1] * floor(dr[1] * boundary.reciprocal_size[1] + T(0.5))
    return dr
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
function vector_pad3D(c1::SVector{2, T}, c2::SVector{2, T}, boundary::RectangularBoundary{T}) where T
    SVector{3, T}(
        vector_1D(c1[1], c2[1], boundary[1]),
        vector_1D(c1[2], c2[2], boundary[2]),
        zero(T),
    )
end

vector_pad3D(c1::SVector{3}, c2::SVector{3}, boundary) = vector(c1, c2, boundary)

# Trim a vector back to 2D if required
trim3D(v::SVector{3, T}, boundary::RectangularBoundary{T}) where T = SVector{2, T}(v[1], v[2])
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
    v_wrap -= bv[2] * floor((v_wrap[2] - v_wrap[3] / boundary.tan_bprojyz_cprojyz) * rs[2])
    dz_projxy = v_wrap[3] / boundary.tan_c_cprojxy
    dx = dz_projxy * boundary.cos_a_cprojxy
    dy = dz_projxy * boundary.sin_a_cprojxy
    # Bound in x-axis
    v_wrap -= bv[1] * floor((v_wrap[1] - dx - (v_wrap[2] - dy) / boundary.tan_a_b) * rs[1])
    return v_wrap
end

function unwrap_global(coords::AbstractVector{<:SVector{D}},
                       boundary, topology; neighbors=nothing) where {D}
    # --- frac<->cart ---
    if hasproperty(boundary, :basis_vectors)
        @assert D == 3
        Bm = reduce(hcat, boundary.basis_vectors)   # unitful
        B  = SMatrix{3,3}(Bm)
        to_frac = (r::SVector{3}) -> B \ r          # dimensionless
        to_cart = (f::SVector{3}) -> B * f          # length units
    else
        sl = boundary.side_lengths                  # SVector{D} with length units
        to_frac = (r::SVector{D}) -> r ./ sl        # dimensionless
        to_cart = (f::SVector{D}) -> f .* sl        # length units
    end
    wrap01(v) = v .- floor.(v .+ eps(eltype(v)))     # keep in [0,1)

    # --- wrapped fractional coords ---
    N  = length(coords)
    f1 = to_frac(coords[1])
    f  = Vector{typeof(f1)}(undef, N)                # dimensionless
    @inbounds for i in 1:N
        f[i] = wrap01(to_frac(coords[i]))
    end

    # --- global adjacency: bonds ∪ neighbor-list pairs ---    
    adj = [Int[] for _ in 1:N]
    if topology !== nothing
        @inbounds for (i32,j32) in topology.bonded_atoms
            i = Int(i32); j = Int(j32); push!(adj[i], j); push!(adj[j], i)
        end
        if neighbors !== nothing
            @inbounds for ni in eachindex(neighbors)
                i, j = neighbors[ni][1], neighbors[ni][2]
                push!(adj[i], j); push!(adj[j], i)
            end
        end
    end

    # --- BFS over whole system (one lattice tiling) ---
    u = similar(f)                                   # dimensionless
    visited = falses(N)
    @inbounds for seed in 1:N
        visited[seed] && continue
        u[seed] = f[seed]
        visited[seed] = true
        stack = Int[seed]
        while !isempty(stack)
            i = pop!(stack); fi = f[i]; ui = u[i]
            for j in adj[i]
                visited[j] && continue
                df = f[j] - fi
                df -= round.(df)                     # shift to (-0.5,0.5]
                u[j] = ui + df
                visited[j] = true
                push!(stack, j)
            end
        end
    end

    # --- back to Cartesian with units ---
    out = Vector{typeof(coords[1])}(undef, N)
    @inbounds for i in 1:N
        out[i] = to_cart(u[i])
    end
    return out
end

"""
    unwrap_molecules(coords, boundary, topology)

Return coordinates unwrapped so that every bonded pair is placed using
the minimum-image displacement. Molecule connectivity is preserved.
"""
unwrap_molecules(coords, boundary, topology) =
    unwrap_global(from_device(coords), boundary, topology; neighbors = nothing)

unwrap_molecules(sys::System; neighbors = nothing) =
    unwrap_global(from_device(sys.coords), sys.boundary, sys.topology; neighbors)

"""
    random_velocity(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature;
                    dims=3, rng=Random.default_rng())
    random_velocity(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature,
                    k::Union{BoltzmannConstUnits, MolarBoltzmannConstUnits};
                    dims=3, rng=Random.default_rng())
    random_velocity(atom_mass::Real, temp::Real, k::Real=ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k);
                    dims=3, rng=Random.default_rng())

Generate a random velocity from the Maxwell-Boltzmann distribution, with
optional custom Boltzmann constant.
"""
function random_velocity(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature;
                         dims::Integer=3, rng=Random.default_rng())
    return SVector([maxwell_boltzmann(atom_mass, temp; rng=rng) for i in 1:dims]...)
end

function random_velocity(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature,
                         k::Union{BoltzmannConstUnits, MolarBoltzmannConstUnits};
                         dims::Integer=3, rng=Random.default_rng())
    return SVector([maxwell_boltzmann(atom_mass, temp, k; rng=rng) for i in 1:dims]...)
end

function random_velocity(atom_mass::Real, temp::Real,
                         k::Real=ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k);
                         dims::Integer=3, rng=Random.default_rng())
    return SVector([maxwell_boltzmann(atom_mass, temp, k; rng=rng) for i in 1:dims]...)
end

function random_velocity_3D(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature,
                            rng=Random.default_rng())
    return SVector(
        maxwell_boltzmann(atom_mass, temp; rng=rng),
        maxwell_boltzmann(atom_mass, temp; rng=rng),
        maxwell_boltzmann(atom_mass, temp; rng=rng),
    )
end

function random_velocity_3D(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature,
                            k::Union{BoltzmannConstUnits, MolarBoltzmannConstUnits},
                            rng=Random.default_rng())
    return SVector(
        maxwell_boltzmann(atom_mass, temp, k; rng=rng),
        maxwell_boltzmann(atom_mass, temp, k; rng=rng),
        maxwell_boltzmann(atom_mass, temp, k; rng=rng),
    )
end

function random_velocity_3D(atom_mass::Real, temp::Real, k::Real, rng=Random.default_rng())
    return SVector(
        maxwell_boltzmann(atom_mass, temp, k; rng=rng),
        maxwell_boltzmann(atom_mass, temp, k; rng=rng),
        maxwell_boltzmann(atom_mass, temp, k; rng=rng),
    )
end

function random_velocity_2D(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature,
                            rng=Random.default_rng())
    return SVector(
        maxwell_boltzmann(atom_mass, temp; rng=rng),
        maxwell_boltzmann(atom_mass, temp; rng=rng),
    )
end

function random_velocity_2D(atom_mass::Union{Unitful.Mass, MolarMass}, temp::Unitful.Temperature,
                            k::Union{BoltzmannConstUnits, MolarBoltzmannConstUnits},
                            rng=Random.default_rng())
    return SVector(
        maxwell_boltzmann(atom_mass, temp, k; rng=rng),
        maxwell_boltzmann(atom_mass, temp, k; rng=rng),
    )
end

function random_velocity_2D(atom_mass::Real, temp::Real, k::Real, rng=Random.default_rng())
    return SVector(
        maxwell_boltzmann(atom_mass, temp, k; rng=rng),
        maxwell_boltzmann(atom_mass, temp, k; rng=rng),
    )
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
    random_velocities(sys, temp; rng=Random.default_rng())

Generate random velocities from the Maxwell-Boltzmann distribution
for a [`System`](@ref).
"""
function random_velocities(sys::AtomsBase.AbstractSystem{3}, temp; rng=Random.default_rng())
    return random_velocity_3D.(masses(sys), temp, sys.k, rng)
end

function random_velocities(sys::AtomsBase.AbstractSystem{2}, temp; rng=Random.default_rng())
    return random_velocity_2D.(masses(sys), temp, sys.k, rng)
end

function random_velocities(sys::System{3, AT}, temp;
                           rng=Random.default_rng()) where AT <: AbstractGPUArray
    return to_device(random_velocity_3D.(from_device(masses(sys)), temp, sys.k, rng), AT)
end

function random_velocities(sys::System{2, AT}, temp;
                           rng=Random.default_rng()) where AT <: AbstractGPUArray
    return to_device(random_velocity_2D.(from_device(masses(sys)), temp, sys.k, rng), AT)
end

"""
    random_velocities!(sys, temp; rng=Random.default_rng())
    random_velocities!(vels, sys, temp; rng=Random.default_rng())

Set the velocities of a [`System`](@ref), or a vector, to random velocities
generated from the Maxwell-Boltzmann distribution.
"""
function random_velocities!(sys, temp; rng=Random.default_rng())
    sys.velocities .= random_velocities(sys, temp; rng=rng)
    return sys
end

function random_velocities!(vels, sys::AbstractSystem, temp; rng=Random.default_rng())
    vels .= random_velocities(sys, temp; rng=rng)
    return vels
end

# Sometimes domain error occurs for acos if the value is > 1.0 or < -1.0
acosbound(x::Real) = acos(clamp(x, -1, 1))

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
    acosbound(dot(vec_ji, vec_jk) / (norm(vec_ji) * norm(vec_jk)))
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

"""
    remove_CM_motion!(system)

Remove the center of mass motion from a [`System`](@ref).
"""
function remove_CM_motion!(sys)
    masses_cpu = from_device(masses(sys))
    velocities_cpu = from_device(sys.velocities)
    cm_momentum = zero(eltype(velocities_cpu)) .* zero(eltype(masses_cpu))
    for i in eachindex(sys)
        cm_momentum += velocities_cpu[i] * masses_cpu[i]
    end
    cm_velocity = cm_momentum / sys.total_mass
    sys.velocities .= sys.velocities .- (cm_velocity,)
    return sys
end

function remove_CM_motion!(sys::System{<:Any, <:AbstractGPUArray})
    cm_momentum = mapreduce((v, m) -> v .* m, +, sys.velocities, masses(sys))
    cm_velocity = cm_momentum / sys.total_mass
    sys.velocities .= sys.velocities .- (cm_velocity,)
    return sys
end

@doc raw"""
    pressure(system, neighbors=find_neighbors(system), step_n=0, buffers=nothing;
             recompute=true, n_threads=Threads.nthreads())

Calculate the pressure tensor of the system.

The pressure is defined as
```math
\bf{P} = \frac{ 2 \cdot \bf{K} + \bf{W} }{V}
```
where ``V`` is the system volume, ``\bf{K}`` is the kinetic energy tensor
and ``\bf{W}`` is the virial tensor.

To calculate the scalar pressure, see [`scalar_pressure`](@ref).

Not compatible with infinite boundaries.
"""
function pressure(sys; n_threads::Integer=Threads.nthreads())
    return pressure(sys, find_neighbors(sys; n_threads=n_threads), 0, nothing;
                    recompute=true, n_threads=n_threads)
end

function pressure(sys::System{D}, neighbors, step_n::Integer=0, buffers_in=nothing;
                  recompute::Bool=true, n_threads::Integer=Threads.nthreads()) where D
    if isnothing(buffers_in)
        buffers = init_buffers!(sys, n_threads)
    else
        buffers = buffers_in
    end
    if recompute
        forces!(zero_forces(sys), sys, neighbors, buffers, Val(true), step_n; n_threads=n_threads)
        kin_tensor = buffers.kin_tensor
        vir_tensor = buffers.virial
    else
        kin_tensor = buffers.kin_tensor
        vir_tensor = buffers.virial
    end

    # Always evaluate K in case velocities were rescaled by a thermostat
    kinetic_energy_tensor!(sys, kin_tensor)
    if has_infinite_boundary(sys.boundary)
        error("pressure calculation not compatible with infinite boundaries")
    end

    K = energy_remove_mol.(kin_tensor) # (1/2) Σ m v⊗v
    W = energy_remove_mol.(vir_tensor) # Σ r⊗f

    P = (2 .* K .+ W) ./ volume(sys.boundary)
    if sys.energy_units == NoUnits || D != 3
        # If implied energy units are (u * nm^2 * ps^-2) and everything is
        #   consistent then this has implied units of (u * nm^-1 * ps^-2)
        #   for 3 dimensions and (u * ps^-2) for 2 dimensions
        buffers.pres_tensor .= P
    else
        # Sensible unit to return by default for 3 dimensions
        P_bar = uconvert.(u"bar", P)
        buffers.pres_tensor .= P_bar
    end
    return buffers.pres_tensor
end

"""
    scalar_pressure(system, neighbors=find_neighbors(system), step_n=0, buffers=nothing;
                    recompute=true, n_threads=Threads.nthreads())

Calculate the pressure of the system as a scalar.

This is the trace of the [`pressure`](@ref) tensor.
"""
function scalar_pressure(sys; n_threads::Integer=Threads.nthreads())
    return scalar_pressure(sys, find_neighbors(sys; n_threads=n_threads), 0, nothing;
                           recompute=true, n_threads=n_threads)
end

function scalar_pressure(sys::System{D}, neighbors, step_n::Integer=0, buffers=nothing;
                         recompute::Bool=true, n_threads::Integer=Threads.nthreads()) where D
    P = pressure(sys, neighbors, step_n, buffers; recompute=recompute, n_threads=n_threads)
    return tr(P) / D
end

"""
    molecule_centers(coords::AbstractArray{SVector{D,C}}, boundary, topology) where {D,C}

Center-of-geometry per molecule using unwrapped **fractional** coordinates.
Works for orthorhombic (with `boundary.side_lengths`) and triclinic (with `boundary.basis_vectors`).

Requires:
- `topology.atom_molecule_inds :: AbstractVector{Int}` (length = n_atoms)
- `topology.molecule_atom_counts :: AbstractVector{Int}`
- `topology.bonded_atoms :: AbstractVector{<:Tuple{Int,Int}}` or `AbstractVector{SVector{2,Int}}`
"""
function molecule_centers(coords::AbstractArray{SVector{D,C}}, boundary, topology) where {D,C}
    # Fallback
    if isnothing(topology)
        return coords
    end

    # Helpers
    wrap01(v) = v .- floor.(v)

    # Boundary transforms
    is_triclinic = hasproperty(boundary, :basis_vectors)
    if is_triclinic && D != 3
        error("Triclinic boundary only defined for D=3")
    end

    # Build frac<->cart transforms
    if is_triclinic
        B = boxmatrix(boundary)
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

    # Topology
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
    for m in 1:n_mol
        atoms = atoms_by_mol[m]
        if isempty(atoms)
            centers[m] = to_cart(zero(f[1]))
            continue
        end

        visited = falses(N)

        # Search over each connected component within the molecule
        for seed in atoms
            if visited[seed]; continue; end
            u[seed] = f[seed]
            visited[seed] = true
            stack = [seed]
            while !isempty(stack)
                i = pop!(stack)
                @inbounds for j in nbrs[i]
                    # stay within molecule
                    if atom_mol[j] != m || visited[j]; continue; end
                    Δ = f[j] - f[i] - round.(f[j] - f[i])
                    u[j] = u[i] + Δ
                    visited[j] = true
                    push!(stack, j)
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

function molecule_centers(coords::AbstractGPUArray,
                          boundary::AbstractBoundary{<:Any, T},
                          topology) where T
    AT = array_type(coords)
    return to_device(molecule_centers(from_device(coords), boundary, topology), AT)
end

rebuild_boundary(b::CubicBoundary,       box) = CubicBoundary(box)
rebuild_boundary(b::RectangularBoundary, box) = RectangularBoundary(box)
rebuild_boundary(b::TriclinicBoundary,   box) = TriclinicBoundary(box)

"""
    scale_coords!(sys::System{<:Any, AT}, μ::SMatrix{D,D};
                  rotate::Bool = true,
                  ignore_molecules::Bool = false,
                  scale_velocities::Bool = false)

Rigid-molecular barostat update with optional rotation.

- Box:        B′ = μ * B
- Positions:  r′ = μ * r  (implemented via COM affine + optional rotation of internal offsets)
- Velocities: v′ = μ⁻¹ * v  (applied when `scale_velocities=true`)
"""
function scale_coords!(sys::System{<:Any, AT},
                       μ::SMatrix{D,D};
                       rotate::Bool           = true,
                       ignore_molecules::Bool = false,
                       scale_velocities::Bool = false) where {AT,D}

    if has_infinite_boundary(sys.boundary)
        throw(AssertionError("Infinite boundary not supported"))
    end
    
    μinv = inv(μ)

    if ignore_molecules || isnothing(sys.topology)
        # box
        B  = SMatrix{D,D}(ustrip.(boxmatrix(sys.boundary)))
        Bu = unit(eltype(eltype(sys.coords)))
        B′ = μ * B
        sys.boundary = rebuild_boundary(sys.boundary, B′ .* Bu)
        # coords
        sys.coords .= to_device([μ * c for c in from_device(sys.coords)], AT)  # keeps units
        # velocities
        if scale_velocities
            sys.velocities .= to_device([μinv * v for v in from_device(sys.velocities)], AT)
        end
        return sys
    else
        # units and host copies
        coord_u = unit(eltype(eltype(sys.coords)))
        coords  = from_device(ustrip_vec.(sys.coords))
        b_old_u = sys.boundary
        b_old   = ustrip(coord_u, b_old_u)

        # cell matrices
        B  = SMatrix{D,D}(ustrip.(boxmatrix(b_old_u)))
        B′ = μ * B

        # topology
        topo    = sys.topology
        mol_of  = topo.atom_molecule_inds
        centers = molecule_centers(coords, b_old, topo)
        c_box   = box_center(b_old)

        # center molecules into same image
        Δcenter = [c_box - centers[m] for m in eachindex(centers)]
        @inbounds for i in eachindex(coords)
            coords[i] = wrap_coords(coords[i] + Δcenter[mol_of[i]], b_old)
        end

        # new COMs
        invB = inv(B)
        centers′ = similar(centers)
        @inbounds for m in eachindex(centers)
            s    = invB * centers[m]
            rcom = B′ * s                  # = μ * centers[m]
            centers′[m] = rcom
        end

        # rotation from right polar decomposition
        if rotate
            sv = svd(Matrix(μ))
            R  = SMatrix{D,D}(sv.U * sv.Vt)
            if det(R) < 0 # Ensure right-handed rotation
                idx  = argmin(sv.S)
                Ufix = Matrix(sv.U); Ufix[:,idx] .*= -1
                R    = SMatrix{D,D}(Ufix * sv.Vt)
            end
        else
            R = SMatrix{D,D}(I)
        end

        # new boundary
        b_new_u = rebuild_boundary(b_old_u, B′ .* coord_u)
        b_new   = ustrip(coord_u, b_new_u) 

        # place atoms
        @inbounds for i in eachindex(coords)
            m  = mol_of[i]
            δ  = coords[i] - c_box
            δ′ = R * δ
            r′ = δ′ + centers′[m]
            coords[i] = wrap_coords(r′, b_new)
        end

        # write back
        sys.coords   .= to_device(coords .* coord_u, AT)
        sys.boundary  = b_new_u

        # velocities
        if scale_velocities
            vels = from_device(sys.velocities)          # keep units
            @inbounds for i in eachindex(vels)
                vels[i] = μinv * vels[i]
            end
            sys.velocities .= to_device(vels, AT)
        end

        return sys
    end
end

"""
    dipole_moment(sys)

The dipole moment μ of a system.

Requires the charges on the atoms to be set.
"""
dipole_moment(sys) = sum(sys.coords .* charges(sys))
