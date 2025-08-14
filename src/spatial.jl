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
    random_velocity,
    maxwell_boltzmann,
    random_velocities,
    random_velocities!,
    bond_angle,
    torsion_angle,
    remove_CM_motion!,
    virial,
    pressure,
    molecule_centers,
    scale_coords!,
    dipole_moment

"""
    CubicBoundary(x, y, z)
    CubicBoundary(x)

Cubic 3D bounding box defined by three side lengths.

If one length is given then all three sides will have that length.
Setting one or more values to `Inf` gives no boundary in that dimension.
"""
struct CubicBoundary{T}
    side_lengths::SVector{3, T}
    function CubicBoundary(side_lengths::SVector{3, D}; check_positive::Bool=true) where D
        if check_positive && any(l -> l <= zero(l), side_lengths)
            throw(DomainError("side lengths must be positive, got $side_lengths"))
        end
        new{D}(side_lengths)
    end
end

CubicBoundary(x, y, z; kwargs...) = CubicBoundary(SVector{3}(x, y, z); kwargs...)
CubicBoundary(x::Number; kwargs...) = CubicBoundary(SVector{3}(x, x, x); kwargs...)

Base.getindex(b::CubicBoundary, i::Integer) = b.side_lengths[i]
Base.firstindex(b::CubicBoundary) = 1
Base.lastindex(b::CubicBoundary) = 3

function Base.zero(b::CubicBoundary{T}) where T
    return CubicBoundary(zero(SVector{3, T}); check_positive=false)
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
struct RectangularBoundary{T}
    side_lengths::SVector{2, T}
    function RectangularBoundary(side_lengths::SVector{2, D}; check_positive::Bool=true) where D
        if check_positive && any(l -> l <= zero(l), side_lengths)
            throw(DomainError("side lengths must be positive, got $side_lengths"))
        end
        new{D}(side_lengths)
    end
end

RectangularBoundary(x, y; kwargs...) = RectangularBoundary(SVector{2}(x, y); kwargs...)
RectangularBoundary(x::Number; kwargs...) = RectangularBoundary(SVector{2}(x, x); kwargs...)

Base.getindex(b::RectangularBoundary, i::Integer) = b.side_lengths[i]
Base.firstindex(b::RectangularBoundary) = 1
Base.lastindex(b::RectangularBoundary) = 2

function Base.zero(b::RectangularBoundary{T}) where T
    return RectangularBoundary(zero(SVector{2, T}); check_positive=false)
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

Not currently able to simulate a cubic box, use [`CubicBoundary`](@ref) or small
offsets instead.
Not currently compatible with infinite boundaries.
"""
struct TriclinicBoundary{T, A, D, I}
    basis_vectors::SVector{3, SVector{3, D}}
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

function TriclinicBoundary(bv::SVector{3}; approx_images::Bool=true)
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
    α = bond_angle(bv[2], bv[3])
    β = bond_angle(bv[1], bv[3])
    γ = bond_angle(bv[1], bv[2])
    # Precompute angles to speed up coordinate wrapping
    tan_bprojyz_cprojyz = tan(bond_angle(
        SVector(zero(bv[2][1]), bv[2][2], bv[2][3]),
        SVector(zero(bv[3][1]), bv[3][2], bv[3][3]),
    ))
    tan_c_cprojxy = tan(bond_angle(SVector(bv[3][1], bv[3][2], zero(bv[3][3])), bv[3]))
    a_cprojxy = bond_angle(bv[1], SVector(bv[3][1], bv[3][2], zero(bv[3][3])))
    tan_a_b = tan(bond_angle(bv[1], bv[2]))
    return TriclinicBoundary{typeof(α), approx_images, eltype(eltype(bv)), eltype(reciprocal_size)}(
                                bv, α, β, γ, reciprocal_size, tan_bprojyz_cprojyz, tan_c_cprojxy,
                                cos(a_cprojxy), sin(a_cprojxy), tan_a_b)
end

function TriclinicBoundary(bv_lengths::SVector{3}, angles::SVector{3}; kwargs...)
    if any(!ispositive, bv_lengths)
        throw(ArgumentError("basis vector lengths must be positive " *
                            "when constructing a TriclinicBoundary, got $bv_lengths"))
    end
    if !all(a -> 0 < a < π, angles)
        throw(ArgumentError("basis vector angles must be 0 to π radians " *
                            "when constructing a TriclinicBoundary, got $angles"))
    end
    α, β, γ = angles
    cos_α, cos_β, cos_γ, sin_γ = cos(α), cos(β), cos(γ), sin(γ)
    z = zero(bv_lengths[1])
    v1 = SVector(bv_lengths[1], z, z)
    v2 = SVector(bv_lengths[2] * cos_γ, bv_lengths[2] * sin_γ, z)
    v3x = bv_lengths[3] * cos_β
    v3y = bv_lengths[3] * (cos_α - cos_β * cos_γ) / sin_γ
    v3z = sqrt(bv_lengths[3] ^ 2 - v3x ^ 2 - v3y ^ 2)
    v3 = SVector(v3x, v3y, v3z)
    return TriclinicBoundary(SVector{3}(v1, v2, v3); kwargs...)
end

TriclinicBoundary(v1, v2, v3; kwargs...) = TriclinicBoundary(SVector{3}(v1, v2, v3); kwargs...)
TriclinicBoundary(arr; kwargs...) = TriclinicBoundary(SVector{3}(arr); kwargs...)

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

AtomsBase.n_dimensions(::CubicBoundary) = 3
AtomsBase.n_dimensions(::RectangularBoundary) = 2
AtomsBase.n_dimensions(::TriclinicBoundary) = 3

Base.broadcastable(b::Union{CubicBoundary, RectangularBoundary}) = b.side_lengths

float_type(b::Union{CubicBoundary, RectangularBoundary}) = typeof(ustrip(b[1]))
float_type(b::TriclinicBoundary{T}) where {T} = T

length_type(b::Union{CubicBoundary{T}, RectangularBoundary{T}}) where {T} = T
length_type(b::TriclinicBoundary{T, A, D}) where {T, A, D} = D

Unitful.ustrip(b::CubicBoundary) = CubicBoundary(ustrip.(b.side_lengths))
Unitful.ustrip(u::Unitful.Units, b::CubicBoundary) = CubicBoundary(ustrip.(u, b.side_lengths))
Unitful.ustrip(b::RectangularBoundary) = RectangularBoundary(ustrip.(b.side_lengths))
Unitful.ustrip(u::Unitful.Units, b::RectangularBoundary) = RectangularBoundary(ustrip.(u, b.side_lengths))

function AtomsBase.cell_vectors(b::CubicBoundary)
    z = zero(b[1])
    bb = (
        SVector(b[1], z   , z   ),
        SVector(z   , b[2], z   ),
        SVector(z   , z   , b[3]),
    )
    return unit(z) == NoUnits ? (bb .* u"nm") : bb # Assume nm without other information
end

function AtomsBase.cell_vectors(b::RectangularBoundary)
    z = zero(b[1])
    bb = (
        SVector(b[1], z   ),
        SVector(z   , b[2]),
    )
    return unit(z) == NoUnits ? (bb .* u"nm") : bb
end

function AtomsBase.cell_vectors(b::TriclinicBoundary)
    bb = (b.basis_vectors[1], b.basis_vectors[2], b.basis_vectors[3])
    return unit(b[1][1]) == NoUnits ? (bb .* u"nm") : bb
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
has_infinite_boundary(b::TriclinicBoundary) = false
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
volume(b::Union{CubicBoundary, RectangularBoundary, TriclinicBoundary}) = prod(box_sides(b))

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
function random_coord(boundary::CubicBoundary; rng=Random.default_rng())
    return rand(rng, SVector{3, float_type(boundary)}) .* boundary
end

function random_coord(boundary::RectangularBoundary; rng=Random.default_rng())
    return rand(rng, SVector{2, float_type(boundary)}) .* boundary
end

function random_coord(boundary::TriclinicBoundary{T}; rng=Random.default_rng()) where T
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

function vector(c1, c2, boundary::TriclinicBoundary{T, true}) where T
    dr = c2 - c1
    dr -= boundary.basis_vectors[3] * floor(dr[3] * boundary.reciprocal_size[3] + T(0.5))
    dr -= boundary.basis_vectors[2] * floor(dr[2] * boundary.reciprocal_size[2] + T(0.5))
    dr -= boundary.basis_vectors[1] * floor(dr[1] * boundary.reciprocal_size[1] + T(0.5))
    return dr
end

function vector(c1, c2, boundary::TriclinicBoundary{T, false}) where T
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

function random_velocities(sys::System{3, AT}, temp; rng=Random.default_rng()) where AT <: AbstractGPUArray
    return to_device(random_velocity_3D.(from_device(masses(sys)), temp, sys.k, rng), AT)
end

function random_velocities(sys::System{2, AT}, temp; rng=Random.default_rng()) where AT <: AbstractGPUArray
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
    total_mass = zero(eltype(masses_cpu))
    for i in eachindex(sys)
        cm_momentum += velocities_cpu[i] * masses_cpu[i]
        total_mass += masses_cpu[i]
    end
    cm_velocity = cm_momentum / total_mass
    sys.velocities .= sys.velocities .- (cm_velocity,)
    return sys
end

@doc raw"""
    virial(sys, neighbors=find_neighbors(sys), step_n=0; n_threads=Threads.nthreads())
    virial(inter, sys, neighbors, step_n; n_threads=Threads.nthreads())

Calculate the virial of a system or the virial resulting from a general interaction.

The virial is defined as
```math
\Xi = -\frac{1}{2} \sum_{i,j>i} r_{ij} \cdot F_{ij}
```
Custom general interaction types can implement this function.

This should only be used on systems containing just pairwise interactions, or
where the specific interactions, constraints and general interactions without
[`virial`](@ref) defined do not contribute to the virial.
"""
function virial(sys; n_threads::Integer=Threads.nthreads())
    return virial(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads)
end

function virial(sys, neighbors, step_n::Integer=0; n_threads::Integer=Threads.nthreads())
    pairwise_inters_nonl = filter(!use_neighbors, values(sys.pairwise_inters))
    pairwise_inters_nl   = filter( use_neighbors, values(sys.pairwise_inters))
    v = virial(sys, neighbors, step_n, pairwise_inters_nonl, pairwise_inters_nl)

    for inter in values(sys.general_inters)
        v += virial(inter, sys, neighbors, step_n; n_threads=n_threads)
    end

    return v
end

function virial(sys::System{D, AT, T}, neighbors_dev, step_n, pairwise_inters_nonl,
                            pairwise_inters_nl) where {D, AT, T}
    if AT <: AbstractGPUArray
        coords     = from_device(sys.coords)
        velocities = from_device(sys.velocities)
        atoms      = from_device(sys.atoms)
        if isnothing(neighbors_dev)
            neighbors = neighbors_dev
        else
            neighbors = NeighborList(neighbors_dev.n, from_device(neighbors_dev.list))
        end
    else
        coords, velocities, atoms = sys.coords, sys.velocities, sys.atoms
        neighbors = neighbors_dev
    end

    boundary = sys.boundary
    v = zero(T) * sys.energy_units

    @inbounds if length(pairwise_inters_nonl) > 0
        n_atoms = length(sys)
        for i in 1:n_atoms
            for j in (i + 1):n_atoms
                dr = vector(coords[i], coords[j], boundary)
                f = force(pairwise_inters_nonl[1], dr, atoms[i], atoms[j], sys.force_units, false,
                          coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                for inter in pairwise_inters_nonl[2:end]
                    f += force(inter, dr, atoms[i], atoms[j], sys.force_units, false,
                               coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                end
                v += dot(f, dr)
            end
        end
    end

    @inbounds if length(pairwise_inters_nl) > 0
        if isnothing(neighbors)
            error("an interaction uses the neighbor list but neighbors is nothing")
        end
        for ni in eachindex(neighbors)
            i, j, special = neighbors[ni]
            dr = vector(coords[i], coords[j], boundary)
            f = force(pairwise_inters_nl[1], dr, atoms[i], atoms[j], sys.force_units, special,
                      coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
            for inter in pairwise_inters_nl[2:end]
                f += force(inter, dr, atoms[i], atoms[j], sys.force_units, special,
                           coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
            end
            v += dot(f, dr)
        end
    end

    return -v / 2
end

# Default for general interactions
function virial(inter, sys::System{D, AT, T}, args...; kwargs...) where {D, AT, T}
    return zero(T) * sys.energy_units
end

@doc raw"""
    pressure(sys, neighbors=find_neighbors(sys), step_n=0; n_threads=Threads.nthreads())

Calculate the pressure of a system.

The pressure is defined as
```math
P = \frac{1}{V} \left( NkT - \frac{2}{D} \Xi \right)
```
where `V` is the system volume, `N` is the number of atoms, `k` is the Boltzmann constant,
`T` is the system temperature, `D` is the number of dimensions and `Ξ` is the virial
calculated using [`virial`](@ref).

This should only be used on systems containing just pairwise interactions, or
where the specific interactions, constraints and general interactions without
[`virial`](@ref) defined do not contribute to the virial.
Not compatible with infinite boundaries.
"""
function pressure(sys; n_threads::Integer=Threads.nthreads())
    return pressure(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads)
end

function pressure(sys::AtomsBase.AbstractSystem{D}, neighbors, step_n::Integer=0;
                  n_threads::Integer=Threads.nthreads()) where D
    if has_infinite_boundary(sys.boundary)
        error("pressure calculation not compatible with infinite boundaries")
    end
    NkT = energy_remove_mol(length(sys) * sys.k * temperature(sys))
    vir = energy_remove_mol(virial(sys, neighbors, step_n; n_threads=n_threads))
    P = (NkT - (2 * vir) / D) / volume(sys.boundary)
    if sys.energy_units == NoUnits || D != 3
        # If implied energy units are (u * nm^2 * ps^-2) and everything is
        #   consistent then this has implied units of (u * nm^-1 * ps^-2)
        #   for 3 dimensions and (u * ps^-2) for 2 dimensions
        return P
    else
        # Sensible unit to return by default for 3 dimensions
        return uconvert(u"bar", P)
    end
end

"""
    molecule_centers(coords, boundary, topology)

Calculate the coordinates of the center of each molecule in a system.

Accounts for periodic boundary conditions by using the circular mean.
If `topology=nothing` then the coordinates are returned.

Not currently compatible with [`TriclinicBoundary`](@ref) if the topology is set.
"""
function molecule_centers(coords::AbstractArray{SVector{D, C}}, boundary, topology) where {D, C}
    if isnothing(topology)
        return coords
    elseif boundary isa TriclinicBoundary
        error("calculating molecule centers is not compatible with a TriclinicBoundary")
    else
        T = float_type(boundary)
        pit = T(π)
        twopit = 2 * pit
        n_molecules = length(topology.molecule_atom_counts)
        unit_circle_angles = broadcast(coords, (boundary.side_lengths,)) do c, sl
            (c ./ sl) .* twopit .- pit # Run -π to π
        end
        mol_sin_sums = zeros(SVector{D, T}, n_molecules)
        mol_cos_sums = zeros(SVector{D, T}, n_molecules)
        for (uca, mi) in zip(unit_circle_angles, topology.atom_molecule_inds)
            mol_sin_sums[mi] += sin.(uca)
            mol_cos_sums[mi] += cos.(uca)
        end
        frac_centers = zeros(SVector{D, T}, n_molecules)
        for mi in 1:n_molecules
            frac_centers[mi] = (atan.(mol_sin_sums[mi], mol_cos_sums[mi]) .+ pit) ./ twopit
        end
        return broadcast((c, b) -> c .* b, frac_centers, (boundary.side_lengths,))
    end
end

function molecule_centers(coords::AbstractGPUArray, boundary, topology)
    AT = array_type(coords)
    return to_device(molecule_centers(from_device(coords), boundary, topology), AT)
end

# Allows scaling multiple vectors at once by broadcasting this function
scale_vec(v, s) = v .* s

"""
    scale_coords!(sys, scale_factor; ignore_molecules=false)

Scale the coordinates and bounding box of a system by a scaling factor.

The scaling factor can be a single number or a `SVector` of the appropriate number
of dimensions corresponding to the scaling factor for each axis.
Velocities are not scaled.
If the topology of the system is set then atoms in the same molecule will be
moved by the same amount according to the center of coordinates of the molecule.
This can be disabled with `ignore_molecules=true`.

Not currently compatible with [`TriclinicBoundary`](@ref) if the topology is set.
"""
function scale_coords!(sys::System{<:Any, AT}, scale_factor; ignore_molecules=false) where AT
    if ignore_molecules || isnothing(sys.topology)
        sys.boundary = scale_boundary(sys.boundary, scale_factor)
        sys.coords .= scale_vec.(sys.coords, (scale_factor,))
    elseif sys.boundary isa TriclinicBoundary
        error("scaling coordinates by molecule is not compatible with a TriclinicBoundary")
    else
        atom_molecule_inds = sys.topology.atom_molecule_inds
        coords_nounits = from_device(ustrip_vec.(sys.coords))
        coord_units = unit(eltype(eltype(sys.coords)))
        boundary_nounits = ustrip(coord_units, sys.boundary)
        mol_centers = molecule_centers(coords_nounits, boundary_nounits, sys.topology)
        # Shift molecules to the center of the box and wrap
        # This puts them all in the same periodic image which is required when scaling
        # This won't work if the molecule can't fit in one box when centered,
        #   but that would likely be a pathological case anyway
        center_shifts = (box_center(boundary_nounits),) .- mol_centers
        for i in eachindex(sys)
            coords_nounits[i] = wrap_coords(
                    coords_nounits[i] .+ center_shifts[atom_molecule_inds[i]], boundary_nounits)
        end
        # Move all atoms in a molecule by the same amount according to the molecule center
        # Then move the atoms back to the molecule center and wrap in the scaled boundary
        shift_vecs = scale_vec.(mol_centers, (scale_factor .- 1,))
        sys.boundary = scale_boundary(sys.boundary, scale_factor)
        boundary_nounits = ustrip(sys.boundary)
        for i in eachindex(sys)
            mi = atom_molecule_inds[i]
            coords_nounits[i] = wrap_coords(
                    coords_nounits[i] .+ shift_vecs[mi] .- center_shifts[mi], boundary_nounits)
        end
        sys.coords .= to_device(coords_nounits .* coord_units, AT)
    end
    return sys
end

"""
    dipole_moment(sys)

The dipole moment μ of a system.

Requires the charges on the atoms to be set.
"""
dipole_moment(sys) = sum(sys.coords .* charges(sys))
