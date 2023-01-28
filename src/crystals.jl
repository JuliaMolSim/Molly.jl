export
    SC,
    FCC,
    BCC


abstract type BravaisLattice end

#Currently only single atom basis supported
#How to incorperate lattice sites to allow for multi-atom crystals eg NaCl
#stop float promotion in primitive vectors??

struct CubicBravaisLattice{PV,LC} <: BravaisLattice
    primitive_vectors::PV
    a::LC
end

function CubicBravaisLattice(a)
    primitive_vectors = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] .* a
    return CubicBravaisLattice{typeof(primitive_vectors),typeof(lattice_constant)}(
                primitive_vectors, lattice_constant)
end

#####################################################

struct OrthorhombicBravaisLattice{PV,LC} <: BravaisLattice
    primitive_vectors::PV
    a::LC
    b::LC
    c::LC
end

function OrthorhombicBravaisLattice(a,b,c)
    primitive_vectors = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] .* [a,b,c]
    return OrthorhombicBravaisLattice{typeof(primitive_vectors),typeof(a)}(
                    primitive_vectors, lattice_constant)
end

#####################################################

struct MonoclinicBravaisLattice{PV,LC,LA} <: BravaisLattice
    primitive_vectors::PV
    a::LC
    b::LC
    c::LC
    β::LA
end

function MonoclinicBravaisLattice(a, b, c, β)
    primitive_vectors = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] .* [a,b,c]
    rotateAboutX!(primitive_vectors, 90*u"°" - β)
    return MonoclinicBravaisLattice{typeof(primitive_vectors),typeof(a),typeof(β)}(
                    primitive_vectors, lattice_constant)
end

#####################################################

struct TriclinicBravaisLattice{PV,LC,LA} <: BravaisLattice
    primitive_vectors::PV
    a::LC
    b::LC
    c::LC
    α::LA
    β::LA
    γ::LA
end

#####################################################

struct TetragonalBravaisLattice{PV,LC} <: BravaisLattice
    primitive_vectors::PV
    a::LC
    c::LC
end

function TetragonalBravaisLattice(a,c)
    temp_lattice_constants = [a, a, c]
    primitive_vectors = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] .* temp_lattice_constants
    return TetragonalBravaisLattice{typeof(primitive_vectors),typeof(a)}(
                    primitive_vectors, lattice_constant)
end

#####################################################

struct RhombohedralBravaisLattice{PV,LC,LA} <: BravaisLattice
    primitive_vectors::PV
    a::LC
    α::LA
end

#####################################################

struct HexagonalBravaisLattice{PV,LC,LA} <: BravaisLattice
    primitive_vectors::PV
    a::LC
    c::LC
    α::LA
end

#####################################################

# All types of bravais lattice have a conventional cell
function conventional(bl::BravaisLattice, num_unit_cells)
    coords = 
    boundary = 
    return coords, boundary
end

FC_supported_types = Union{CubicBravaisLattice, OrthorhombicBravaisLattice}
function face_centered(bl::FC_supported_types, num_unit_cells)

    return coords, boundary
end

Base_supported_types = Union{MonoclinicBravaisLattice, OrthorhombicBravaisLattice}
function base_centered(bl::Base_supported_types, num_unit_cells)

    return coords, boundary
end

BC_supported_types = Union{CubicBravaisLattice, OrthorhombicBravaisLattice, TetragonalBravaisLattice}
function body_centered(bl::BC_supported_types, num_unit_cells)

    return coords, boundary
end


#####################################################
# Implement some crystals

SC(a, num_unit_cells) = conventional(CubicBravaisLattice(a, num_unit_cells))
FCC(a, num_unit_cells) = face_centered(CubicBravaisLattice(a, num_unit_cells))
BCC(a, num_unit_cells) = body_centered(CubicBravaisLattice(a, num_unit_cells))


#####################################################
# Helper functions

function rotateAboutX!(v, θ)
    
end

function rotateAboutY!(v, θ)

end

function rotateAboutZ!(v, θ)

end




# function get_positions(ab::AtomicBasis)

# end

# struct AtomicBasis{} <: AbstractCrystal

# end