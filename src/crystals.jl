#Only supports 3d crystals, cant make something like graphene

#stop float promotion in primitive vectors??

export
    SC,
    FCC,
    BCC

#########
# TYEPS #
#########

abstract type Lattice end
abstract type BravaisLattice <: Lattice end #is this the right way to allow both LatticeWithBasis and a BravaisLattice to be passed to Crystal()?

#Allows construction of non-bravais lattices (e.g. diamond)
struct LatticeWithBasis{B} <: Lattice
    lattice::BravaisLattice
    basis_vectors::SMatrix{B,3}
end

#Specific pattern at each bravais lattice point
struct Basis{T}
    position::SVector{3,T}
    atom::Atom
end

struct Crystal
    lattice::Lattice
    basis::SVector{BasisAtom}
end

#####################################################

struct CubicBravaisLattice{LC} <: BravaisLattice
    lattice_constants::SVector{3,LC}
end

function CubicBravaisLattice(a)
    return CubicBravaisLattice{typeof(lattice_constant)}(SVector(a,a,a))
end

#####################################################

struct OrthorhombicBravaisLattice{LC} <: BravaisLattice
    lattice_constants::SVector{3,LC}
end

function OrthorhombicBravaisLattice(a,b,c)
    return OrthorhombicBravaisLattice{typeof(a)}(SVector(a,b,c))
end

#####################################################

struct MonoclinicBravaisLattice{LC,LA} <: BravaisLattice
    lattice_constants::SVector{3,LC}
    lattice_angles::SVector{3,LA}
end

function MonoclinicBravaisLattice(a, b, c, β)
    return MonoclinicBravaisLattice{typeof(a),typeof(β)}(SVector(a,b,c), SVector(90u"°", β, 90u"°"))
end

#####################################################

struct TriclinicBravaisLattice{LC,LA} <: BravaisLattice
    lattice_constants::SVector{3,LC}
    lattice_angles::SVector{3,LA}
end

function TriclinicBravaisLattice(a, b, c, α, β, γ)
    return TriclinicBravaisLattice{typeof(a),typeof(β)}(SVector(a,b,c), SVector(α, β, γ))
end
#####################################################

struct TetragonalBravaisLattice{LC} <: BravaisLattice
    lattice_constants::SVector{3,LC}
end

function TetragonalBravaisLattice(a, c)
    return TetragonalBravaisLattice{typeof(a)}(SVector(a,a,c))
end

#####################################################

struct RhombohedralBravaisLattice{LC,LA} <: BravaisLattice
    lattice_constants::SVector{3,LC}
    lattice_angles::SVector{3,LA}
end

function RhombohedralBravaisLattice(a, α)
    return RhombohedralBravaisLattice{typeof(a),typeof(α)}(SVector(a,a,a),SVector(α, α, α))
end

#####################################################

struct HexagonalBravaisLattice{PV,LC,LA} <: BravaisLattice
    edge_vectors::SVector{3, SVector{3, PV}}
    a::LC
    c::LC
    α::LA
end

function HexagonalBravaisLattice(a,c,γ)
    return HexagonalBravaisLattice{typeof(a),typeof(γ)}(SVector(a,a,c),SVector(90u"°", 90u"°", γ))
end

#####################################################


function Conventional(bl::BravaisLattice)
    primitive_vectors = MMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
    primitive_vectors .*= transpose(bl.lattice_constants)
    return primitive_vectors
end

FC_supported_types = Union{CubicBravaisLattice, OrthorhombicBravaisLattice}
function FaceCentered(bl::FC_supported_types)
    primitive_vectors = MMatrix{3,3}([0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0])
    primitive_vectors .*= transpose(bl.lattice_constants)
    return primitive_vectors
end

BC_supported_types = Union{CubicBravaisLattice, OrthorhombicBravaisLattice, TetragonalBravaisLattice}
function BodyCentered(bl::BC_supported_types)
    primitive_vectors = MMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.5  0.5  0.5])
    primitive_vectors .*= transpose(bl.lattice_constants)
    return primitive_vectors
end

Base_supported_types = Union{MonoclinicBravaisLattice, OrthorhombicBravaisLattice}
function BaseCentered(bl::Base_supported_types)

end

#####################################################

function BuildCrystal(crystal::Crystal, num_unit_cells)
    # Build crystal geometry

    return atoms, coords, boundary
end


# These functions populate the conventional cell for the specific structures

# cubic_unit_cell = [0.0, 0.0, 0.0; ]
# return coords, boundary, atoms



# #####################################################
# # Implement some crystals



# # Cubic Crystals
# Cubic(a, num_unit_cells) = conventional(CubicBravaisLattice(a), num_unit_cells)
# FCC(a, num_unit_cells) = face_centered(CubicBravaisLattice(a), num_unit_cells)
# BCC(a, num_unit_cells) = body_centered(CubicBravaisLattice(a), num_unit_cells)

# # Triclinic Crystals
# Triclinic(a,b,c,α,β,γ,num_unit_cells) = conventional(TriclinicBravaisLattice(a,b,c,α,β,γ),num_unit_cells)

# #Monoclinic Crystals
# Monoclinic() = conventional()
# BaseCenteredMonoclinic() = base_centered()

# #Orthorhombic Crystals
# Orthorhombic() = conventional()
# BaseCenteredOrthorhombic() = base_centered()
# BodyCenteredOrthorhombic() = body_centered()
# FaceCenteredOrthorhombic() = face_centered()


#Multi-Atom Basis Crystals


#####################################################
# Helper functions

function rotateAboutX!(v, θ)
    
end

function rotateAboutY!(v, θ)

end

function rotateAboutZ!(v, θ)

end
