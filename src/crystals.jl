#Add support for vectors in N_unit_cells
#Only supports 3d crystals, cant make something like graphene

#stop float promotion in primitive vectors??

export
    nothing

#########
# TYEPS #
#########

abstract type Lattice end
abstract type BravaisLattice <: Lattice end

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
    lattice_constants::SVector{3,LC}
    lattice_angles::SVector{3,LA}
end

function HexagonalBravaisLattice(a,c,γ)
    return HexagonalBravaisLattice{typeof(a),typeof(γ)}(SVector(a,a,c),SVector(90u"°", 90u"°", γ))
end

#####################################################


function conventional(bl::BravaisLattice)
    primitive_vectors = MMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
    primitive_vectors .*= transpose(bl.lattice_constants)

    if hasfield(bl, lattice_angles)
        #rotate a-axis
        rotateAboutB!(view(pv,1,:), 90u"°" - bl.lattice_angles[2])
        rotateAboutC!(view(pv,1,:), 90u"°" - bl.lattice_angles[3])
        #rotate b vector
        rotateAboutA!(view(pv,2,:) ,90u"°" - bl.lattice_angles[1])
        rotateAboutC!(view(pv,2,:), 90u"°" - bl.lattice_angles[3])
        #rotate c vector
        rotateAboutA!(view(pv,3,:), 90u"°" - bl.lattice_angles[1])
        rotateAboutB!(view(pv,3,:), 90u"°" - bl.lattice_angles[2])
    end

    return primitive_vectors
end

FC_supported_types = Union{CubicBravaisLattice, OrthorhombicBravaisLattice}
function face_centered(bl::FC_supported_types)
    primitive_vectors = MMatrix{3,3}([0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0])
    primitive_vectors .*= transpose(bl.lattice_constants)
    return primitive_vectors
end

BC_supported_types = Union{CubicBravaisLattice, OrthorhombicBravaisLattice, TetragonalBravaisLattice}
function body_centered(bl::BC_supported_types)
    primitive_vectors = MMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.5  0.5  0.5])
    primitive_vectors .*= transpose(bl.lattice_constants)
    return primitive_vectors
end

Base_supported_types = Union{MonoclinicBravaisLattice, OrthorhombicBravaisLattice}
function base_centered(bl::Base_supported_types)

end

#####################################################

function get_lattice_points(lattice::LatticeWithBasis, N_unit_cells)

    lattice_points = MArray{Tuple{N_unit_cells,N_unit_cells,N_unit_cells},SVector{3}}(undef)
end

function get_lattice_points(lattice::BravaisLattice, N_unit_cells)

    lattice_points = MArray{Tuple{N_unit_cells,N_unit_cells,N_unit_cells},SVector{3}}(undef)

    for i in range(0,N_unit_cells), j in range(0,N_unit_cells), k in range(0,N_unit_cells)
        lattice_points[i,j,k] = i*crystal.b
    end
end

function build_crystal(crystal::Crystal, N_unit_cells)
    @assert N_unit_cells > 0 "Number of unit cells should be positive"
    # Generate coordiantes from lattice
    lattice_points = get_lattice_points(crystal.lattice, N_unit_cells)

    #Superimpose basis onto those points

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

rotateAboutA!(v, θ) = copyto!(v, MMatrix{3,3}([1.0 0.0 0.0; 0.0 cos(θ) -sin(θ); 0  sin(θ)  cos(θ)]) * v)
rotateAboutB!(v, θ) = copyto!(v, MMatrix{3,3}([cos(θ) 0.0 sin(θ); 0.0 1.0 0.0; -sin(θ) 0.0 cos(θ)]) * v)
rotateAboutC!(v, θ) = copyto!(v, MMatrix{3,3}([cos(θ) -sin(θ) 0.0; sin(θ) cos(θ) 0.0; 0.0  0.0  1.0]) * v)

