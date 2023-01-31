#Add support for vectors in N_unit_cells
#Only supports 3d crystals, cant make something like graphene
#This framework should capture most of the crystals one would want to make
    #but this framework is by no means generic enought to create anything

#stop float promotion in primitive vectors??

export
    nothing

 
#########
# TYEPS #
#########

abstract type Lattice end
abstract type CrystalFamily end
abstract type CenteringType end

#Centering types for multiple dispatch
struct Primitive <: CenteringType end
struct FaceCentered <: CenteringType end
struct BodyCentered <: CenteringType end
struct BaseCentered <: CenteringType end

struct BravaisLattice{T} <: Lattice
    crystal_family::CrystalFamily
    centering_type::CenteringTypes
    primitive_vectors::T
end

struct BasisAtom{T}
    basis_vector::SVector{3,T}
    atom::Atom
end

struct Crystal
    lattice::Lattice
    basis::SVector{BasisAtom}
end


function BravaisLattice(cf::CrystalFamily, ct::CenteringTypes)
    p_vec = get_primitive_vectors(cf,ct)
    return BravaisLattice{typeof(p_vec)}(cf, ct, p_vec)
end


#####################################################

struct Cubic{LC} <: CrystalFamily
    lattice_constants::SVector{3,LC}
end

function Cubic(a)
    return Cubic{typeof(lattice_constant)}(SVector(a,a,a))
end

#####################################################

struct Orthorhombic{LC} <: CrystalFamily
    lattice_constants::SVector{3,LC}
end

function Orthorhombic(a,b,c)
    return Orthorhombic{typeof(a)}(SVector(a,b,c))
end

#####################################################

struct Monoclinic{LC,LA} <: CrystalFamily
    lattice_constants::SVector{3,LC}
    lattice_angles::SVector{3,LA}
end

function Monoclinic(a, b, c, β)
    return Monoclinic{typeof(a),typeof(β)}(SVector(a,b,c), SVector(90u"°", β, 90u"°"))
end

#####################################################

struct Triclinic{LC,LA} <: CrystalFamily
    lattice_constants::SVector{3,LC}
    lattice_angles::SVector{3,LA}
end

function Triclinic(a, b, c, α, β, γ)
    return Triclinic{typeof(a),typeof(β)}(SVector(a,b,c), SVector(α, β, γ))
end
#####################################################

struct Tetragonal{LC} <: CrystalFamily
    lattice_constants::SVector{3,LC}
end

function Tetragonal(a, c)
    return Tetragonal{typeof(a)}(SVector(a,a,c))
end

#####################################################

struct Rhombohedral{LC,LA} <: CrystalFamily
    lattice_constants::SVector{3,LC}
    lattice_angles::SVector{3,LA}
end

function Rhombohedral(a, α)
    return Rhombohedral{typeof(a),typeof(α)}(SVector(a,a,a),SVector(α, α, α))
end

#####################################################

struct Hexagonal{LC,LA} <: CrystalFamily
    lattice_constants::SVector{3,LC}
    lattice_angles::SVector{3,LA}
end

function Hexagonal(a,c,γ)
    return HexagonalBravaisLattice{typeof(a),typeof(γ)}(SVector(a,a,c),SVector(90u"°", 90u"°", γ))
end

#####################################################

function get_primitive_vectors(cf::CrystalFamily, ct::Primitive)
    primitive_vectors = MMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
    primitive_vectors .*= transpose(cf.lattice_constants)

    if hasfield(cf, lattice_angles)
        #rotate a-axis
        rotateAboutB!(view(primitive_vectors,1,:), 90u"°" - cf.lattice_angles[2])
        rotateAboutC!(view(primitive_vectors,1,:), 90u"°" - cf.lattice_angles[3])
        #rotate b-axis
        rotateAboutA!(view(primitive_vectors,2,:) ,90u"°" - cf.lattice_angles[1])
        rotateAboutC!(view(primitive_vectors,2,:), 90u"°" - cf.lattice_angles[3])
        #rotate c-axis
        rotateAboutA!(view(primitive_vectors,3,:), 90u"°" - cf.lattice_angles[1])
        rotateAboutB!(view(primitive_vectors,3,:), 90u"°" - cf.lattice_angles[2])
    end

    return primitive_vectors
end

FaceCenteredSupportedTypes = Union{Cubic, Orthorhombic}
function get_primitive_vectors(cf::FaceCenteredSupportedTypes, ct::FaceCentered)
    primitive_vectors = MMatrix{3,3}([0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0])
    primitive_vectors .*= transpose(cf.lattice_constants)
    return primitive_vectors
end

BodyCenteredSupportedTypes = Union{Cubic, Orthorhombic, Tetragonal}
function get_primitive_vectors(cf::BodyCenteredSupportedTypes, ct::BodyCentered)
    primitive_vectors = MMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.5  0.5  0.5])
    primitive_vectors .*= transpose(cf.lattice_constants)
    return primitive_vectors
end

BaseCenteredSupportedTypes = Union{Monoclinic, Orthorhombic}
function get_primitive_vectors(cf::BaseCenteredSupportedTypes, ct::BaseCentered)

end


#####################################################


function get_lattice_points(lattice::BravaisLattice, N_unit_cells)

    #Store in 1D?
    lattice_points = MArray{Tuple{N_unit_cells,N_unit_cells,N_unit_cells},SVector{3}}(undef)

    for i in range(1,N_unit_cells), j in range(1,N_unit_cells), k in range(1,N_unit_cells)
        lattice_points[i,j,k] = i.*lattice.primitive_vectors[1,:] .+ j.*lattice.primitive_vectors[2,:] .+ k.lattice.primitive_vectors[3,:]
    end

    return lattice_points
end

function build_crystal(crystal::Crystal, N_unit_cells)
    @assert N_unit_cells > 0 "Number of unit cells should be positive"

    #Probably a way to get LP an not allocate all this memory
    lattice_points = get_lattice_points(crystal.lattice, N_unit_cells)
    N_atoms = sum(length, lattice_points) * length(crystal.basis)

    #Create flat arrays for atoms & coords
    atoms = SVector{N_atoms,Atom}
    coords = SVector{N_atoms,SVector{3}}

    #Superimpose basis onto lattice points
    i = 1
    for lp in lattice_points
        for basis_atom in crystal.basis
            coords[i] = lp .+ basis_atom.position
            atoms[i] = basis_atom.atom
            i += 1
        end
    end

    #Create boundary that captures crystal
    boundary = CubicBoundary()

    return atoms, coords, boundary
end


#####################################################
# Helper functions

rotateAboutA!(v, θ) = copyto!(v, MMatrix{3,3}([1.0 0.0 0.0; 0.0 cos(θ) -sin(θ); 0  sin(θ)  cos(θ)]) * v)
rotateAboutB!(v, θ) = copyto!(v, MMatrix{3,3}([cos(θ) 0.0 sin(θ); 0.0 1.0 0.0; -sin(θ) 0.0 cos(θ)]) * v)
rotateAboutC!(v, θ) = copyto!(v, MMatrix{3,3}([cos(θ) -sin(θ) 0.0; sin(θ) cos(θ) 0.0; 0.0  0.0  1.0]) * v)


#Implement common crystal structures

function FCC(a,N_unit_cells)
    bv = BravaisLattice(Cubic(a), face_centered::CenteringTypes)
    basis = SVector(BasisAtom(SVector(0.0,0.0,0.0),Atom(mass=10u"u", σ=0.34u"nm", ϵ=1.005u"kJ * mol^-1"))
    crys = Crystal(bv,basis)
    return build_crystal(crys,N_unit_cells)
end