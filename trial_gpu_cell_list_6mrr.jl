using Molly
using CUDA
using Chemfiles
using StaticArrays
using Unitful

pdb_path = joinpath(
    dirname(pathof(Molly)),
    "..",
    "data",
    "6mrr_equil.pdb",
)

println("Reading PDB...")

trajectory = Chemfiles.Trajectory(pdb_path)
frame = Chemfiles.read(trajectory)

coords_cpu = [
    Float32.(SVector{3}(column) / 10.0)
    for column in eachcol(Chemfiles.positions(frame))
]

n_atoms = length(coords_cpu)

boundary = Molly.boundary_from_chemfiles(
    Chemfiles.UnitCell(frame),
    Float32,
    NoUnits,
)

println("Number of atoms: ", n_atoms)
println("Boundary: ", Molly.box_sides(boundary))
println("Copying system to GPU...")

atoms = CuArray([
    Molly.Atom(index=i, mass=1.0f0)
    for i in 1:n_atoms
])

finder = GPUCellListNeighborFinder(
    dist_cutoff=1.0f0,
    n_steps=10,
    max_neighbors=640,
)

sys = System(
    atoms=atoms,
    coords=CuArray(coords_cpu),
    boundary=boundary,
    neighbor_finder=finder,
    force_units=NoUnits,
    energy_units=NoUnits,
)

println("Running GPU cell-list finder...")

neighbors = find_neighbors(sys)
CUDA.synchronize()

counts = Array(neighbors.counts)
maximum_count = maximum(counts)
total_count = sum(Int64, counts)
capacity = size(neighbors.neighbors, 1)

println("Maximum neighbours per atom: ", maximum_count)
println("Total directed neighbours: ", total_count)
println("Capacity: ", capacity)
println("Within capacity: ", maximum_count <= capacity)
println("Result type: ", typeof(neighbors))


sample_atoms = collect(1:100:n_atoms)

coords_reference = Array(sys.coords)
gpu_counts = Array(neighbors.counts)

reference_counts = zeros(Int, length(sample_atoms))
cutoff2 = 1.0f0^2

for (sample_i, atom_i) in enumerate(sample_atoms)
    coord_i = coords_reference[atom_i]
    count = 0

    for atom_j in eachindex(coords_reference)
        atom_i == atom_j && continue

        dr = Molly.vector(
            coord_i,
            coords_reference[atom_j],
            sys.boundary,
        )

        if sum(abs2, dr) <= cutoff2
            count += 1
        end
    end

    reference_counts[sample_i] = count
end

sample_gpu_counts = Int.(gpu_counts[sample_atoms])

println(
    "Sample counts agree: ",
    sample_gpu_counts == reference_counts,
)

if sample_gpu_counts != reference_counts
    mismatches = findall(sample_gpu_counts .!= reference_counts)
    println("Number of sample mismatches: ", length(mismatches))

    for mismatch in first(mismatches, min(10, length(mismatches)))
        atom_i = sample_atoms[mismatch]
        println(
            "Atom ",
            atom_i,
            ": GPU=",
            sample_gpu_counts[mismatch],
            ", reference=",
            reference_counts[mismatch],
        )
    end
end