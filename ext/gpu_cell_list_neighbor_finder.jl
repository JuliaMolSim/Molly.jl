# CUDA implementation of GPUCellListNeighborFinder

# CUDA kernel
function split_gpu_cell_list_coordinates!(
    x, y, z,
    coords,
    n_atoms,
)
    atom_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if atom_i <= n_atoms
        coord = ustrip_vec(coords[atom_i])

        x[atom_i] = Float32(coord[1])
        y[atom_i] = Float32(coord[2])
        z[atom_i] = Float32(coord[3])
    end

    return nothing
end

function split_gpu_cell_list_coordinates!(
    x,
    y,
    z,
    coords,
)
    n_atoms = length(coords)
    n_threads = 256
    n_blocks = cld(n_atoms, n_threads)

    @cuda threads=n_threads blocks=n_blocks split_gpu_cell_list_coordinates!(
        x,
        y,
        z,
        coords,
        n_atoms,
    )

    return nothing
end

# Allocates three arrays then launches CUDA kernel
function split_gpu_cell_list_coordinates(coords)
    n_atoms = length(coords)

    x = CUDA.zeros(Float32, n_atoms)
    y = CUDA.zeros(Float32, n_atoms)
    z = CUDA.zeros(Float32, n_atoms)

    split_gpu_cell_list_coordinates!(x, y, z, coords)

    return x, y, z
end

function Molly.find_neighbors(
    sys::Molly.System{3, AT},
    nf::Molly.GPUCellListNeighborFinder,
    current_neighbors=nothing,
    step_n::Integer=0,
    force_recompute::Bool=false;
    kwargs...,
) where {AT<:CuArray}
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    sys.boundary isa Molly.CubicBoundary || throw(
        ArgumentError(
            "GPUCellListNeighborFinder currently supports only " *
            "three-dimensional CubicBoundary systems, got " *
            "$(typeof(sys.boundary))",
        ),
    )

    Molly.has_infinite_boundary(sys.boundary) && throw(
        ArgumentError(
            "GPUCellListNeighborFinder does not support infinite boundaries",
        ),
    )

    dist_unit = unit(zero(eltype(eltype(sys.coords))))

    box = box_sides(sys.boundary)
    box_Lx = Float32(ustrip(dist_unit, box[1]))
    box_Ly = Float32(ustrip(dist_unit, box[2]))
    box_Lz = Float32(ustrip(dist_unit, box[3]))
    cutoff = Float32(ustrip(dist_unit, nf.dist_cutoff))

    num_cell_x = floor(Int, box_Lx / cutoff)
    num_cell_y = floor(Int, box_Ly / cutoff)
    num_cell_z = floor(Int, box_Lz / cutoff)

    minimum((num_cell_x, num_cell_y, num_cell_z)) >= 3 || throw(
        ArgumentError(
            "GPUCellListNeighborFinder requires at least three cells " *
            "along every box axis; box sides are " *
            "($box_Lx, $box_Ly, $box_Lz) and cutoff is $cutoff",
        ),
    )

    build_pairs = nf.output !== :ragged
    pair_mode = nf.output === :molly_pairs ?
                Val(:molly) :
                Val(:geometric)

    can_reuse_state = (
        current_neighbors isa Molly.GPUCellListNeighborList &&
        current_neighbors.state !== nothing &&
        (!build_pairs || current_neighbors.state.pair_list !== nothing) &&
        current_neighbors.state.n_atoms == length(sys) &&
        current_neighbors.state.box_Lx == box_Lx &&
        current_neighbors.state.box_Ly == box_Ly &&
        current_neighbors.state.box_Lz == box_Lz &&
        current_neighbors.state.cutoff2 == cutoff * cutoff &&
        current_neighbors.state.max_neighbours == Int32(nf.max_neighbors)
    )

    if can_reuse_state
        state = current_neighbors.state

        split_gpu_cell_list_coordinates!(
            state.x,
            state.y,
            state.z,
            sys.coords,
        )
    else
        x, y, z = split_gpu_cell_list_coordinates(sys.coords)

        state = allocate_optimised_gpu_state(
            x,
            y,
            z,
            box_Lx,
            box_Ly,
            box_Lz,
            cutoff;
            max_neighbours=Int32(nf.max_neighbors),
            allocate_pairs=build_pairs,
        )
    end

    build_optimised_cell_list!(state)
    query_gpu_cell_list!(state)
    check_gpu_cell_list_capacity(state)

    n_pairs = if build_pairs
        build_pair_list!(
            state,
            pair_mode,
            nf.eligible,
            nf.special,
        )
    else
        0
    end

    pair_list = build_pairs ? state.pair_list : nothing

    return Molly.GPUCellListNeighborList(
        state.neighbour_counts,
        state.neighbours,
        n_pairs,
        pair_list,
        state,
    )
end

