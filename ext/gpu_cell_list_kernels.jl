
const CELL_BLOCK_SIZE = 32

# GPU kernel for getting CELL IDs
function get_cell_id!(cell_ids, x, y, z, n_atoms, num_cell_x, num_cell_y, num_cell_z, cell_Lx, cell_Ly, cell_Lz)

    global_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x # define global thread ID 

    if global_id <= n_atoms
        atom_i_x = x[global_id] # global read of atom i's coords
        atom_i_y = y[global_id]
        atom_i_z = z[global_id]

        cell_id_x = floor(Int32, atom_i_x / cell_Lx) # calculate cell id / dim, 0-based index
        cell_id_y = floor(Int32, atom_i_y / cell_Ly)
        cell_id_z = floor(Int32, atom_i_z / cell_Lz)

        cell_id_x = min(max(cell_id_x, Int32(0)), num_cell_x - Int32(1)) # protect cell id / dim against edge case
        cell_id_y = min(max(cell_id_y, Int32(0)), num_cell_y - Int32(1))
        cell_id_z = min(max(cell_id_z, Int32(0)), num_cell_z - Int32(1))

        cell_id_flat = (Int32(1) + cell_id_x # calculate global cell id converting from 0 to 1-based
                    + (cell_id_y * num_cell_x) # move in iy jumps by a whole ix, iz jumps by whole ix and iy
                    + (cell_id_z * num_cell_x * num_cell_y))

        cell_ids[global_id] = cell_id_flat # global write
    end
    
    return nothing 
end

# GPU kernel for taking cell IDs and ouputting cell counts 
function get_cell_counts_gpu!(cell_counts, cell_ids, n_atoms)

    global_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if global_id <= n_atoms
        cell = cell_ids[global_id]
        CUDA.@atomic cell_counts[cell] += Int32(1)
    end

    return nothing
end

# GPU kernel
function get_cell_particles_gpu!(cell_particles, cell_write_counts, cell_ids, cell_offsets, n_atoms)
    # output: cell_particles, length n_atoms where atom indices are arranged by cell 

    global_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x 

    if global_id <= n_atoms
        cell = cell_ids[global_id] 

        slot = CUDA.atomic_add!(pointer(cell_write_counts, cell), Int32(1)) # atomic increment the writing slot for the thread

        write_pos = cell_offsets[cell] + slot

        cell_particles[write_pos] = global_id # write id based on position
        
    end
    return nothing 
end

function gather_cell_coordinates_gpu!(
    cell_x,
    cell_y,
    cell_z,
    cell_particles,
    x,
    y,
    z,
    n_atoms,
)
    cell_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        
    if cell_index <= n_atoms
        atom_index = cell_particles[cell_index]

        cell_x[cell_index] = x[atom_index]
        cell_y[cell_index] = y[atom_index]
        cell_z[cell_index] = z[atom_index]
    end

    return nothing
end

function inclusive_to_offsets_gpu!(
    cell_offsets,
    inclusive_counts,
    n_cells,
)
    cell = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if cell <= n_cells
        if cell == Int32(1)
            cell_offsets[cell] = Int32(1)
        else
            cell_offsets[cell] = inclusive_counts[cell - Int32(1)] + Int32(1)
        end
    end

    return nothing
end

function count_geometric_half_pairs_gpu!(
    pair_counts,
    neighbour_counts,
    neighbours,
    n_atoms,
)
    atom_i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if atom_i <= n_atoms
        count = Int32(0)
        n_neighbours = neighbour_counts[atom_i]

        for slot in Int32(1):n_neighbours
            atom_j = neighbours[slot, atom_i]

            if atom_i > atom_j
                count += Int32(1)
            end
        end

        pair_counts[atom_i] = count
    end

    return nothing
end

function write_geometric_half_pairs_gpu!(
    pair_list,
    pair_offsets,
    neighbour_counts,
    neighbours,
    n_atoms,
)
    atom_i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if atom_i <= n_atoms
        write_position = pair_offsets[atom_i]
        n_neighbours = neighbour_counts[atom_i]

        for slot in Int32(1):n_neighbours
            atom_j = neighbours[slot, atom_i]

            if atom_i > atom_j
                pair_list[write_position] = (
                    Int32(atom_i),
                    Int32(atom_j),
                    false,
                )

                write_position += Int32(1)
            end
        end
    end

    return nothing
end


function get_neighbours_cell_shared_contiguous_gpu!(
    neighbour_counts,
    neighbours,
    cell_counts,
    cell_offsets,
    cell_particles,
    cell_x,
    cell_y,
    cell_z,
    n_atoms,
    num_cell_x,
    num_cell_y,
    num_cell_z,
    box_Lx,
    box_Ly,
    box_Lz,
    cutoff2,
    max_neighbours
)
    # Block and thread ID - dont care about global IDs anymore
    host_cell = blockIdx().x
    lane = threadIdx().x

    # Where host cell atom data starts and host cell atom counts
    host_start = cell_offsets[host_cell] 
    n_host = cell_counts[host_cell]

    # Make sure every block knows the 3D position of its host cell
    cell0 = host_cell - Int32(1)
    cx = cell0 % num_cell_x
    tmp = cell0 ÷ num_cell_x
    cy = tmp % num_cell_y
    cz = tmp ÷ num_cell_y

    # Arrays to hold one tile of candidate atoms 
    # These are not for host atoms or neighbour lists 
    shared_x = CuStaticSharedArray(Float32, CELL_BLOCK_SIZE)
    shared_y = CuStaticSharedArray(Float32, CELL_BLOCK_SIZE)
    shared_z = CuStaticSharedArray(Float32, CELL_BLOCK_SIZE)
    shared_ids = CuStaticSharedArray(Int32, CELL_BLOCK_SIZE)

    # Outer loop incase a cell contains more than 32 atoms, but so all threads are firing
    for host_tile_start in Int32(0):Int32(CELL_BLOCK_SIZE):n_host-Int32(1)
        host_local = host_tile_start + lane - Int32(1)
        host_active = host_local < n_host # check if it is a real atom 
        
        # Initialise values for all lanes incase of inactive lanes
        atom_i = Int32(0)
        x_i = 0.0f0
        y_i = 0.0f0
        z_i = 0.0f0
        count = Int32(0)

        if host_active
            host_index = host_start + host_local

            atom_i = cell_particles[host_index]
            x_i = cell_x[host_index]
            y_i = cell_y[host_index]
            z_i = cell_z[host_index]
        end

        for dz in Int32(-1):Int32(1)
            nz = (cz + dz + num_cell_z) % num_cell_z

            for dy in Int32(-1):Int32(1)
                ny = (cy + dy + num_cell_y) % num_cell_y

                for dx in Int32(-1):Int32(1)
                    nx = (cx + dx + num_cell_x) % num_cell_x

                    candidate_cell = Int32(1) + nx + num_cell_x * (ny + num_cell_y * nz)
                    candidate_start = cell_offsets[candidate_cell]
                    n_candidates = cell_counts[candidate_cell]

                    # Candidate tiling here now 
                    candidate_tile_start = Int32(0)

                    while candidate_tile_start < n_candidates
                        candidate_local = candidate_tile_start + lane - Int32(1) 
                        
                        tile_count = min(Int32(CELL_BLOCK_SIZE), n_candidates 
                            - candidate_tile_start)
                        
                        if candidate_local < n_candidates
                            candidate_index = candidate_start + candidate_local

                            shared_ids[lane] = cell_particles[candidate_index]
                            shared_x[lane] = cell_x[candidate_index]
                            shared_y[lane] = cell_y[candidate_index]
                            shared_z[lane] = cell_z[candidate_index]
                        end

                        sync_threads()

                        # Compare active host atoms against the tile
                        if host_active

                            for candidate_lane in Int32(1):tile_count
                                atom_j = shared_ids[candidate_lane]
                                
                                if atom_j != atom_i
                                    dx_ij = shared_x[candidate_lane] - x_i
                                    dy_ij = shared_y[candidate_lane] - y_i
                                    dz_ij = shared_z[candidate_lane] - z_i

                                    dx_ij -= box_Lx * floor(dx_ij / box_Lx + 0.5f0)
                                    dy_ij -= box_Ly * floor(dy_ij / box_Ly + 0.5f0)
                                    dz_ij -= box_Lz * floor(dz_ij / box_Lz + 0.5f0)

                                    r2 = (
                                        dx_ij * dx_ij +
                                        dy_ij * dy_ij +
                                        dz_ij * dz_ij
                                    )

                                    if r2 <= cutoff2
                                        count += Int32(1)

                                        if count <= max_neighbours
                                            neighbours[count, atom_i] = atom_j
                                            # global write each time 
                                        end
                                    end

                                end
                            end
                        end

                        sync_threads()

                        candidate_tile_start += Int32(CELL_BLOCK_SIZE)

                    end
                end
            end
        end

        if host_active
            neighbour_counts[atom_i] = count
        end

        # Every thread must still participate in all sync_threads() calls
    end

    
    return nothing 
end

function allocate_optimised_gpu_state(
    x_gpu,
    y_gpu,
    z_gpu,
    box_Lx,
    box_Ly,
    box_Lz,
    cutoff;
    max_neighbours=Int32(128),
    allocate_pairs=false,
)
    n_atoms = length(x_gpu)

    num_cell_x = floor(Int32, box_Lx / cutoff)
    num_cell_y = floor(Int32, box_Ly / cutoff)
    num_cell_z = floor(Int32, box_Lz / cutoff)

    cell_Lx = box_Lx / num_cell_x
    cell_Ly = box_Ly / num_cell_y
    cell_Lz = box_Lz / num_cell_z

    n_cells = Int(num_cell_x * num_cell_y * num_cell_z)

    pair_capacity = cld(n_atoms * Int(max_neighbours), 2)

    return (
        x=x_gpu,
        y=y_gpu,
        z=z_gpu,
        cell_ids=CUDA.zeros(Int32, n_atoms),
        cell_counts=CUDA.zeros(Int32, n_cells),
        inclusive_counts=CUDA.zeros(Int32, n_cells),
        cell_offsets=CUDA.zeros(Int32, n_cells),
        cell_write_counts=CUDA.zeros(Int32, n_cells),
        cell_particles=CUDA.zeros(Int32, n_atoms),
        cell_x=CUDA.zeros(Float32, n_atoms),
        cell_y=CUDA.zeros(Float32, n_atoms),
        cell_z=CUDA.zeros(Float32, n_atoms),
        neighbour_counts=CUDA.zeros(Int32, n_atoms),
        neighbours=CUDA.zeros(
            Int32,
            Int(max_neighbours),
            n_atoms,
        ),
        pair_counts=allocate_pairs ? CUDA.zeros(Int32, n_atoms) : nothing,
        pair_inclusive_counts=allocate_pairs ? CUDA.zeros(Int32, n_atoms) : nothing,
        pair_offsets=allocate_pairs ? CUDA.zeros(Int32, n_atoms) : nothing,
        pair_list=allocate_pairs ? CuArray{
            Tuple{Int32,Int32,Bool},
        }(undef, pair_capacity) : nothing,
        pair_capacity=pair_capacity,
        n_atoms=n_atoms,
        n_cells=n_cells,
        num_cell_x=num_cell_x,
        num_cell_y=num_cell_y,
        num_cell_z=num_cell_z,
        cell_Lx=cell_Lx,
        cell_Ly=cell_Ly,
        cell_Lz=cell_Lz,
        box_Lx=box_Lx,
        box_Ly=box_Ly,
        box_Lz=box_Lz,
        cutoff2=cutoff * cutoff,
        max_neighbours=max_neighbours,
    )
end

function build_optimised_cell_list!(state)
    n_threads = 256
    n_blocks = cld(state.n_atoms, n_threads)

    fill!(state.cell_counts, Int32(0))
    fill!(state.cell_write_counts, Int32(0))

    @cuda threads=n_threads blocks=n_blocks get_cell_id!(
        state.cell_ids,
        state.x,
        state.y,
        state.z,
        state.n_atoms,
        state.num_cell_x,
        state.num_cell_y,
        state.num_cell_z,
        state.cell_Lx,
        state.cell_Ly,
        state.cell_Lz,
    )

    @cuda threads=n_threads blocks=n_blocks get_cell_counts_gpu!(
        state.cell_counts,
        state.cell_ids,
        state.n_atoms,
    )

    accumulate!(
        +,
        state.inclusive_counts,
        state.cell_counts,
    )

    offset_threads = 256
    offset_blocks = cld(state.n_cells, offset_threads)

    @cuda threads=offset_threads blocks=offset_blocks inclusive_to_offsets_gpu!(
        state.cell_offsets,
        state.inclusive_counts,
        state.n_cells,
    )

    @cuda threads=n_threads blocks=n_blocks get_cell_particles_gpu!(
        state.cell_particles,
        state.cell_write_counts,
        state.cell_ids,
        state.cell_offsets,
        state.n_atoms,
    )

    @cuda threads=n_threads blocks=n_blocks gather_cell_coordinates_gpu!(
        state.cell_x,
        state.cell_y,
        state.cell_z,
        state.cell_particles,
        state.x,
        state.y,
        state.z,
        state.n_atoms,
    )

    return nothing
end

function query_gpu_cell_list!(state)
    fill!(state.neighbour_counts, Int32(0))

    @cuda threads=CELL_BLOCK_SIZE blocks=state.n_cells get_neighbours_cell_shared_contiguous_gpu!(
            state.neighbour_counts,
            state.neighbours,
            state.cell_counts,
            state.cell_offsets,
            state.cell_particles,
            state.cell_x,
            state.cell_y,
            state.cell_z,
            state.n_atoms,
            state.num_cell_x,
            state.num_cell_y,
            state.num_cell_z,
            state.box_Lx,
            state.box_Ly,
            state.box_Lz,
            state.cutoff2,
            state.max_neighbours,
        )

    return nothing
end

function build_geometric_pair_list!(state)
    state.pair_list === nothing && error(
        "pair buffers were not allocated for this GPU cell-list state",
    )

    maximum_neighbours = maximum(state.neighbour_counts)

    maximum_neighbours <= size(state.neighbours, 1) || error(
        "neighbor capacity exceeded: " *
        "$maximum_neighbours > $(size(state.neighbours, 1))",
    )

    n_threads = 256
    n_blocks = cld(state.n_atoms, n_threads)

    @cuda threads=n_threads blocks=n_blocks count_geometric_half_pairs_gpu!(
        state.pair_counts,
        state.neighbour_counts,
        state.neighbours,
        state.n_atoms,
    )

    accumulate!(
        +,
        state.pair_inclusive_counts,
        state.pair_counts,
    )

    @cuda threads=n_threads blocks=n_blocks inclusive_to_offsets_gpu!(
        state.pair_offsets,
        state.pair_inclusive_counts,
        state.n_atoms,
    )

    @cuda threads=n_threads blocks=n_blocks write_geometric_half_pairs_gpu!(
        state.pair_list,
        state.pair_offsets,
        state.neighbour_counts,
        state.neighbours,
        state.n_atoms,
    )

    n_pairs = Int(
        only(
            Array(
                state.pair_inclusive_counts[
                    state.n_atoms:state.n_atoms
                ],
            ),
        ),
    )

    n_pairs <= state.pair_capacity || error(
        "geometric pair capacity exceeded: " *
        "$n_pairs > $(state.pair_capacity)",
    )

    return Molly.NeighborList(
        n_pairs,
        state.pair_list,
    )
end
