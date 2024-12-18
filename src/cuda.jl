# CUDA.jl kernels
const WARPSIZE = UInt32(32)

macro shfl_multiple_sync(mask, target, width, vars...)
    all_lines = map(vars) do v
        Expr(:(=), v,
            Expr(:call, :shfl_sync,
                mask, v, target, width
            )
        )
    end
    return esc(Expr(:block, all_lines...))
end

CUDA.shfl_recurse(op, x::Quantity) = op(x.val) * unit(x)
CUDA.shfl_recurse(op, x::SVector{1, C}) where C = SVector{1, C}(op(x[1]))
CUDA.shfl_recurse(op, x::SVector{2, C}) where C = SVector{2, C}(op(x[1]), op(x[2]))
CUDA.shfl_recurse(op, x::SVector{3, C}) where C = SVector{3, C}(op(x[1]), op(x[2]), op(x[3]))

function cuda_threads_blocks_pairwise(n_neighbors)
    n_threads_gpu = min(n_neighbors, parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512")))
    n_blocks = cld(n_neighbors, n_threads_gpu)
    return n_threads_gpu, n_blocks
end

function cuda_threads_blocks_specific(n_inters)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_SPECIFIC", "128"))
    n_blocks = cld(n_inters, n_threads_gpu)
    return n_threads_gpu, n_blocks
end

function pairwise_force_gpu!(buffers, sys::System{D, true, T}, pairwise_inters, nbs, step_n) where {D, T}
    if typeof(nbs) == NoNeighborList
        kernel = @cuda launch=false pairwise_force_kernel_nonl!(
			    buffers.fs_mat, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters, step_n,
                Val(D), Val(sys.force_units))
        conf = launch_configuration(kernel.fun)
        threads_basic = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512"))
        nthreads = min(length(sys.atoms), threads_basic, conf.threads)
        nthreads = cld(nthreads, WARPSIZE) * WARPSIZE
        n_blocks_i = cld(length(sys.atoms), WARPSIZE)
        n_blocks_j = cld(length(sys.atoms), nthreads)
        kernel(buffers.fs_mat, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters, step_n, Val(D),
               Val(sys.force_units); threads=nthreads, blocks=(n_blocks_i, n_blocks_j))
	else    
		N = length(sys.coords)
		n_blocks = cld(N, WARPSIZE)
		r_cut = sys.neighbor_finder.dist_cutoff
		if step_n % sys.neighbor_finder.n_steps_reorder == 0 || !sys.neighbor_finder.initialized
            Morton_bits = 4
            w = r_cut - typeof(ustrip(r_cut))(0.1) * unit(r_cut)
            Morton_seq_cpu = sorted_Morton_seq(Array(sys.coords), w, Morton_bits)
			copyto!(buffers.Morton_seq, Morton_seq_cpu)
            fill!(buffers.box_mins, zero(eltype(buffers.box_mins)))
            fill!(buffers.box_maxs, zero(eltype(buffers.box_maxs)))
            CUDA.@sync @cuda blocks=(cld(N, WARPSIZE),) threads=(32,) kernel_min_max!(buffers.Morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, Val(N), sys.boundary)
            buffers = ForcesBuffer(buffers.fs_mat, buffers.box_mins, buffers.box_maxs, buffers.Morton_seq)
			sys.neighbor_finder.initialized = true
        end
		CUDA.@sync @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true force_kernel!(buffers.Morton_seq, buffers.fs_mat, buffers.box_mins, buffers.box_maxs, sys.coords, sys.velocities, sys.atoms, Val(N), r_cut, Val(sys.force_units), pairwise_inters, sys.boundary, step_n, sys.neighbor_finder.special, sys.neighbor_finder.eligible, Val(T))
	end
    return buffers
end


function pairwise_pe_gpu!(pe_vec_nounits, buffers, sys::System{D, true, T}, pairwise_inters, nbs, step_n) where {D, T}
	if typeof(nbs) == NoNeighborList
		n_threads_gpu, n_blocks = cuda_threads_blocks_pairwise(length(nbs))
    	CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks pairwise_pe_kernel!(
                pe_vec_nounits, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters, nbs,
                step_n, Val(sys.energy_units))
	else
		N = length(sys.coords)
		n_blocks = cld(N, WARPSIZE)
		r_cut = sys.neighbor_finder.dist_cutoff
		Morton_bits = 4
		w = r_cut - typeof(ustrip(r_cut))(0.1) * unit(r_cut)
		Morton_seq_cpu = sorted_Morton_seq(Array(sys.coords), w, Morton_bits)
		copyto!(buffers.Morton_seq, Morton_seq_cpu)
		fill!(buffers.box_mins, zero(eltype(buffers.box_mins)))
		fill!(buffers.box_maxs, zero(eltype(buffers.box_maxs)))
		CUDA.@sync @cuda blocks=(cld(N, WARPSIZE),) threads=(32,) kernel_min_max!(buffers.Morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, Val(N), sys.boundary)
		buffers = ForcesBuffer(buffers.fs_mat, buffers.box_mins, buffers.box_maxs, buffers.Morton_seq)
		sys.neighbor_finder.initialized = true
		CUDA.@sync @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true energy_kernel!(buffers.Morton_seq, 
		pe_vec_nounits, buffers.box_mins, buffers.box_maxs, sys.coords, sys.velocities, sys.atoms, Val(N), r_cut, Val(sys.energy_units), pairwise_inters, sys.boundary, step_n, sys.neighbor_finder.special, sys.neighbor_finder.eligible, Val(T))
	end
	return pe_vec_nounits
end


function pairwise_force_kernel_nl!(forces, coords_var, velocities_var, atoms_var, boundary, inters,
    neighbors_var, step_n, ::Val{D}, ::Val{F}) where {D, F}
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    neighbors = CUDA.Const(neighbors_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        f = sum_pairwise_forces(inters, atoms[i], atoms[j], Val(F), special, coords[i], coords[j],
                                boundary, velocities[i], velocities[j], step_n)
        for dim in 1:D
            fval = ustrip(f[dim])
            Atomix.@atomic :monotonic forces[dim, i] += -fval
            Atomix.@atomic :monotonic forces[dim, j] +=  fval
        end
    end
    return nothing
end

function pairwise_pe_kernel!(energy, coords_var, velocities_var, atoms_var, boundary, inters,
                             neighbors_var, step_n, ::Val{E}) where E
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    neighbors = CUDA.Const(neighbors_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        coord_i, coord_j, vel_i, vel_j = coords[i], coords[j], velocities[i], velocities[j]
        dr = vector(coord_i, coord_j, boundary)
        pe = potential_energy_gpu(inters[1], dr, atoms[i], atoms[j], E, special, coord_i, coord_j,
                                  boundary, vel_i, vel_j, step_n)
        for inter in inters[2:end]
            pe += potential_energy_gpu(inter, dr, atoms[i], atoms[j], E, special, coord_i, coord_j,
                                       boundary, vel_i, vel_j, step_n)
        end
		
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
    return nothing
end

# ********************************************* TO BE TESTED ************************************************************
function sorted_Morton_seq(positions, w, bits::Int)
	N = length(positions)
	Morton_sequence = Vector{Int32}(undef, N)
    for i in 1:N
		x, y, z = positions[i][1], positions[i][2], positions[i][3]
        Morton_sequence[i] = Morton_code(floor(Int32, x/w), floor(Int32, y/w), floor(Int32, z/w), bits)
    end
	sort = Int32.(sortperm(Morton_sequence))
    return sort
end


function boxes_dist(x1_min::D, x1_max::D, x2_min::D, x2_max::D, Lx::D) where D

	a = abs(Molly.vector_1D(x2_max, x1_min, Lx))
	b = abs(Molly.vector_1D(x1_max, x2_min, Lx))

	return ifelse(
		x1_min - x2_max <= zero(D) && x2_min - x1_max <= zero(D),
		zero(D),
		ifelse(a < b, a, b)	
	)
end

# Function to compute Morton code (Z-order) from voxel indices (i, j, k)
function Morton_code(i::Int32, j::Int32, k::Int32, bits::Int)
    code = 0
    for bit in 0:(bits-1)
        code |= ((i >> bit) & 1) << (3 * bit)
        code |= ((j >> bit) & 1) << (3 * bit + 1)
        code |= ((k >> bit) & 1) << (3 * bit + 2)
    end
    return Int32(code)
end


function kernel_min_max!(
	sorted_seq,
	mins::AbstractArray{D}, 
	maxs::AbstractArray{D}, 
	coords, 
	::Val{n}, 
	boundary) where {n, D}

	D32 = Int32(32)
	a = Int32(1)
	b = Int32(3)
	r = Int32(n % D32)
    i = threadIdx().x + (blockIdx().x - a) * blockDim().x
    local_i = threadIdx().x
	mins_smem = CuStaticSharedArray(D, (D32, b))
	maxs_smem = CuStaticSharedArray(D, (D32, b))
	r_smem = CuStaticSharedArray(D, (r, b))
 
	if i <= n - r && local_i <= D32
		for xyz in a:b
			mins_smem[local_i, xyz] = coords[sorted_seq[i]][xyz]
			maxs_smem[local_i, xyz] = coords[sorted_seq[i]][xyz]
		end
	end
	if i > n - r && i <= n && local_i <= r
		for xyz in a:b
			r_smem[local_i, xyz] = coords[sorted_seq[i]][xyz]
		end
	end
	
    sync_threads() 

	# Same condition as before
	if i <= n - r && local_i <= D32
		for k in a:Int32(log2(D32))
			for xyz in a:b
				@inbounds begin
					if local_i % Int32(2^k) == Int32(0)
						if mins_smem[local_i, xyz] > mins_smem[local_i - Int32(2^(k - 1)), xyz] 
							mins_smem[local_i, xyz] = mins_smem[local_i - Int32(2^(k - 1)), xyz]
						end
						if maxs_smem[local_i, xyz] < maxs_smem[local_i - Int32(2^(k - 1)), xyz] 
							maxs_smem[local_i, xyz] = maxs_smem[local_i - Int32(2^(k - 1)), xyz]
						end
					end
				end
			end
		end
		
		# Select the last thread (in which the minimum is stored) and fill the vectors of minima and maxima 
		if local_i == D32 
			for k in a:b
				mins[blockIdx().x, k] = mins_smem[local_i, k]
				maxs[blockIdx().x, k] = maxs_smem[local_i, k]
			end
		end

	end 

	# Since the remainder array is low-dimensional, we can do the scan
	xyz_min = CuStaticSharedArray(D, b)
	xyz_max = CuStaticSharedArray(D, b)
	for k in a:b
		xyz_min[k] = 10 * boundary.side_lengths[k] # We use a very large (arbitrary) value
		xyz_max[k] = -10 * boundary.side_lengths[k]
	end

	# Turn off all the threads except one
	if local_i == a
		for j in a:r
			@inbounds begin
				for k in a:b
					if r_smem[j, k] < xyz_min[k] 
						xyz_min[k] = r_smem[j, k]
					end
					if r_smem[j, k] > xyz_max[k] 
						xyz_max[k] = r_smem[j, k]
					end
				end
			end
		end
	
		# Select the last block and complete the minima and maxima vectors 
		if blockIdx().x == Int32(ceil(n/D32)) && r != Int32(0)
			for k in 1:3
				mins[blockIdx().x, k] = xyz_min[k] 
				maxs[blockIdx().x, k] = xyz_max[k]
			end
		end
	end

	return nothing
end


function force_kernel!( 
	sorted_seq,
	forces_nounits, 
	mins::AbstractArray{D}, 
	maxs::AbstractArray{D},
	coords, 
	velocities,
	atoms,
	::Val{N}, 
	r_cut, 
	::Val{force_units},
	inters_tuple,
	boundary,
	step_n,
	special_matrix,
	eligible_matrix,
	::Val{T}) where {N, D, force_units, T}

	# Converted factors
	a = Int32(1)
	b = Int32(3)
	n_blocks = Int32(ceil(N / 32))

    # Get the indices that run on the blocks and threads
    i = blockIdx().x
    j = blockIdx().y
	i_0_tile = (i - 1) * warpsize()
	j_0_tile = (j - 1) * warpsize()
	index_i = i_0_tile + laneid()
	index_j = j_0_tile + laneid()
	
	# Keep track of the interactions between particles in r-block and the others in 32-blocks
	force_smem = CuStaticSharedArray(T, (32, 3))
	opposites_sum = CuStaticSharedArray(T, (32, 3))
	eligible = CuStaticSharedArray(Bool, (32, 32))
	special = CuStaticSharedArray(Bool, (32, 32))
	r = Int32((N - 1) % 32 + 1)
	@inbounds for k in a:b
		force_smem[laneid(), k] = zero(T)
		opposites_sum[laneid(), k] = zero(T)
	end


    # The code is organised in 4 mutually excluding parts (this is the first (1) one)
	if j < n_blocks && i < j
		d_block = zero(D)
		dist_block = zero(D) * zero(D)
		@inbounds for k in a:b	
			d_block = boxes_dist(mins[i, k], maxs[i, k], mins[j, k], maxs[j, k], boundary.side_lengths[k])
			dist_block += d_block * d_block	
		end
		
		# Check on block-block distance
		if dist_block <= r_cut * r_cut

			s_idx_i = sorted_seq[index_i]
			coords_i = coords[s_idx_i] 
			vel_i = velocities[s_idx_i] 
			atoms_i = atoms[s_idx_i]
			d_pb = zero(D)
			dist_pb = zero(D) * zero(D)
			@inbounds for k in a:b	
				d_pb = boxes_dist(coords_i[k], coords_i[k], mins[j, k], maxs[j, k], boundary.side_lengths[k])
				dist_pb += d_pb * d_pb
			end

			Bool_excl = dist_pb <= r_cut * r_cut
			s_idx_j = sorted_seq[index_j]
			coords_j = coords[s_idx_j]
			vel_j = velocities[s_idx_j] 
			shuffle_idx = laneid()
			atoms_j = atoms[s_idx_j]
			atype_j = atoms_j.atom_type
			aindex_j = atoms_j.index
			amass_j = atoms_j.mass
			acharge_j = atoms_j.charge
			aσ_j = atoms_j.σ
			aϵ_j = atoms_j.ϵ

			@inbounds for m in a:warpsize()
				s_idx_j_m = sorted_seq[j_0_tile + m]
				eligible[laneid(), m] = eligible_matrix[s_idx_i, s_idx_j_m]
				special[laneid(), m] = special_matrix[s_idx_i, s_idx_j_m]
			end

			# Shuffle
			for m in a:warpsize()
				sync_warp()
				coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, laneid() + a, warpsize())
				vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, laneid() + a, warpsize())
				s_idx_j = CUDA.shfl_sync(0xFFFFFFFF, s_idx_j, laneid() + a, warpsize())
				shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, laneid() + a, warpsize())
				atype_j = CUDA.shfl_sync(0xFFFFFFFF, atype_j, laneid() + a, warpsize())
				aindex_j = CUDA.shfl_sync(0xFFFFFFFF, aindex_j, laneid() + a, warpsize())
				amass_j = CUDA.shfl_sync(0xFFFFFFFF, amass_j, laneid() + a, warpsize())
				acharge_j = CUDA.shfl_sync(0xFFFFFFFF, acharge_j, laneid() + a, warpsize())
				aσ_j = CUDA.shfl_sync(0xFFFFFFFF, aσ_j, laneid() + a, warpsize())
				aϵ_j = CUDA.shfl_sync(0xFFFFFFFF, aϵ_j, laneid() + a, warpsize())
				
				atoms_j_shuffle = Atom(atype_j, aindex_j, amass_j, acharge_j, aσ_j, aϵ_j)
				dr = vector(coords_j, coords_i, boundary)
				r2 = dr[1]^2 + dr[2]^2 + dr[3]^2
				condition = eligible[laneid(), shuffle_idx] == true && Bool_excl == true && r2 <= r_cut * r_cut
 				
				f = condition ? sum_pairwise_forces(
					inters_tuple,
					atoms_i, atoms_j_shuffle,
					Val(force_units),
					special[laneid(), shuffle_idx],
					coords_i, coords_j,
					boundary,
					vel_i, vel_j,
					step_n) : SVector{3, T}(zero(T), zero(T), zero(T))

				@inbounds for k in a:b
					force_smem[laneid(), k] += ustrip(f[k])
					opposites_sum[shuffle_idx, k] -= ustrip(f[k])
				end
			end
			sync_threads()
			@inbounds for k in a:b
				CUDA.atomic_add!(
					pointer(forces_nounits, s_idx_i * b - (b - k)), 
					-force_smem[laneid(), k]
				) 
				CUDA.atomic_add!(
					pointer(forces_nounits, s_idx_j * b - (b - k)), 
					-opposites_sum[laneid(), k]
				) 
			end
		end
	end


	# part (2)
	if j == n_blocks && i < n_blocks

		d_block = zero(D)
		dist_block = zero(D) * zero(D)
		@inbounds for k in a:b
			d_block = boxes_dist(mins[i, k], maxs[i, k], mins[j, k], maxs[j, k], boundary.side_lengths[k])
			dist_block += d_block * d_block	
		end

		if dist_block <= r_cut * r_cut 

			s_idx_i = sorted_seq[index_i]
			coords_i = coords[s_idx_i]
			vel_i = velocities[s_idx_i]
			atoms_i = atoms[s_idx_i]
			d_pb = zero(D)
			dist_pb = zero(D) * zero(D)			
			@inbounds for k in a:b	
				d_pb = boxes_dist(coords_i[k], coords_i[k], mins[j, k], maxs[j, k], boundary.side_lengths[k])
				dist_pb += d_pb * d_pb
			end
			Bool_excl = dist_pb <= r_cut * r_cut

			@inbounds for m in a:r
				s_idx_j = sorted_seq[j_0_tile + m]
				eligible[laneid(), m] = eligible_matrix[s_idx_i, s_idx_j]
				special[laneid(), m] = special_matrix[s_idx_i, s_idx_j]
			end
			
			# Compute the 32 * r distances
			for m in a:r
				s_idx_j = sorted_seq[j_0_tile + m]
				coords_j = coords[s_idx_j]
				vel_j = velocities[s_idx_j]
				atoms_j = atoms[s_idx_j]
				dr = vector(coords_j, coords_i, boundary)
				r2 = dr[1]^2 + dr[2]^2 + dr[3]^2
				condition = eligible[laneid(), m] == true && Bool_excl == true && r2 <= r_cut * r_cut

				f = condition ? sum_pairwise_forces(
					inters_tuple,
					atoms_i, atoms_j,
					Val(force_units),
					special[laneid(), m],
					coords_i, coords_j,
					boundary,
					vel_i, vel_j,
					step_n) : SVector{3, T}(zero(T), zero(T), zero(T))

				@inbounds for k in a:b
					force_smem[laneid(), k] += ustrip(f[k])
					CUDA.atomic_add!(
						pointer(forces_nounits, s_idx_j * b - (b - k)), 
						ustrip(f[k])
					)
				end
			end

			# Sum contributions of the r-block to the other standard blocks
			@inbounds for k in a:b
				CUDA.atomic_add!(
					pointer(forces_nounits, s_idx_i * b - (b - k)), 
					-force_smem[laneid(), k]
				) 
			end
		end
	end


	# part (3)
	if i == j && i < n_blocks

		s_idx_i = sorted_seq[index_i]
		coords_i = coords[s_idx_i]
		vel_i = velocities[s_idx_i]
		atoms_i = atoms[s_idx_i]

		@inbounds for m in a:warpsize()
			s_idx_j = sorted_seq[j_0_tile + m]
			eligible[laneid(), m] = eligible_matrix[s_idx_i, s_idx_j]
			special[laneid(), m] = special_matrix[s_idx_i, s_idx_j]
		end

		for m in laneid() + a : warpsize()
			s_idx_j = sorted_seq[j_0_tile + m]
			coords_j = coords[s_idx_j]
			vel_j = velocities[s_idx_j]
			atoms_j = atoms[s_idx_j]
			dr = vector(coords_j, coords_i, boundary)
			r2 = dr[1]^2 + dr[2]^2 + dr[3]^2
			condition = eligible[laneid(), m] == true && r2 <= r_cut * r_cut

			f = condition ? sum_pairwise_forces(
				inters_tuple,
				atoms_i, atoms_j,
				Val(force_units),
				special[laneid(), m],
				coords_i, coords_j,
				boundary,
				vel_i, vel_j,
				step_n) : SVector{3, T}(zero(T), zero(T), zero(T))
			
			@inbounds for k in a:b
				force_smem[laneid(), k] += ustrip(f[k])
				opposites_sum[m, k] -= ustrip(f[k])
			end
		end	

		@inbounds for k in a:b

			# In this case i == j, so we can call atomic_add! only once
			CUDA.atomic_add!(
				pointer(forces_nounits, s_idx_i * b - (b - k)), 
				-force_smem[laneid(), k] - opposites_sum[laneid(), k]
			) 
		end
	end


	# part (4)
	if i == n_blocks && j == n_blocks

		if laneid() <= r
			
			s_idx_i = sorted_seq[index_i]
			coords_i = coords[s_idx_i]
			vel_i = velocities[s_idx_i]
			atoms_i = atoms[s_idx_i]

			@inbounds for m in a:r
				s_idx_j = sorted_seq[j_0_tile + m]
				eligible[laneid(), m] = eligible_matrix[s_idx_i, s_idx_j]
				special[laneid(), m] = special_matrix[s_idx_i, s_idx_j]
			end

			for m in laneid() + a : r
				s_idx_j = sorted_seq[j_0_tile + m]
				coords_j = coords[s_idx_j]
				vel_j = velocities[s_idx_j]
				atoms_j = atoms[s_idx_j]
				dr = vector(coords_j, coords_i, boundary)
				r2 = dr[1]^2 + dr[2]^2 + dr[3]^2
				condition = eligible[laneid(), m]== true && r2 <= r_cut * r_cut
				
				f = condition ? sum_pairwise_forces(
					inters_tuple,
					atoms_i, atoms_j,
					Val(force_units),
					special[laneid(), m],
					coords_i, coords_j,
					boundary,
					vel_i, vel_j,
					step_n) : SVector{3, T}(zero(T), zero(T), zero(T))

				@inbounds for k in a:b
					force_smem[laneid(), k] += ustrip(f[k])
					opposites_sum[m, k] -= ustrip(f[k])
				end
			end
			@inbounds for k in a:b
				CUDA.atomic_add!(
					pointer(forces_nounits, s_idx_i * b - (b - k)), 
					-force_smem[laneid(), k] - opposites_sum[laneid(), k]
				) 
			end
		end
	end

    return nothing
end




function energy_kernel!( 
	sorted_seq,
	energy_nounits, 
	mins::AbstractArray{D}, 
	maxs::AbstractArray{D}, 
	coords, 
	velocities,
	atoms,
	::Val{N}, 
	r_cut, 
	::Val{energy_units},
	inters_tuple,
	boundary,
	step_n, 
	special_matrix,
	eligible_matrix,
	::Val{T}) where {N, D, energy_units, T}

	# Converted factors
	a = Int32(1)
	b = Int32(3)
	n_blocks = Int32(ceil(N / 32))
	r = Int32((N - 1) % 32 + 1)

    # Indices and shared memory
    i = blockIdx().x
    j = blockIdx().y
	i_0_tile = (i - 1) * warpsize()
	j_0_tile = (j - 1) * warpsize()
	index_i = i_0_tile + laneid()
	index_j = j_0_tile + laneid()
	E_smem = CuStaticSharedArray(T, 32)
	E_smem[laneid()] = zero(T)
	eligible = CuStaticSharedArray(Bool, (32, 32))
	special = CuStaticSharedArray(Bool, (32, 32))

    # The code is organised in 4 mutually excluding parts (this is the first (1) one)
	if j < n_blocks && i < j
		d_block = zero(D)
		dist_block = zero(D) * zero(D)
		@inbounds for k in a:b	
			d_block = boxes_dist(mins[i, k], maxs[i, k], mins[j, k], maxs[j, k], boundary.side_lengths[k])
			dist_block += d_block * d_block	
		end
		
		# Check on block-block distance
		if dist_block <= r_cut * r_cut

			s_idx_i = sorted_seq[index_i]
			coords_i = coords[s_idx_i] 
			vel_i = velocities[s_idx_i]
			atoms_i = atoms[s_idx_i]
			d_pb = zero(D)
			dist_pb = zero(D) * zero(D)
			@inbounds for k in a:b	
				d_pb = boxes_dist(coords_i[k], coords_i[k], mins[j, k], maxs[j, k], boundary.side_lengths[k])
				dist_pb += d_pb * d_pb
			end

			Bool_excl = dist_pb <= r_cut * r_cut
			s_idx_j = sorted_seq[index_j]
			coords_j = coords[s_idx_j]
			vel_j = velocities[s_idx_j]
			shuffle_idx = laneid()
			atoms_j = atoms[s_idx_j]
			atype_j = atoms_j.atom_type
			aindex_j = atoms_j.index
			amass_j = atoms_j.mass
			acharge_j = atoms_j.charge
			aσ_j = atoms_j.σ
			aϵ_j = atoms_j.ϵ

			@inbounds for m in a:warpsize()
				eligible[laneid(), m] = eligible_matrix[s_idx_i, sorted_seq[j_0_tile + m]]
				special[laneid(), m] = special_matrix[s_idx_i, sorted_seq[j_0_tile + m]]
			end

			# Shuffle
			for m in a:warpsize()
				sync_warp()
				coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, laneid() + a, warpsize())
				vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, laneid() + a, warpsize())
				s_idx_j = CUDA.shfl_sync(0xFFFFFFFF, s_idx_j, laneid() + a, warpsize())
				shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, laneid() + a, warpsize())
				atype_j = CUDA.shfl_sync(0xFFFFFFFF, atype_j, laneid() + a, warpsize())
				aindex_j = CUDA.shfl_sync(0xFFFFFFFF, aindex_j, laneid() + a, warpsize())
				amass_j = CUDA.shfl_sync(0xFFFFFFFF, amass_j, laneid() + a, warpsize())
				acharge_j = CUDA.shfl_sync(0xFFFFFFFF, acharge_j, laneid() + a, warpsize())
				aσ_j = CUDA.shfl_sync(0xFFFFFFFF, aσ_j, laneid() + a, warpsize())
				aϵ_j = CUDA.shfl_sync(0xFFFFFFFF, aϵ_j, laneid() + a, warpsize())
				
				atoms_j_shuffle = Atom(atype_j, aindex_j, amass_j, acharge_j, aσ_j, aϵ_j)
				dr = vector(coords_j, coords_i, boundary)
				r2 = dr[1]^2 + dr[2]^2 + dr[3]^2
				condition = eligible[laneid(), shuffle_idx] == true && Bool_excl == true && r2 <= r_cut * r_cut

				pe = condition ? sum_pairwise_potentials(
					inters_tuple,
					atoms_i, atoms_j_shuffle,
					Val(energy_units),
					special[laneid(), shuffle_idx],
					coords_i, coords_j,
					boundary,
					vel_i, vel_j,
					step_n) : SVector{1, T}(zero(T))

				E_smem[laneid()] += ustrip(pe[1])
			end
		end
	end

	# part (2)
	if j == n_blocks && i < n_blocks

		d_block = zero(D)
		dist_block = zero(D) * zero(D)
		@inbounds for k in a:b
			d_block = boxes_dist(mins[i, k], maxs[i, k], mins[j, k], maxs[j, k], boundary.side_lengths[k])
			dist_block += d_block * d_block	
		end

		if dist_block <= r_cut * r_cut 

			s_idx_i = sorted_seq[index_i]
			coords_i = coords[s_idx_i]
			vel_i = velocities[s_idx_i]
			atoms_i = atoms[s_idx_i]
			d_pb = zero(D)
			dist_pb = zero(D) * zero(D)			
			@inbounds for k in a:b	
				d_pb = boxes_dist(coords_i[k], coords_i[k], mins[j, k], maxs[j, k], boundary.side_lengths[k])
				dist_pb += d_pb * d_pb
			end
			Bool_excl = dist_pb <= r_cut * r_cut

			@inbounds for m in a:r
				s_idx_j = sorted_seq[j_0_tile + m]
				eligible[laneid(), m] = eligible_matrix[s_idx_i, s_idx_j]
				special[laneid(), m] = special_matrix[s_idx_i, s_idx_j]
			end
			
			# Compute the 32 * r distances
			for m in a:r
				s_idx_j = sorted_seq[j_0_tile + m]
				coords_j = coords[s_idx_j]
				vel_j = velocities[s_idx_j]
				atoms_j = atoms[s_idx_j]
				dr = vector(coords_j, coords_i, boundary)
				r2 = dr[1]^2 + dr[2]^2 + dr[3]^2
				condition = eligible[laneid(), m] == true && Bool_excl == true && r2 <= r_cut * r_cut

				pe = condition ? sum_pairwise_potentials(
					inters_tuple,
					atoms_i, atoms_j,
					Val(energy_units),
					special[laneid(), m],
					coords_i, coords_j,
					boundary,
					vel_i, vel_j,
					step_n) : SVector{1, T}(zero(T))

				E_smem[laneid()] += ustrip(pe[1])
			end
		end
	end


	# part (3)
	if i == j && i < n_blocks

		s_idx_i = sorted_seq[index_i]
		coords_i = coords[s_idx_i]
		vel_i = velocities[s_idx_i]
		atoms_i = atoms[s_idx_i]

		@inbounds for m in a:warpsize()
			s_idx_j = sorted_seq[j_0_tile + m]
			eligible[laneid(), m] = eligible_matrix[s_idx_i, s_idx_j]
			special[laneid(), m] = special_matrix[s_idx_i, s_idx_j]
		end

		@inbounds for m in laneid() + a : warpsize()
			s_idx_j = sorted_seq[j_0_tile + m]
			coords_j = coords[s_idx_j]
			vel_j = velocities[s_idx_j]
			atoms_j = atoms[s_idx_j]
			dr = vector(coords_j, coords_i, boundary)
			r2 = dr[1]^2 + dr[2]^2 + dr[3]^2
			condition = eligible[laneid(), m] == true && r2 <= r_cut * r_cut

			pe = condition ? sum_pairwise_potentials(
					inters_tuple,
					atoms_i, atoms_j,
					Val(energy_units),
					special[laneid(), m],
					coords_i, coords_j,
					boundary,
					vel_i, vel_j,
					step_n) : SVector{1, T}(zero(T))

			E_smem[laneid()] += ustrip(pe[1])
		end	
	end


	# part (4)
	if i == n_blocks && j == n_blocks

		if laneid() <= r
			
			s_idx_i = sorted_seq[index_i]
			coords_i = coords[s_idx_i]
			vel_i = velocities[s_idx_i]
			atoms_i = atoms[s_idx_i]

			@inbounds for m in a:r
				s_idx_j = sorted_seq[j_0_tile + m]
				eligible[laneid(), m] = eligible_matrix[s_idx_i, s_idx_j]
				special[laneid(), m] = special_matrix[s_idx_i, s_idx_j]
			end

			@inbounds for m in laneid() + a : r
				s_idx_j = sorted_seq[j_0_tile + m]
				coords_j = coords[s_idx_j]
				vel_j = velocities[s_idx_j]
				atoms_j = atoms[s_idx_j]
				dr = vector(coords_j, coords_i, boundary)
				r2 = dr[1]^2 + dr[2]^2 + dr[3]^2
				condition = eligible[laneid(), m] == true && r2 <= r_cut * r_cut
				
				pe = condition ? sum_pairwise_potentials(
					inters_tuple,
					atoms_i, atoms_j,
					Val(energy_units),
					special[laneid(), m],
					coords_i, coords_j,
					boundary,
					vel_i, vel_j,
					step_n) : SVector{1, T}(zero(T))

				E_smem[laneid()] += ustrip(pe[1])
			end
		end
	end

	if threadIdx().x == 1
		sum = T(0.0)
		for k in 1:warpsize()
			sum += E_smem[k]
		end
		CUDA.atomic_add!(pointer(energy_nounits), sum)
	end
    return nothing
end
# ********************************************* TO BE TESTED (SEE CODE ABOVE) ************************************************************


#=
**The No-neighborlist pairwise force summation kernel**: This kernel calculates all the pairwise forces in the system of
`n_atoms` atoms, this is done by dividing the complete matrix of `n_atoms`×`n_atoms` interactions into small tiles. Most
of the tiles are of size `WARPSIZE`×`WARPSIZE`, but when `n_atoms` is not divisible by `WARPSIZE`, some tiles on the
edges are of a different size are dealt as a separate case. The force summation for the tiles are done in the following
way:
1. `WARPSIZE`×`WARPSIZE` tiles: For such tiles each row is assiged to a different tread in a warp which calculates the
forces for the entire row in `WARPSIZE` steps (or `WARPSIZE - 1` steps for tiles on the diagonal of `n_atoms`×`n_atoms`
matrix of interactions). This is done such that some data can be shuffled from `i+1`'th thread to `i`'th thread in each
subsequent iteration of the force calculation in a row. If `a, b, ...` are different atoms and `1, 2, ...` are order in
which each thread calculates the interatomic forces, then we can represent this scenario as (considering `WARPSIZE=8`):
```
    × | i j k l m n o p
    --------------------
    a | 1 2 3 4 5 6 7 8
    b | 8 1 2 3 4 5 6 7
    c | 7 8 1 2 3 4 5 6
    d | 6 7 8 1 2 3 4 5
    e | 5 6 7 8 1 2 3 4
    f | 4 5 6 7 8 1 2 3
    g | 3 4 5 6 7 8 1 2
    h | 2 3 4 5 6 7 8 1
```

2. Edge tiles when `n_atoms` is not divisible by warpsize: In such cases, it is not possible to shuffle data generally
so there is no need to order calculations for each thread diagonally and it is also a bit more complicated to do so.
That's why the calculations are done in the following order:
```
    × | i j k l m n
    ----------------
    a | 1 2 3 4 5 6
    b | 1 2 3 4 5 6
    c | 1 2 3 4 5 6
    d | 1 2 3 4 5 6
    e | 1 2 3 4 5 6
    f | 1 2 3 4 5 6
    g | 1 2 3 4 5 6
    h | 1 2 3 4 5 6
```
=#
function pairwise_force_kernel_nonl!(forces::AbstractArray{T}, coords_var, velocities_var,
                        atoms_var, boundary, inters, step_n, ::Val{D}, ::Val{F}) where {T, D, F}
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    n_atoms = length(atoms)

    tidx = threadIdx().x
    i_0_tile = (blockIdx().x - 1) * warpsize()
    j_0_block = (blockIdx().y - 1) * blockDim().x
    warpidx = cld(tidx, warpsize())
    j_0_tile = j_0_block + (warpidx - 1) * warpsize()
    i = i_0_tile + laneid()

    forces_shmem = CuStaticSharedArray(T, (3, 1024))
    @inbounds for dim in 1:3
        forces_shmem[dim, tidx] = zero(T)
    end

    if i_0_tile + warpsize() > n_atoms || j_0_tile + warpsize() > n_atoms
        @inbounds if i <= n_atoms
            njs = min(warpsize(), n_atoms - j_0_tile)
            atom_i, coord_i, vel_i = atoms[i], coords[i], velocities[i]
            for del_j in 1:njs
                j = j_0_tile + del_j
                if i != j
                    atom_j, coord_j, vel_j = atoms[j], coords[j], velocities[j]
                    f = sum_pairwise_forces(inters, atom_i, atom_j, Val(F), false, coord_i, coord_j,
                                            boundary, vel_i, vel_j, step_n)
                    for dim in 1:D
                        forces_shmem[dim, tidx] += -ustrip(f[dim])
                    end
                end
            end

            for dim in 1:D
                Atomix.@atomic :monotonic forces[dim, i] += forces_shmem[dim, tidx]
            end
        end
    else
        j = j_0_tile + laneid()
        tilesteps = warpsize()
        if i_0_tile == j_0_tile  # To not compute i-i forces
            j = j_0_tile + laneid() % warpsize() + 1
            tilesteps -= 1
        end

        atom_i, coord_i, vel_i = atoms[i], coords[i], velocities[i]
        coord_j, vel_j = coords[j], velocities[j]
        @inbounds for _ in 1:tilesteps
            sync_warp()
            atom_j = atoms[j]
            f = sum_pairwise_forces(inters, atom_i, atom_j, Val(F), false, coord_i, coord_j,
                                    boundary, vel_i, vel_j, step_n)
            for dim in 1:D
                forces_shmem[dim, tidx] += -ustrip(f[dim])
            end
            @shfl_multiple_sync(FULL_MASK, laneid() + 1, warpsize(), j, coord_j)
        end

        @inbounds for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += forces_shmem[dim, tidx]
        end
    end

    return nothing
end

@inline function sum_pairwise_forces(inters, atom_i, atom_j, ::Val{F}, special, coord_i, coord_j,
                                     boundary, vel_i, vel_j, step_n) where F
    dr = vector(coord_i, coord_j, boundary)
    f_tuple = ntuple(length(inters)) do inter_type_i
        force_gpu(inters[inter_type_i], dr, atom_i, atom_j, F, special, coord_i, coord_j, boundary,
                  vel_i, vel_j, step_n)
    end
    f = sum(f_tuple)
    if unit(f[1]) != F
        # This triggers an error but it isn't printed
        # See https://discourse.julialang.org/t/error-handling-in-cuda-kernels/79692
        #   for how to throw a more meaningful error
        error("wrong force unit returned, was expecting $F but got $(unit(f[1]))")
    end
    return f
end

@inline function sum_pairwise_potentials(inters, atom_i, atom_j, ::Val{E}, special, coord_i, coord_j,
                                     boundary, vel_i, vel_j, step_n) where E
    dr = vector(coord_i, coord_j, boundary)

    pe_tuple = ntuple(length(inters)) do inter_type_i
        SVector(potential_energy_gpu(inters[inter_type_i], dr, atom_i, atom_j, E, special, coord_i, coord_j, boundary,
                  vel_i, vel_j, step_n))
				  # why? 
    end
    pe = sum(pe_tuple)
    if unit(pe[1]) != E
        # This triggers an error but it isn't printed
        # See https://discourse.julialang.org/t/error-handling-in-cuda-kernels/79692
        #   for how to throw a more meaningful error
        error("wrong force unit returned, was expecting $E but got $(unit(pe[1]))")
    end
    return pe
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList1Atoms, coords::AbstractArray{SVector{D, C}},
                            velocities, atoms, boundary, step_n, force_units, ::Val{T}) where {D, C, T}
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_1_atoms_kernel!(fs_mat,
            coords, velocities, atoms, boundary, step_n, inter_list.is, inter_list.inters,
            Val(D), Val(force_units))
    return fs_mat
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                            velocities, atoms, boundary, step_n, force_units, ::Val{T}) where {D, C, T}
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_2_atoms_kernel!(fs_mat,
            coords, velocities, atoms, boundary, step_n, inter_list.is, inter_list.js,
            inter_list.inters, Val(D), Val(force_units))
    return fs_mat
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                            velocities, atoms, boundary, step_n, force_units, ::Val{T}) where {D, C, T}
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_3_atoms_kernel!(fs_mat,
            coords, velocities, atoms, boundary, step_n, inter_list.is, inter_list.js,
            inter_list.ks, inter_list.inters, Val(D), Val(force_units))
    return fs_mat
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                            velocities, atoms, boundary, step_n, force_units, ::Val{T}) where {D, C, T}
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_4_atoms_kernel!(fs_mat,
            coords, velocities, atoms, boundary, step_n, inter_list.is, inter_list.js,
            inter_list.ks, inter_list.ls, inter_list.inters, Val(D), Val(force_units))
    return fs_mat
end

function specific_force_1_atoms_kernel!(forces, coords_var, velocities_var, atoms_var, boundary,
                        step_n, is_var, inters_var, ::Val{D}, ::Val{F}) where {D, F}
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    is = CUDA.Const(is_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i = is[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], boundary, atoms[i], F, velocities[i], step_n)
        if unit(fs.f1[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += ustrip(fs.f1[dim])
        end
    end
    return nothing
end

function specific_force_2_atoms_kernel!(forces, coords_var, velocities_var, atoms_var, boundary,
                        step_n, is_var, js_var, inters_var, ::Val{D}, ::Val{F}) where {D, F}
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], coords[j], boundary, atoms[i], atoms[j], F,
                       velocities[i], velocities[j], step_n)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += ustrip(fs.f1[dim])
            Atomix.@atomic :monotonic forces[dim, j] += ustrip(fs.f2[dim])
        end
    end
    return nothing
end

function specific_force_3_atoms_kernel!(forces, coords_var, velocities_var, atoms_var, boundary,
                        step_n, is_var, js_var, ks_var, inters_var, ::Val{D}, ::Val{F}) where {D, F}
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], coords[j], coords[k], boundary, atoms[i],
                       atoms[j], atoms[k], F, velocities[i], velocities[j], velocities[k], step_n)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += ustrip(fs.f1[dim])
            Atomix.@atomic :monotonic forces[dim, j] += ustrip(fs.f2[dim])
            Atomix.@atomic :monotonic forces[dim, k] += ustrip(fs.f3[dim])
        end
    end
    return nothing
end

function specific_force_4_atoms_kernel!(forces, coords_var, velocities_var, atoms_var, boundary,
                        step_n, is_var, js_var, ks_var, ls_var, inters_var,
                        ::Val{D}, ::Val{F}) where {D, F}
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    ls = CUDA.Const(ls_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], coords[j], coords[k], coords[l], boundary,
                       atoms[i], atoms[j], atoms[k], atoms[l], F, velocities[i], velocities[j],
                       velocities[k], velocities[l], step_n)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F || unit(fs.f4[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += ustrip(fs.f1[dim])
            Atomix.@atomic :monotonic forces[dim, j] += ustrip(fs.f2[dim])
            Atomix.@atomic :monotonic forces[dim, k] += ustrip(fs.f3[dim])
            Atomix.@atomic :monotonic forces[dim, l] += ustrip(fs.f4[dim])
        end
    end
    return nothing
end


function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList1Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_pe_1_atoms_kernel!(
            pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.inters, Val(energy_units))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_pe_2_atoms_kernel!(
            pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.inters, Val(energy_units))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_pe_3_atoms_kernel!(
            pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.inters, Val(energy_units))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_pe_4_atoms_kernel!(
            pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.ls, inter_list.inters, Val(energy_units))
    return pe_vec_nounits
end

function specific_pe_1_atoms_kernel!(energy, coords_var, velocities_var, atoms_var, boundary,
                    step_n, is_var, inters_var, ::Val{E}) where E
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    is = CUDA.Const(is_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i = is[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], boundary, atoms[i], E,
                                  velocities[i], step_n)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
    return nothing
end

function specific_pe_2_atoms_kernel!(energy, coords_var, velocities_var, atoms_var, boundary,
                    step_n, is_var, js_var, inters_var, ::Val{E}) where E
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], boundary, atoms[i],
                                  atoms[j], E, velocities[i], velocities[j], step_n)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
    return nothing
end

function specific_pe_3_atoms_kernel!(energy, coords_var, velocities_var, atoms_var, boundary,
                    step_n, is_var, js_var, ks_var, inters_var, ::Val{E}) where E
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], coords[k], boundary,
                                  atoms[i], atoms[j], atoms[k], E, velocities[i], velocities[j],
                                  velocities[k], step_n)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
    return nothing
end

function specific_pe_4_atoms_kernel!(energy, coords_var, velocities_var, atoms_var, boundary,
                    step_n, is_var, js_var, ks_var, ls_var, inters_var, ::Val{E}) where E
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    ls = CUDA.Const(ls_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], coords[k], coords[l],
                                  boundary, atoms[i], atoms[j], atoms[k], atoms[l], E,
                                  velocities[i], velocities[j], velocities[k], velocities[l],
                                  step_n)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
    return nothing
end
