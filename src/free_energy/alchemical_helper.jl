# Alchemical helper functions for protein design
# Project Molly v0.23.0
# ENZYME version: v"0.13.116"
# Julia version: 1.11

export
    print_interaction,
    kabsch_AA,
    step_logger,
    dict_reverse,
    read_trajectory

function print_interaction(interaction; start_idx=1, end_idx=1)
    field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
    if end_idx==1
        end_idx = length(interaction)
    end
    for (i,ft) in enumerate(zip(field_names[1:end-1]...))
        if i>=start_idx && i<=end_idx
            n_fields = length(ft)
            println("i:",i,"   ",[string(ft[i])*"," for i in 1:n_fields-2]..., string(ft[end]))
            println(ft[end-1])
        end
    end
end

function print_interaction(interaction, indexes; idx_all=false)
    field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
    if idx_all
        for (i,ft) in enumerate(zip(field_names[1:end-1]...))
            if all(x-> x in indexes, ft[1:end-2])
                n_fields = length(ft)
                print("i:",i,"   ",[string(ft[i])*"," for i in 1:n_fields-2]..., string(ft[end]), " -> ")
                println(ft[end-1])
            end
        end
    else
        for (i,ft) in enumerate(zip(field_names[1:end-1]...))
            if any(x-> x in indexes, ft[1:end-2])
                n_fields = length(ft)
                print("i:",i,"   ",[string(ft[i])*"," for i in 1:n_fields-2]..., string(ft[end]), " -> ")
                println(ft[end-1])
            end
        end
    end
end

function get_interaction(interaction, indexes; idx_all=false)
    list = []
    field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
    if idx_all
        for (i,ft) in enumerate(zip(field_names[1:end-1]...))
            if all(x-> x in indexes, ft[1:end-2])
                n_fields = length(ft)
                line = "i:"*string(i)*"   "
                for i in 1:n_fields-2
                    line *= string(ft[i])*","
                end
                for f in ft[end]
                    line *= string(f)
                end
                line *= " -> "
                line *= string(ft[end-1])
                push!(list, line)
            end
        end
    else
        for (i,ft) in enumerate(zip(field_names[1:end-1]...))
            if any(x-> x in indexes, ft[1:end-2])
                n_fields = length(ft)
                line = "i:"*string(i)*"   "
                for i in 1:n_fields-2
                    line *= string(ft[i])*","
                end
                # for f in ft[end]
                #     line *= string(f)
                # end
                line *= " -> "
                line *= string(ft[end-1])
                push!(list, line)
            end
        end
    end
    return list
end

function print_yaml(data; indent=0)
    if typeof(data) == Dict{Any,Any}
        for (key, value) in data
            println(" " ^ indent * "• ", key)
            # Recursively print nested data structures
            print_yaml(value, indent=indent+2)
        end
    else
        println(" " ^ indent * "• ", data)  # Handle non-Dict types
    end
end

function step_logger(sys, buffers, neighbors, step_n::Integer; n_threads::Integer,
                                  current_potential_energy=nothing, kwargs...)
    if isnothing(current_potential_energy)
        energy = potential_energy(sys, neighbors; n_threads=n_threads)
        println("Step "*string(step_n)*" : potential_energy = "*string(energy))
        flush(stdout)
        return energy
    else
        println("Step "*string(step_n)*" : potential_energy = "*string(current_potential_energy))
        flush(stdout)
        return current_potential_energy
    end
end

function read_trajectory(sys, traj_file, units)
    traj = Chemfiles.Trajectory(traj_file)
    size = traj.size()
    total_traj = []
    for i in range(1,traj.size())
        frame = Chemfiles.read_step(traj, i - 1) # Zero-based indexing
        coord_unit = unit(length_type(sys.boundary))
        sys.boundary = boundary_from_chemfiles(Chemfiles.UnitCell(frame), T, coord_unit)

        coords_arr_Å = T.(Chemfiles.positions(frame)) * u"Å"
        if units
            coords_arr = ustrip.(u"nm", coords_arr_Å) # Assume nm
        else
            coords_arr = uconvert.(coord_unit, coords_arr_Å)
        end
        coords = SVector{3}.(eachcol(coords_arr))
        coords .= wrap_coords.(coords, (sys.boundary,))
        push!(total_traj, coords)
    end
    return total_traj
end


function dict_reverse(dic::AbstractDict)
    ks = collect(keys(dic))
    vs = collect(values(dic))
    new_dic = Dict()
    for (k,v) in zip(ks,vs)
        new_dic[v] = k
    end
    return new_dic
end

function kabsch_AA(P,Q)
    P = ustrip(P')
    Q = ustrip(Q')
    cP = mean(P, dims=2)
    cQ = mean(Q, dims=2)
    
    P_centered = P .- cP
    Q_centered = Q .- cQ
    
    cov = P_centered * Q_centered'
    svd_res = svd(ustrip.(cov))
    Ut = transpose(svd_res.U)
    d = sign(det(svd_res.V * Ut))
    dmat = [1 0 0; 0 1 0; 0 0 d]
    rot = svd_res.V * dmat * Ut
    
    t = Q[:,1] - (rot * P)[:,1]
    return rot, t
end

# Rotamer search
"""
    rotate_side_chain(coords, p1_idx, p2_idx, moving_indices, theta)

Rotates a set of atoms in `coords` around an axis defined by atoms at `p1_idx` and `p2_idx`.
- `coords`: Matrix of size (3, N) or (N, 3). This code assumes (N, 3).
- `theta`: Angle in radians.
"""
function rotate_side_chain(coords::Vector{Any}, p1_idx::Int, p2_idx::Int, moving_indices::Vector{Any}, theta::Float64)
    # 1. Define the rotation axis
    p1 = coords[p1_idx, :]
    p2 = coords[p2_idx, :]
    axis = ustrip.(p2 - p1)
    axis /= norm(axis)
    
    # Pre-calculate rotation components
    cos_t = cos(theta)
    sin_t = sin(theta)
    one_minus_cos = 1.0 - cos_t
    
    # 2. Process each moving atom
    for i in moving_indices
        # Translate atom so p1 is at origin
        v = coords[i, :] - p1
        
        # 3. Rodrigues' Rotation Formula
        v_rot = v * cos_t + 
                [cross(axis[1], v[1])] * sin_t + 
                axis * dot(axis, v) * one_minus_cos
        
        # 4. Translate back and update
        coords[i, :] = v_rot + p1
    end
    
    return coords
end

"""
    count_clashes(moving_coords, static_coords; cutoff=2.0)

Returns the number of atom pairs that are closer than the `cutoff` distance.
`moving_coords`: Matrix (M, 3) of the side chain being tested.
`static_coords`: Matrix (N, 3) of the rest of the protein.
"""
function count_clashes(moving_coords::Vector{Any}, static_coords; cutoff::Float64=0.15)
    clash_count = 0
    
    # Iterate through each atom in the side chain
    for i in 1:size(moving_coords, 1)
        m_atom = moving_coords[i, :]
        
        # Compare against every 'static' atom in the environment
        for j in 1:size(static_coords, 1)
            s_atom = static_coords[j, :]
            
            # Squared distance is faster (avoids sqrt)
            dist_sq = sqrt(sum((m_atom .- s_atom)[1].^2))
            
            if ustrip.(dist_sq) < cutoff
                clash_count += 1
            end
        end
    end
    return clash_count
end

# Energy
function myfindneighbors(atoms, coords, boundary, neighborfinder)
     sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        pairwise_inters=(),
        specific_inter_lists=(),
        general_inters=(),
        neighbor_finder=neighborfinder,
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    return find_neighbors(sys)
end

@inline function potential_energy(coords, boundary, neighbors, n_neighbors, velocities, energy_units,
                            atoms, b_g, a_g, p_g, i_g, c_g, cl_g, 
                            lj_g, lg_g, cg_g, pme_g)
    T = typeof(ustrip(zero(eltype(eltype(coords)))))
    pe = zero(T) * energy_units

    # Pairwise interactions
    for ni in 1:n_neighbors
        i, j, special = neighbors[ni]
        dr = vector(coords[i], coords[j], boundary)
        pe += potential_energy(lj_g, dr, atoms[i], atoms[j], energy_units, special,
                        coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
        pe += potential_energy(lg_g, dr, atoms[i], atoms[j], energy_units, special,
                        coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
        pe += potential_energy(cg_g, dr, atoms[i], atoms[j], energy_units, special,
                        coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
    end

    # PME
    pe += potential_energy(pme_g, atoms, coords, boundary, NoUnits)

    # Bonds
    for (i, j, inter) in zip(b_g.is, b_g.js, b_g.inters)
        pe +=  potential_energy(inter, coords[i], coords[j], boundary, atoms[i], atoms[j],
                              energy_units, velocities[i], velocities[j], 0, b_g.data)
    end

    # Angles
    for (i, j, k, inter) in zip(a_g.is, a_g.js, a_g.ks, a_g.inters)
        pe += potential_energy(inter, coords[i], coords[j], coords[k], boundary, atoms[i],
                              atoms[j], atoms[k], energy_units, velocities[i], velocities[j],
                              velocities[k], 0, a_g.data)
    end

    # Periodic Torsions
    for (i, j, k, l, inter) in zip(p_g.is, p_g.js, p_g.ks, p_g.ls,
                                   p_g.inters)
        pe += potential_energy(inter, coords[i], coords[j], coords[k], coords[l], boundary,
                              atoms[i], atoms[j], atoms[k], atoms[l], energy_units,
                              velocities[i], velocities[j], velocities[k], velocities[l],
                              0, p_g.data)
    end

    # Custom Torsions
    for (i, j, k, l, inter) in zip(i_g.is, i_g.js, i_g.ks, i_g.ls,
                                   i_g.inters)
        pe += potential_energy(inter, coords[i], coords[j], coords[k], coords[l], boundary,
                              atoms[i], atoms[j], atoms[k], atoms[l], energy_units,
                              velocities[i], velocities[j], velocities[k], velocities[l],
                              0, i_g.data)
    end

    # CMAP Torsions
    for (i, j, k, l, m, inter) in zip(c_g.is, c_g.js, c_g.ks, c_g.ls,
                                   c_g.ms, c_g.inters)
        pe += potential_energy(inter, coords[i], coords[j], coords[k], coords[l], coords[m], 
                              boundary, atoms[i], atoms[j], atoms[k], atoms[l], atoms[m], energy_units,
                              velocities[i], velocities[j], velocities[k], velocities[l], velocities[m],
                              0, c_g.data)
    end

    # CMAP Torsions lambda
    for (i, j, k, l, m, inter) in zip(cl_g.is, cl_g.js, cl_g.ks, cl_g.ls,
                                   cl_g.ms, cl_g.inters)
        pe += potential_energy(inter, coords[i], coords[j], coords[k], coords[l], coords[m], 
                              boundary, atoms[i], atoms[j], atoms[k], atoms[l], atoms[m], energy_units,
                              velocities[i], velocities[j], velocities[k], velocities[l], velocities[m],
                              0, cl_g.data)
    end
    
    return pe
end

function inject_lambda(sys::System, λ, AT)
    Atoms = []
    for (i,a) in enumerate(sys.atoms)
        if a.alch_role == Molly.InsertRole || a.alch_role == Molly.DeleteRole || a.alch_role == Molly.CoreIRole || a.alch_role == Molly.CoreDRole
            push!(Atoms, Atom(index=a.index, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, λ=λ, alch_role=a.alch_role))
        else
            push!(Atoms, Atom(index=a.index, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, λ=a.λ, alch_role=a.alch_role))
        end
    end
    Atoms = Vector{typeof(Atoms[1])}(Atoms)
    GenerInteraction = []
    for inter in sys.general_inters
        if inter isa PME
            push!(GenerInteraction, PME(inter.dist_cutoff, to_device(Atoms, AT), sys.boundary, grad_safe=inter.grad_safe; 
                                        error_tol=inter.error_tol, fixed_charges=false, scheduler=inter.scheduler),
                        )
        elseif inter isa LJDispersionCorrectionλ
            push!(GenerInteraction, LJDispersionCorrectionλ(Molly.to_device(Atoms, AT), inter.dist_cutoff, Molly.DefaultLambdaScheduler(), 
                            Molly.MinimumMixing(), inter.p_σ, inter.p_ϵ)
                            )
        end
        push!(GenerInteraction, inter)
    end
    general_inters = tuple(GenerInteraction...)
    return System(sys, 
                    atoms=Molly.to_device(Atoms, AT),
                    general_inters=general_inters,)
end