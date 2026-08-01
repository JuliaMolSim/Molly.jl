export add_atom!, remove_atom!

# ==========================================
# Explicit Topological Mutators
# ==========================================
# Instead of reflection, we use Julia's multiple dispatch to handle specific interaction types.

function remove_atom!(list::InteractionList2Atoms, deleted_idx::Int)
    to_delete = Int[]
    for n in 1:length(list.is)
        if list.is[n] == deleted_idx || list.js[n] == deleted_idx
            push!(to_delete, n)
        end
    end
    
    deleteat!(list.is, to_delete)
    deleteat!(list.js, to_delete)
    deleteat!(list.inters, to_delete)
    
    for n in 1:length(list.is)
        list.is[n] > deleted_idx && (list.is[n] -= 1)
        list.js[n] > deleted_idx && (list.js[n] -= 1)
    end
    return list
end

function remove_atom!(list::InteractionList3Atoms, deleted_idx::Int)
    to_delete = Int[]
    for n in 1:length(list.is)
        if list.is[n] == deleted_idx || list.js[n] == deleted_idx || list.ks[n] == deleted_idx
            push!(to_delete, n)
        end
    end
    
    deleteat!(list.is, to_delete)
    deleteat!(list.js, to_delete)
    deleteat!(list.ks, to_delete)
    deleteat!(list.inters, to_delete)
    
    for n in 1:length(list.is)
        list.is[n] > deleted_idx && (list.is[n] -= 1)
        list.js[n] > deleted_idx && (list.js[n] -= 1)
        list.ks[n] > deleted_idx && (list.ks[n] -= 1)
    end
    return list
end

function remove_atom!(list::InteractionList4Atoms, deleted_idx::Int)
    to_delete = Int[]
    for n in 1:length(list.is)
        if list.is[n] == deleted_idx || list.js[n] == deleted_idx || list.ks[n] == deleted_idx || list.ls[n] == deleted_idx
            push!(to_delete, n)
        end
    end
    
    deleteat!(list.is, to_delete)
    deleteat!(list.js, to_delete)
    deleteat!(list.ks, to_delete)
    deleteat!(list.ls, to_delete)
    deleteat!(list.inters, to_delete)
    
    for n in 1:length(list.is)
        list.is[n] > deleted_idx && (list.is[n] -= 1)
        list.js[n] > deleted_idx && (list.js[n] -= 1)
        list.ks[n] > deleted_idx && (list.ks[n] -= 1)
        list.ls[n] > deleted_idx && (list.ls[n] -= 1)
    end
    return list
end

# Fallback: If a list (like a specific constraint or virtual site) does not have an explicit 
# remove_atom! method defined yet, it safely passes through without crashing.
remove_atom!(list, deleted_idx::Int) = list

# ==========================================
# System Mutators
# ==========================================

function add_atom!(sys::System, atom::Atom, coord, velocity=nothing)
    push!(sys.atoms, atom)
    push!(sys.coords, coord)
    
    if !isnothing(sys.velocities) && length(sys.velocities) > 0
        push!(sys.velocities, isnothing(velocity) ? zero(eltype(sys.velocities)) : velocity)
    end
    
    if !isnothing(sys.atoms_data) && length(sys.atoms_data) > 0
        push!(sys.atoms_data, Dict{String, Any}())
    end
    
    return sys
end

function remove_atom!(sys::System, i::Int)
    n = length(sys.atoms)
    @boundscheck (1 <= i <= n) || throw(BoundsError(sys.atoms, i))
    
    # Standard ordered deletion to preserve topology
    deleteat!(sys.atoms, i)
    deleteat!(sys.coords, i)
    
    if !isnothing(sys.velocities) && length(sys.velocities) > 0
        deleteat!(sys.velocities, i)
    end
    
    if !isnothing(sys.atoms_data) && length(sys.atoms_data) > 0
        deleteat!(sys.atoms_data, i)
    end
    
    # Use multiple dispatch to dynamically apply the correct typed function
    foreach(list -> remove_atom!(list, i), sys.specific_inter_lists)
    
    if !isnothing(sys.constraints)
        foreach(list -> remove_atom!(list, i), sys.constraints)
    end
    
    return sys
end