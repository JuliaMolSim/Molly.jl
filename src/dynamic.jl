export add_atom!, remove_atom!

# ==========================================
# Topological Shifting Engine
# ==========================================

function prune_and_shift!(list::InteractionList2Atoms, deleted_idx::Int)
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

function prune_and_shift!(list::InteractionList3Atoms, deleted_idx::Int)
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

function prune_and_shift!(list::InteractionList4Atoms, deleted_idx::Int)
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

# Fallback for unsupported lists
prune_and_shift!(list, deleted_idx::Int) = list

# ==========================================
# Array Mutators
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
    
    # Broadcast shifting engine across all specific interaction tuples
    map(list -> prune_and_shift!(list, i), sys.specific_inter_lists)
    
    return sys
end