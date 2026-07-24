export add_atom, remove_atom

# ==========================================
# GPU/CPU Memory Bridge
# ==========================================

function _device_deleteat(arr, indices)
    cpu_arr = Array(arr)
    deleteat!(cpu_arr, indices)
    return typeof(arr)(cpu_arr)
end

function _device_push(arr, val)
    cpu_arr = Array(arr)
    push!(cpu_arr, val)
    return typeof(arr)(cpu_arr)
end

# ==========================================
# Out-of-Place Topological Shifting
# ==========================================

function prune_and_shift(list, deleted_idx::Int)
    has_i = hasproperty(list, :is)
    has_j = hasproperty(list, :js)
    has_k = hasproperty(list, :ks)
    has_l = hasproperty(list, :ls)
    
    if !has_i return list end 
    
    cpu_is = has_i ? Array(list.is) : Int[]
    cpu_js = has_j ? Array(list.js) : Int[]
    cpu_ks = has_k ? Array(list.ks) : Int[]
    cpu_ls = has_l ? Array(list.ls) : Int[]
    
    to_delete = Int[]
    for n in 1:length(cpu_is)
        if (has_i && cpu_is[n] == deleted_idx) || 
           (has_j && cpu_js[n] == deleted_idx) || 
           (has_k && cpu_ks[n] == deleted_idx) || 
           (has_l && cpu_ls[n] == deleted_idx)
            push!(to_delete, n)
        end
    end
    
    new_props = Dict{Symbol, Any}()
    for prop in propertynames(list)
        val = getproperty(list, prop)
        
        if prop in (:is, :js, :ks, :ls, :inters, :distances)
            pruned_val = _device_deleteat(val, to_delete)
            if prop in (:is, :js, :ks, :ls)
                pruned_val = pruned_val .- (pruned_val .> deleted_idx)
            end
            new_props[prop] = pruned_val
        else
            new_props[prop] = val
        end
    end
    
    T = typeof(list)
    return T((get(new_props, p, getproperty(list, p)) for p in fieldnames(T))...)
end

# ==========================================
# System Rebuilders
# ==========================================

function add_atom(sys::System, atom::Atom, coord, velocity=nothing)
    new_atoms = _device_push(sys.atoms, atom)
    new_coords = _device_push(sys.coords, coord)
    
    new_velocities = (!isnothing(sys.velocities) && length(sys.velocities) > 0) ? 
                     _device_push(sys.velocities, isnothing(velocity) ? zero(eltype(sys.velocities)) : velocity) : sys.velocities
                     
    new_atoms_data = (!isnothing(sys.atoms_data) && length(sys.atoms_data) > 0) ? 
                     _device_push(sys.atoms_data, Dict{String, Any}()) : sys.atoms_data
                     
    T = typeof(sys)
    return T((
        p == :atoms ? new_atoms :
        p == :coords ? new_coords :
        p == :velocities ? new_velocities :
        p == :atoms_data ? new_atoms_data :
        getproperty(sys, p) for p in fieldnames(T)
    )...)
end

function remove_atom(sys::System, i::Int)
    n = length(sys.atoms)
    @boundscheck (1 <= i <= n) || throw(BoundsError(sys.atoms, i))
    
    new_atoms = _device_deleteat(sys.atoms, i)
    new_coords = _device_deleteat(sys.coords, i)
    
    new_velocities = (!isnothing(sys.velocities) && length(sys.velocities) > 0) ? 
                     _device_deleteat(sys.velocities, i) : sys.velocities
                     
    new_atoms_data = (!isnothing(sys.atoms_data) && length(sys.atoms_data) > 0) ? 
                     _device_deleteat(sys.atoms_data, i) : sys.atoms_data
                     
    new_specific = map(list -> prune_and_shift(list, i), sys.specific_inter_lists)
    new_constraints = map(list -> prune_and_shift(list, i), sys.constraints)
    
    T = typeof(sys)
    return T((
        p == :atoms ? new_atoms :
        p == :coords ? new_coords :
        p == :velocities ? new_velocities :
        p == :atoms_data ? new_atoms_data :
        p == :specific_inter_lists ? new_specific :
        p == :constraints ? new_constraints :
        getproperty(sys, p) for p in fieldnames(T)
    )...)
end