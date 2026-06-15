# src/dynamic.jl

export add_atom!, remove_atom!

"""
    add_atom!(sys::System, atom::Atom, coord, velocity=nothing)

Dynamically appends a new atom to the system. 
Note: Modifying system size during simulation requires neighbor list invalidation.
"""
function add_atom!(sys::System, atom::Atom, coord, velocity=nothing)
    push!(sys.atoms, atom)
    push!(sys.coords, coord)
    
    # Only pushing if the system is actively tracking velocities
    if !isnothing(sys.velocities) && length(sys.velocities) > 0
        if !isnothing(velocity)
            push!(sys.velocities, velocity)
        else
            push!(sys.velocities, zero(eltype(sys.velocities)))
        end
    end
    
    # Only pushing if the system is actively tracking atom data
    if !isnothing(sys.atoms_data) && length(sys.atoms_data) > 0
        push!(sys.atoms_data, Dict{String, Any}())
    end
    
    return sys
end

"""
    remove_atom!(sys::System, i::Int)

Removes the atom at index `i` from all system arrays using an O(1) swap-and-pop.
Note: Modifying system size during simulation requires neighbor list invalidation.
"""
function remove_atom!(sys::System, i::Int)
    n = length(sys.atoms)
    @boundscheck (1 <= i <= n) || throw(BoundsError(sys.atoms, i))
    
    # Fast O(1) delete: move the last element into the deleted slot, then pop
    sys.atoms[i] = sys.atoms[n]
    pop!(sys.atoms)
    
    sys.coords[i] = sys.coords[n]
    pop!(sys.coords)
    
    # Only popping if the system is actively tracking velocities
    if !isnothing(sys.velocities) && length(sys.velocities) > 0
        sys.velocities[i] = sys.velocities[n]
        pop!(sys.velocities)
    end
    
    # Only popping if the system is actively tracking atom data
    if !isnothing(sys.atoms_data) && length(sys.atoms_data) > 0
        sys.atoms_data[i] = sys.atoms_data[n]
        pop!(sys.atoms_data)
    end
    
    return sys
end
