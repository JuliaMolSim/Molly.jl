export
    EnsembleSystem,
    read_frame!

"""
    EnsembleSystem(coordinate_file, trajectory_file, force_field; <keyword arguments>)
    EnsembleSystem(system, trajectory_file)

An object allowing data to be read from a trajectory or ensemble
associated with a [`System`](@ref).

The keyword arguments are the same as [`System`](@ref) setup from a file.
In the case of passing a [`System`](@ref) directly, a copy of the system is made.
"""
struct EnsembleSystem{S, TR}
    sys::S
    trajectory::TR
end

function EnsembleSystem(coord_file::AbstractString, traj_file::AbstractString,
                        force_field::MolecularForceField; kwargs...)
    sys = System(coord_file, force_field; kwargs...)
    traj = Chemfiles.Trajectory(traj_file)
    return EnsembleSystem(sys, traj)
end

function EnsembleSystem(sys::System, traj_file::AbstractString; kwargs...)
    sys_cp = System(deepcopy(sys); kwargs...)
    traj = Chemfiles.Trajectory(traj_file)
    return EnsembleSystem(sys_cp, traj)
end

"""
    read_frame!(ens_sys::EnsembleSystem, frame_idx::Integer)

Read a frame from an [`EnsembleSystem`](@ref) and return a [`System`](@ref)
representing the frame.
"""
function read_frame!(ens_sys::EnsembleSystem{<:System{D, AT, T}},
                     frame_idx::Integer) where {D, AT, T}
    sys = ens_sys.sys
    frame = Chemfiles.read_step(ens_sys.trajectory, frame_idx - 1) # Zero-based indexing
    coord_unit = unit(length_type(sys.boundary))
    sys.boundary = boundary_from_chemfiles(Chemfiles.UnitCell(frame), T, coord_unit)

    coords_arr_Å = T.(Chemfiles.positions(frame)) * u"Å"
    if coord_unit == NoUnits
        coords_arr = ustrip.(u"nm", coords_arr_Å) # Assume nm
    else
        coords_arr = uconvert.(coord_unit, coords_arr_Å)
    end
    coords = SVector{3}.(eachcol(coords_arr))
    coords .= wrap_coords.(coords, (sys.boundary,))

    sys.coords .= to_device(coords, AT) # Overwrite coordinates
    return sys
end
