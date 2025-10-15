export
    TrajSystem,
    read_frame!

"""
    An abstraction that sits on top of the [`System`](@ref) struct
    that allows to read data from trajectories.
"""
struct TrajSystem{TR, S <: System}
    trajectory::TR
    system::S
end

function TrajSystem(structpath::String, trjpath::String, ffpaths::Vector{String};
                    float = Float64,
                    kwargs...)

    ff = MolecularForceField(float, ffpaths..., units = true)

    sys  = System(structpath, ff; kwargs...)
    traj = Chemfiles.Trajectory(trjpath)
    
    return TrajSystem(traj, sys)
end

function TrajSystem(system::System, trjpath::String;
                    kwargs...)
    
    sys  = System(deepcopy(system); kwargs...)
    traj = Chemfiles.Trajectory(trjpath)
    
    return TrajSystem(traj, sys)
end

"""
    read_frame(trjsystem::TrajSystem, frame_idx::Int)

    Reads a frame from a [`TrajSystem`](@ref) and returns a [`System`](@ref)
    representation of said frame.
"""
function read_frame!(trjsystem::TrajSystem{TR,<:System{D,AT,T}}, frame_idx::Int) where {TR,D,AT,T}
    sys = trjsystem.system

    frame = Chemfiles.read_step(trjsystem.trajectory, frame_idx - 1)

    bmat = T.(Chemfiles.matrix(Chemfiles.UnitCell(frame)) ./ T(10) * unit(eltype(Molly.from_device(sys.coords)[1])))
    sys.boundary = sys.boundary isa CubicBoundary     ? CubicBoundary(SMatrix{D,D}(bmat)) :
                   sys.boundary isa TriclinicBoundary ? TriclinicBoundary(SMatrix{D,D}(bmat)) : sys.boundary

    coords  = T.(Chemfiles.positions(frame))
    coords  = SVector{3}.(eachcol(coords)) ./ T(10) * unit(eltype(Molly.from_device(sys.coords)[1]))
    coords .= wrap_coords.(coords, (sys.boundary,))

    # overwrite device buffer (no new System, no new device alloc)
    sys.coords .= Molly.to_device(coords, AT)

    return sys
end