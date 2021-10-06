# Visualize simulations
# This file is only loaded when GLMakie is imported

using .GLMakie

function visualize(coord_logger,
                    box_size,
                    out_filepath::AbstractString;
                    connections=Tuple{Int, Int}[],
                    connection_frames=[trues(length(connections)) for i in coord_logger.coords],
                    trails::Integer=0,
                    framerate::Integer=30,
                    color=:purple,
                    connection_color=:orange,
                    markersize=20.0,
                    linewidth=2.0,
                    transparency=true,
                    kwargs...)
    coords_start = first(coord_logger.coords)
    dims = length(first(coords_start))
    if dims == 3
        PointType = Point3f
    elseif dims == 2
        PointType = Point2f
    else
        throw(ArgumentError("Found $dims dimensions but can only visualize 2 or 3 dimensions"))
    end

    scene = Scene()
    positions = Node(PointType.(ustripvec.(coords_start)))
    scatter!(scene, positions; color=color, markersize=markersize,
                transparency=transparency, kwargs...)

    connection_nodes = []
    for (ci, (i, j)) in enumerate(connections)
        if first(connection_frames)[ci] && norm(coords_start[i] - coords_start[j]) < (first(box_size) / 2)
            if dims == 3
                push!(connection_nodes, Node(PointType.(
                        ustrip.([coords_start[i][1], coords_start[j][1]]),
                        ustrip.([coords_start[i][2], coords_start[j][2]]),
                        ustrip.([coords_start[i][3], coords_start[j][3]]))))
            elseif dims == 2
                push!(connection_nodes, Node(PointType.(
                        ustrip.([coords_start[i][1], coords_start[j][1]]),
                        ustrip.([coords_start[i][2], coords_start[j][2]]))))
            end
        else
            if dims == 3
                push!(connection_nodes, Node(PointType.([0.0, 0.0], [0.0, 0.0],
                                                        [0.0, 0.0])))
            elseif dims == 2
                push!(connection_nodes, Node(PointType.([0.0, 0.0], [0.0, 0.0])))
            end
        end
    end
    for (ci, cn) in enumerate(connection_nodes)
        lines!(scene, cn,
                color=isa(connection_color, Array) ? connection_color[ci] : connection_color,
                linewidth=isa(linewidth, Array) ? linewidth[ci] : linewidth,
                transparency=transparency)
    end

    trail_positions = []
    for trail_i in 1:trails
        push!(trail_positions, Node(PointType.(ustripvec.(coords_start))))
        col = parse.(Colorant, color)
        alpha = 1 - (trail_i / (trails + 1))
        alpha_col = RGBA.(red.(col), green.(col), blue.(col), alpha)
        scatter!(scene, trail_positions[end]; color=alpha_col,
                    markersize=markersize, transparency=transparency, kwargs...)
    end

    dist_unit = unit(first(first(coords_start)))
    box_size_conv = ustrip.(dist_unit, box_size)
    xlims!(scene, 0.0, box_size_conv[1])
    ylims!(scene, 0.0, box_size_conv[2])
    dims == 3 && zlims!(scene, 0.0, box_size_conv[3])

    GLMakie.record(scene, out_filepath, eachindex(coord_logger.coords); framerate=framerate) do frame_i
        coords = coord_logger.coords[frame_i]

        for (ci, (i, j)) in enumerate(connections)
            if connection_frames[frame_i][ci] && norm(coords[i] - coords[j]) < (first(box_size) / 2)
                if dims == 3
                    connection_nodes[ci][] = PointType.(
                                ustrip.([coords[i][1], coords[j][1]]),
                                ustrip.([coords[i][2], coords[j][2]]),
                                ustrip.([coords[i][3], coords[j][3]]))
                elseif dims == 2
                    connection_nodes[ci][] = PointType.(
                                ustrip.([coords[i][1], coords[j][1]]),
                                ustrip.([coords[i][2], coords[j][2]]))
                end
            else
                if dims == 3
                    connection_nodes[ci][] = PointType.([0.0, 0.0], [0.0, 0.0],
                                                        [0.0, 0.0])
                elseif dims == 2
                    connection_nodes[ci][] = PointType.([0.0, 0.0], [0.0, 0.0])
                end
            end
        end

        positions[] = PointType.(ustripvec.(coords))
        for (trail_i, trail_position) in enumerate(trail_positions)
            trail_position[] = PointType.(ustripvec.(coord_logger.coords[max(frame_i - trail_i, 1)]))
        end
    end
end
