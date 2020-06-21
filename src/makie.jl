# Visualize simulations
# This file is only loaded when Makie is imported

using .Makie

function visualize(coord_logger,
                    box_size,
                    out_filepath::AbstractString;
                    connections=Tuple{Int, Int}[],
                    trails::Integer=0,
                    framerate::Integer=30,
                    color=:purple,
                    connection_color=:orange,
                    markersize=0.1,
                    linewidth=2.0,
                    transparency=true,
                    kwargs...)
    coords_start = first(coord_logger.coords)
    dims = length(first(coords_start))
    if dims == 3
        PointType = Point3f0
    elseif dims == 2
        PointType = Point2f0
    else
        throw(ArgumentError("Found $dims dimensions but can only visualize 2 or 3 dimensions"))
    end

    scene = Scene()
    connection_nodes = []
    for (i, j) in connections
        if norm(coords_start[i] - coords_start[j]) < (box_size / 2)
            if dims == 3
                push!(connection_nodes, Node(PointType.(
                        [coords_start[i][1], coords_start[j][1]],
                        [coords_start[i][2], coords_start[j][2]],
                        [coords_start[i][3], coords_start[j][3]])))
            elseif dims == 2
                push!(connection_nodes, Node(PointType.(
                        [coords_start[i][1], coords_start[j][1]],
                        [coords_start[i][2], coords_start[j][2]])))
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

    positions = Node(PointType.(coords_start))
    scatter!(scene, positions; color=color, markersize=markersize,
                transparency=transparency, kwargs...)

    trail_positions = []
    for trail_i in 1:trails
        push!(trail_positions, Node(PointType.(coords_start)))
        col = parse.(Colorant, color)
        alpha = 1 - (trail_i / (trails + 1))
        alpha_col = RGBA.(red.(col), green.(col), blue.(col), alpha)
        scatter!(scene, trail_positions[end]; color=alpha_col,
                    markersize=markersize, transparency=transparency, kwargs...)
    end

    xlims!(scene, 0.0, box_size)
    ylims!(scene, 0.0, box_size)
    zlims!(scene, 0.0, box_size)

    record(scene, out_filepath, eachindex(coord_logger.coords); framerate=framerate) do frame_i
        coords = coord_logger.coords[frame_i]

        for (ci, (i, j)) in enumerate(connections)
            if norm(coords[i] - coords[j]) < (box_size / 2)
                if dims == 3
                    connection_nodes[ci][] = PointType.(
                                [coords[i][1], coords[j][1]],
                                [coords[i][2], coords[j][2]],
                                [coords[i][3], coords[j][3]])
                elseif dims == 2
                    connection_nodes[ci][] = PointType.(
                                [coords[i][1], coords[j][1]],
                                [coords[i][2], coords[j][2]])
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

        positions[] = PointType.(coords)
        for (trail_i, trail_position) in enumerate(trail_positions)
            trail_position[] = PointType.(coord_logger.coords[max(frame_i - trail_i, 1)])
        end
    end
end
