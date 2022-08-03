# Visualize simulations
# This file is only loaded when GLMakie is imported

using .GLMakie

function visualize(coord_logger,
                    boundary,
                    out_filepath::AbstractString;
                    connections=Tuple{Int, Int}[],
                    connection_frames=[trues(length(connections)) for i in values(coord_logger)],
                    trails::Integer=0,
                    framerate::Integer=30,
                    color=:purple,
                    connection_color=:orange,
                    markersize=0.05,
                    linewidth=2.0,
                    transparency=true,
                    show_boundary::Bool=true,
                    boundary_linewidth=2.0,
                    boundary_color=:black,
                    kwargs...)
    coords_start = first(values(coord_logger))
    dist_unit = unit(first(first(coords_start)))
    dims = n_dimensions(boundary)
    fig = Figure()

    if dims == 3
        PointType = Point3f
        ax = Axis3(fig[1, 1], aspect=:data)
        max_connection_dist = cbrt(box_volume(boundary)) / 2
    elseif dims == 2
        PointType = Point2f
        ax = Axis(fig[1, 1])
        ax.aspect = DataAspect()
        max_connection_dist = sqrt(box_volume(boundary)) / 2
    else
        throw(ArgumentError("Found $dims dimensions but can only visualize 2 or 3 dimensions"))
    end

    positions = Observable(PointType.(ustrip_vec.(coords_start)))
    scatter!(ax, positions; color=color, markersize=markersize, transparency=transparency,
                markerspace=:data, kwargs...)

    if show_boundary
        lines!(
            ax,
            bounding_box_lines(boundary, dist_unit)...;
            color=boundary_color,
            linewidth=boundary_linewidth,
        )
    end

    connection_nodes = []
    for (ci, (i, j)) in enumerate(connections)
        # Don't display connected atoms that are likely connected over the box edge
        if first(connection_frames)[ci] && norm(coords_start[i] - coords_start[j]) < max_connection_dist
            if dims == 3
                push!(connection_nodes, Observable(PointType.(
                        ustrip.([coords_start[i][1], coords_start[j][1]]),
                        ustrip.([coords_start[i][2], coords_start[j][2]]),
                        ustrip.([coords_start[i][3], coords_start[j][3]]))))
            elseif dims == 2
                push!(connection_nodes, Observable(PointType.(
                        ustrip.([coords_start[i][1], coords_start[j][1]]),
                        ustrip.([coords_start[i][2], coords_start[j][2]]))))
            end
        else
            if dims == 3
                push!(connection_nodes, Observable(PointType.([0.0, 0.0], [0.0, 0.0],
                                                        [0.0, 0.0])))
            elseif dims == 2
                push!(connection_nodes, Observable(PointType.([0.0, 0.0], [0.0, 0.0])))
            end
        end
    end
    for (ci, cn) in enumerate(connection_nodes)
        lines!(ax, cn;
                color=isa(connection_color, AbstractArray) ? connection_color[ci] : connection_color,
                linewidth=isa(linewidth, AbstractArray) ? linewidth[ci] : linewidth,
                transparency=transparency)
    end

    trail_positions = []
    for trail_i in 1:trails
        push!(trail_positions, Observable(PointType.(ustrip_vec.(coords_start))))
        col = parse.(Colorant, color)
        alpha = 1 - (trail_i / (trails + 1))
        alpha_col = RGBA.(red.(col), green.(col), blue.(col), alpha)
        scatter!(ax, trail_positions[end]; color=alpha_col,  markersize=markersize,
                    transparency=transparency, markerspace=:data, kwargs...)
    end

    boundary_conv = ustrip.(dist_unit, cubic_bounding_box(boundary))
    xlims!(ax, axis_limits(boundary_conv, coord_logger, 1))
    ylims!(ax, axis_limits(boundary_conv, coord_logger, 2))
    dims == 3 && zlims!(ax, axis_limits(boundary_conv, coord_logger, 3))

    GLMakie.record(fig, out_filepath, eachindex(values(coord_logger)); framerate=framerate) do frame_i
        coords = values(coord_logger)[frame_i]

        for (ci, (i, j)) in enumerate(connections)
            if connection_frames[frame_i][ci] && norm(coords[i] - coords[j]) < max_connection_dist
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

        positions[] = PointType.(ustrip_vec.(coords))
        for (trail_i, trail_position) in enumerate(trail_positions)
            trail_position[] = PointType.(ustrip_vec.(values(coord_logger)[max(frame_i - trail_i, 1)]))
        end
    end
end
