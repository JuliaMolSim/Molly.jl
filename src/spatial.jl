# Spatial calculations

export
    vector1D,
    vector,
    adjust_bounds

"""
    vector1D(c1, c2, box_size)

Displacement between two 1D coordinate values, accounting for the bounding box.
The minimum image convention is used, so the displacement is to the closest
version of the coordinate accounting for the periodic boundaries.
"""
function vector1D(c1, c2, box_size)
    if c1 < c2
        return (c2 - c1) < (c1 - c2 + box_size) ? (c2 - c1) : (c2 - c1 - box_size)
    else
        return (c1 - c2) < (c2 - c1 + box_size) ? (c2 - c1) : (c2 - c1 + box_size)
    end
end

"""
    vector(c1, c2, box_size)

Displacement between two coordinate values, accounting for the bounding box.
The minimum image convention is used, so the displacement is to the closest
version of the coordinates accounting for the periodic boundaries.
"""
vector(c1, c2, box_size) = vector1D.(c1, c2, box_size)

@generated function vector(c1::SVector{N}, c2::SVector{N}, box_size) where N
    quote
        Base.Cartesian.@ncall $N SVector{$N} i->vector1D(c1[i], c2[i], box_size)
    end
end

"""
    adjust_bounds(c, box_size)

Ensure a coordinate is within the simulation box and return the coordinate.
"""
adjust_bounds(c, box_size) = c - floor(c / box_size) * box_size

adjust_bounds_vec(v, bs) = adjust_bounds.(v, bs)
