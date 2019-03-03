# Spatial calculations

"Vector between two coordinate values, accounting for the bounding box."
function vector1D(c1::Real, c2::Real, box_size::Real)
    if c1 < c2
        return (c2 - c1) < (c1 - c2 + box_size) ? (c2 - c1) : (c2 - c1 - box_size)
    else
        return (c1 - c2) < (c2 - c1 + box_size) ? (c2 - c1) : (c2 - c1 + box_size)
    end
end

"3D vector between two `Coordinates`, accounting for the bounding box."
vector(c1::Coordinates, c2::Coordinates, box_size::Real) = [
        vector1D(c1.x, c2.x, box_size),
        vector1D(c1.y, c2.y, box_size),
        vector1D(c1.z, c2.z, box_size)]

"Square distance between two `Coordinates`, accounting for the bounding box."
sqdist(c1::Coordinates, c2::Coordinates, box_size::Real) =
        vector1D(c1.x, c2.x, box_size) ^ 2 +
        vector1D(c1.y, c2.y, box_size) ^ 2 +
        vector1D(c1.z, c2.z, box_size) ^ 2
