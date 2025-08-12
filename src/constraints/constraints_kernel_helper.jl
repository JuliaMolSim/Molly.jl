


# No-op when backends are same
to_backend(arr, old::T, new::T) where {T <: KA.Backend} = arr

# Allocates and copies when backends are different
function to_backend(arr, old::A, new::B) where {A <: KA.Backend, B <: KA.Backend}
    out = allocate(new, eltype(arr), size(arr))
    copy!(out, arr)
    return out
end


# These types will enable coalesced memory access
# via StructArray{ConstraintKernelData}
abstract type ConstraintKernelData{D, N, M} end

n_constraints(::ConstraintKernelData{D,N,M}) where {D,N,M} = N
n_atoms_cluster(::ConstraintKernelData{D,N,M}) where {D,N,M} = M

struct NoClusterData <: ConstraintKernelData{Nothing, 0, 0} end

# This is effectivelly just a distance constraint
struct Cluster12Data{D} <: ConstraintKernelData{D, 1, 2}
    k1::Int32
    k2::Int32
    dist12::D
end

interactions(kd::Cluster12Data) = ((kd.k1, kd.k2, kd.dist12), )

function ConstraintKernelData(k1::Int32, k2::Int32, dist12::D) where D
    return Cluster12Data{D}(k1, k2, dist12)
end

struct Cluster23Data{D} <: ConstraintKernelData{D, 2, 3}
    k1::Int32
    k2::Int32
    k3::Int32
    dist12::D
    dist13::D
end

interactions(kd::Cluster23Data) = ((kd.k1, kd.k2, kd.dist12), (kd.k1, kd.k3, kd.dist13))

function ConstraintKernelData(k1::Int32, k2::Int32, k3::Int32, dist12::D, dist13::D) where D
    return Cluster23Data{D}(k1, k2, k3, dist12, dist13)
end

struct Cluster34Data{D} <: ConstraintKernelData{D, 3, 4}
    k1::Int32
    k2::Int32
    k3::Int32
    k4::Int32
    dist12::D
    dist13::D
    dist14::D
end

interactions(kd::Cluster34Data) = ((kd.k1, kd.k2, kd.dist12), (kd.k1, kd.k3, kd.dist13), (kd.k1, kd.k4, kd.dist14))

function ConstraintKernelData(k1::Int32, k2::Int32, k3::Int32, k4::Int32, dist12::D, dist13::D, dist14::D) where D
    return Cluster34Data{D}(k1, k2, k3, k4, dist12, dist13, dist14)
end

struct AngleClusterData{D} <: ConstraintKernelData{D, 3, 3}
    k1::Int32
    k2::Int32
    k3::Int32
    dist12::D
    dist13::D
    dist23::D
end

interactions(kd::AngleClusterData) = ((kd.k1, kd.k2, kd.dist12), (kd.k1, kd.k3, kd.dist13), (kd.k2, kd.k3, kd.dist23))

central_atom(kd::K) where {K <: ConstraintKernelData} = kd.k1
float_type(::ConstraintKernelData{D}) where D = float_type(D)