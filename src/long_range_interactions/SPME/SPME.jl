function vol(lat_vecs::Vector{Vector{T}}) where T
    return dot(lat_vecs[1], cross(lat_vecs[2], lat_vecs[3]))
end

function reciprocal_vecs(lat_vecs::Vector{Vector{T}}) where T
    V = vol(lat_vecs)
    m1 = cross(lat_vecs[2], lat_vecs[3])/V
    m2 = cross(lat_vecs[3], lat_vecs[1])/V
    m3 = cross(lat_vecs[1], lat_vecs[2])/V
    return [m1,m2,m3]
end

function reciprocal_vecs_twopi(lat_vecs)
    V = vol(lat_vecs)
    m1 = 2*π*cross(lat_vecs[2], lat_vecs[3])/V
    m2 = 2*π*cross(lat_vecs[3], lat_vecs[1])/V
    m3 = 2*π*cross(lat_vecs[1], lat_vecs[2])/V
    return [m1,m2,m3]
end


struct SPME{T, R, B, E, C, L}
    sys::System
    error_tol::T
    r_cut_dir::R
    β::B
    self_energy::E
    BC::Array{C,3}
    K::SVector{3,Integer}
    spline_order::Integer
    recip_lat::Vector{SVector{3,L}}
end

function SPME(sys, error_tol, r_cut_dir, spline_order)
    
    β = sqrt(-log(2*error_tol))/r_cut_dir
    box_sizes = norm.(lattice_vec(sys))
    K = ceil.(Int, 2*β.*box_sizes./(3*(error_tol ^ 0.2))) 

    recip_lat = reciprocal_vecs(sys.lattice_vec)

    # Calculate self-energy
    self_energy = -(β/sqrt(π))*sum(x -> x*x, charges(sys))
    # Calcualte pre-factors for mesh component of energy/force
    C_type = Complex{float_type(sys)}
    BC = zeros(C_type, K)
    calc_BC!(BC, spme)

    return SPME{typeof(error_tol), typeof(r_cut_dir), typeof(β), typeof(self_energy), C_type, eltype(recip_lat[1])}(
                sys, error_tol, r_cut_dir, β, self_energy, K,
                spline_order, recip_lat)
end

reciprocal_lattice(spme::SPME) = spme.recip_lat
self_energy(spme::SPME) = spme.self_energy
n_mesh(spme::SPME) = spme.K
spline_order(spme::SPME) = spme.spline_order

function run!(spme::SPME)

    #& could write to reuse storage for F
    E_dir, F_dir = particle_particle(spme) # Uses Molly pair-pair
    E_rec, F_rec = particle_mesh(spme)
    E_self = self_energy(spme) #calculated on construction, re-use storage

    #* Need to convert units here, try not to hard code to specific unit system
    #* Probably best to just use Unitful and multiply E by 1/4πϵ₀

    E_SPME = E_dir + E_rec + E_self
    F_SPME = F_dir .+ F_rec

    return E_SPME, F_SPME

end


# #& GPU version of Q
# u2 = [Vector{Float64}(undef, (length(n_mesh(spme)), )) for _ in eachindex(positions)];
# u2 = scaled_fractional_coords!(u2, spme.sys.positions, n_mesh(spme), spme.recip_lat);
# M0, M1, M2, _, _, _ = calc_spline_values(u2, n, N_atoms);
# cuQ = CUDA.zeros(Float32,n_mesh(spme)...);

# cuM0 = CuArray{Float32}(M0);
# cuM1 = CuArray{Float32}(M1);
# cuM2 = CuArray{Float32}(M2);
# n_half = ceil(Int64,n/2);
# cu_u = CuArray{Float32}(reduce(hcat, u2)'); #try transposing
# cuCharges = CuArray{Int32}(spme.sys.atoms.charge);
# BC_cuda = CuArray{Float32}(BC)

# thread_per_block = 64
# N_blocks = ceil(Int64, N_atoms/thread_per_block)


# @cuda threads=thread_per_block blocks=N_blocks interpolate_charge_kernel!(cu_u, cuM0, cuM1, cuM2, cuQ, 
#     n_half,cuCharges, n_mesh(spme)..., n, N_atoms)