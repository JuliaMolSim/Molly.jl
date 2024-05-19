

# err_tol = 1e-5
# n = 6
# spme = SPME(sys, SingleThread(), err_tol, r_cut_real, n);
# BC = calc_BC(spme); #* ONLY NEED TO DO ONCE I THINK

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


# Q_inv = fft(cuQ)
# Q_inv .*= BC_cuda
# Q_inv = ifft!(Q_inv) # Q_conv_theta, but do in place

# A = 332.0637132991921
# E = 0.5 * A* sum(real(Q_inv) .* real(cuQ))
