export interpolate_charge!, interpolate_charge2!, interpolate_charge_kernel!,calc_spline_values

#Assumes Vector of Vectors for r and recip_vectors
function scaled_fractional_coords!(u, r, n_mesh::AbstractVector, recip_vectors)

    Threads.@threads for i in eachindex(r)
        for dim in eachindex(n_mesh)
            u[i][dim] = n_mesh[dim]*dot(recip_vectors[dim], r[i])
        end
    end
    return u
end

function calc_spline_values(u, n, N_atoms)

    #For each atom & dir there are n spline values
    M0 = zeros(N_atoms, n+1)
    dM0 = zeros(N_atoms, n+1)
    M1 = zeros(N_atoms, n+1)
    dM1 = zeros(N_atoms, n+1)
    M2 = zeros(N_atoms, n+1)
    dM2 = zeros(N_atoms, n+1)
    Threads.@threads for i in range(1,N_atoms)
        for c in range(0,n) #&either the zero or n case is not used???
            l0 = round(u[i][1]) - c
            l1 = round(u[i][2]) - c
            l2 = round(u[i][3]) - c
           
            M0[i,c+1] = M(u[i][1] - l0, n)
            dM0[i,c+1] = dMdu(u[i][1] - l0, n)
            M1[i,c+1] = M(u[i][2] - l1, n)
            dM1[i,c+1] = dMdu(u[i][2] - l1, n)
            M2[i,c+1] = M(u[i][3] - l2, n)
            dM2[i,c+1] = dMdu(u[i][3] - l2, n)
        end
    end
    return M0, M1, M2, dM0, dM1, dM2
end


function interpolate_charge!(u, Q, dQdr, spme::SPME{SingleThread})
    K1,K2,K3 = n_mesh(spme)
    recip_lat = reciprocal_lattice(spme)
    q_arr = charges(spme.sys)
    N_atoms = length(q_arr)
    n = spme.spline_order

    u = scaled_fractional_coords!(u, positions(spme.sys), n_mesh(spme), recip_lat)


    for i in 1:N_atoms
        for c0 in 0:n
            l0 = round(Int64,u[i][1]) - c0 # Grid point to interpolate onto

            M0 = M(u[i][1] - l0, n)
            q_n_0 = q_arr[i]*M0 #if 0 <= u_i0 - l0 <= n will be non-zero
            dM0 = dMdu(u[i][1] - l0,n)

            l0 += ceil(Int64,n/2) # Shift
            if l0 < 0 # Apply PBC
                l0 += K1
            elseif l0 >= K1
                l0 -= K1
            end

            for c1 in 0:n
                l1 = round(Int64,u[i][2]) - c1 # Grid point to interpolate onto

                M1 = M(u[i][2] - l1, n)
                q_n_1 = q_n_0*M1 #if 0 <= u_i1 - l1 <= n will be non-zero
                dM1 = dMdu(u[i][2] - l1,n)


                l1 += ceil(Int64,n/2) # Shift
                if l1 < 0 # Apply PBC
                    l1 += K2
                elseif l1 >= K2
                    l1 -= K2
                end
                
                for c2 in 0:n
                    l2 = round(Int64,u[i][3]) - c2 # Grid point to interpolate onto

                    M2 = M(u[i][3] - l2, n)
                    q_n_2 = q_n_1*M2 #if 0 <= u_i1 - l1 <= n will be non-zero
                    dM2 = dMdu(u[i][3] - l2,n)

                    l2 += ceil(Int64,n/2) # Shift
                    if l2 < 0 # Apply PBC
                        l2 += K3
                    elseif l2 >= K3
                        l2 -= K3
                    end

                    Q[l0+1,l1+1,l2+1] += q_arr[i]*M0*M1*M2
                    
                    #*Does it matter that l0,l1,l2 is also a function of r_ia
                    #*This looks like its probably equivalent to some matrix multiply
                    # dQdr[i, 1, l0+1, l1+1, l2+1] = q_arr[i]*(K1*recip_lat[1][1]*dM0*M1*M2 + K2*recip_lat[2][1]*dM1*M0*M2 + K3*recip_lat[3][1]*dM2*M0*M1)
                    # dQdr[i, 2, l0+1, l1+1, l2+1] = q_arr[i]*(K1*recip_lat[1][2]*dM0*M1*M2 + K2*recip_lat[2][2]*dM1*M0*M2 + K3*recip_lat[3][2]*dM2*M0*M1)
                    # dQdr[i, 3, l0+1, l1+1, l2+1] = q_arr[i]*(K1*recip_lat[1][3]*dM0*M1*M2 + K2*recip_lat[2][3]*dM1*M0*M2 + K3*recip_lat[3][3]*dM2*M0*M1)
    
                end
            end
        end
    end
    
    return Q, dQdr
end

function interpolate_charge_kernel!(u, Mx_vals, My_vals, Mz_vals, Q, 
    n_half, charges, K1, K2, K3, n, N_atoms)


    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x


    if i <= N_atoms
        #Load atom data into thread registers
        u_ix = round(Int32,u[i,1])
        u_iy = round(Int32,u[i,2])
        u_iz = round(Int32,u[i,3])
        q = charges[i]
        l0 = 0i32; l1 = 0i32; l2 = 0i32;
        for c0 in 0:n
            l0 = u_ix - c0 # Grid point to interpolate onto


            l0 += n_half # Shift
            if l0 < 0 # Apply PBC
                l0 += K1
            elseif l0 >= K1
                l0 -= K1
            end

            for c1 in 0:n
                l1 = u_iy - c1 # Grid point to interpolate onto

                l1 += n_half # Shift
                if l1 < 0 # Apply PBC
                    l1 += K2
                elseif l1 >= K2
                    l1 -= K2
                end
                
                for c2 in 0:n
                    l2 = u_iz - c2 # Grid point to interpolate onto

                    l2 += n_half # Shift
                    if l2 < 0 # Apply PBC
                        l2 += K3
                    elseif l2 >= K3
                        l2 -= K3
                    end

                    v = q*Mx_vals[i,c0+1]*My_vals[i,c1+1]*Mz_vals[i,c2+1]
                    CUDA.@atomic Q[l0+1,l1+1,l2+1] += v
                end
            end
        end
    end

end
                    