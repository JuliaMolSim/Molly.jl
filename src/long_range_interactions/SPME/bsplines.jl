export calc_BC

function M(u, n)
    if n > 2
        return (u/(n-1))*M(u,n-1) + ((n-u)/(n-1))*M(u-1,n-1)
    elseif n == 2
        if u >= 0 && u <= 2
            return 1 - abs(u-1)
        else
            return 0
        end
    else
        println("Shouldn't be here")
    end
end

function dMdu(u,n)
    return M(u, n-1) - M(u-1, n-1)
end

function b(mⱼ,n,Kⱼ)
    if iseven(n) && (2*abs(mⱼ) == Kⱼ) #interp fails in this case
        return 0.0
    end

    m_K = 2*π*mⱼ/Kⱼ
    v = m_K*(n-1)
    num = cos(v) + 1im*sin(v)
    denom = 0.0 + 0.0im
    for k in 0:n-2
        v2 = m_K*k
        denom += M(k+1, n)*(cos(v2) + 1im*sin(v2))
    end
    return  num/denom 
end

function calc_C(β, V, ms, recip_lat)
    m_star = ms[1].*recip_lat[1] .+ ms[2].*recip_lat[2] .+ ms[3].*recip_lat[3]
    m_sq = dot(m_star,m_star)
    return (1/(π*V))*(exp(-(π^2)*m_sq/(β^2))/m_sq)
end
    

function calc_BC(spme::SPME)
    V = vol(spme.sys)
    K1, K2, K3 = n_mesh(spme)
    recip_lat = reciprocal_lattice(spme) #SPME uses the 1 normalized version
    n = spline_order(spme)

    BC = zeros(ComplexF64, K1, K2, K3)
    hs = [0,0,0]

    #Gather h indices 
    h1s = collect(0:(K1-1))
    h1s[h1s .> K1/2] .-= K1 #* this has allocation

    h2s = collect(0:(K2-1))
    h2s[h2s .> K2/2] .-= K2

    h3s = collect(0:(K3-1))
    h3s[h3s .> K3/2] .-= K3

    B1 = abs2.(b.(0:K1-1, n, K1))
    B2 = abs2.(b.(0:K2-1, n, K2))
    B3 = abs2.(b.(0:K3-1, n, K3))

    hs = [0,0,0]
    for m1 in range(0,K1-1)
        hs[1] = h1s[m1+1]
        for m2 in range(0,K2-1)
            hs[2] = h2s[m2+1]
            for m3 in range(0,K3-1) 
                hs[3] = h3s[m3+1]        
                C = calc_C(spme.β, V, hs, recip_lat)
                BC[m1+1,m2+1,m3+1] = B1[m1+1]*B2[m2+1]*B3[m3+1]*C
            end
        end
    end

    BC[1,1,1] = 0

    return BC
end

#equivalent, time to see whats faster
# def M(u, n):
#     return (1/math.factorial(n-1)) * np.sum([((-1)**k)*math.comb(n,k)*np.power(max(u-k, 0), n-1) for k in range(n+1)])
