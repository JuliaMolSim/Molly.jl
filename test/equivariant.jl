# Tests for the real spherical-harmonic primitives in src/equivariant/spherical_harmonics.jl (used by
# the native Allegro potential). These run with a bare `using Molly` — no Lux/HDF5/GPU — and validate
# normalization, equivariance, the analytic SH gradient against finite differences, and the exact e3nn
# values (axis order / normalization / l=2 basis) that let trained e3nn/Allegro weights transfer.
# The full bit-for-bit match to the nequip-allegro package is checked in test/ml_potentials.jl.

using Molly
using Molly: SVector, SMatrix  # re-exported from StaticArrays (not a direct test dependency)
using LinearAlgebra
using Test

# Return `v` with component `b` shifted by `dx` (avoids StaticArrays.setindex).
perturb(v::SVector{3}, b, dx) = SVector{3}(ntuple(i -> i == b ? v[i] + dx : v[i], 3))

# Deterministic proper rotation (QR of a fixed pseudo-random matrix; no RNG-seed coupling).
function _rand_rotation(seed)
    A = zeros(3, 3)
    for i in 1:3, j in 1:3
        A[i, j] = mod(sin(seed * 12.9898 + i * 3.17 + j * 7.13) * 43758.5453, 1.0) - 0.5
    end
    R = Matrix(qr(A).Q)
    det(R) < 0 && (R[:, 1] .*= -1)
    return SMatrix{3,3,Float64}(R)
end

_rand_vec(seed) = SVector{3,Float64}(sin(seed * 1.1) + 0.3, cos(seed * 2.3) - 0.2, sin(seed * 0.7) * 0.9 + 0.4)

@testset "Equivariant primitives" begin
    @testset "Real SH component normalization (Σ_m Y²= 2l+1)" begin
        for s in 1:20
            Y = Molly.real_sph_harm(2, _rand_vec(s))
            @test isapprox(Y[1]^2, 1.0; atol=1e-10)
            @test isapprox(sum(Y[i]^2 for i in 2:4), 3.0; atol=1e-9)
            @test isapprox(sum(Y[i]^2 for i in 5:9), 5.0; atol=1e-9)
        end
    end

    @testset "Real SH l=1 equivariance (D¹ = R)" begin
        for s in 1:10
            R = _rand_rotation(s); r = _rand_vec(s + 100)
            Y = Molly.real_sph_harm(1, r); Yr = Molly.real_sph_harm(1, R * r)
            @test isapprox(SVector{3}(Yr[2], Yr[3], Yr[4]), R * SVector{3}(Y[2], Y[3], Y[4]); atol=1e-9)
        end
    end

    @testset "Real SH gradient vs finite differences" begin
        for s in 1:10
            r = _rand_vec(s + 7)
            Y, J = Molly.real_sph_harm_grad(2, r)
            @test isapprox(Y, Molly.real_sph_harm(2, r); atol=1e-10)
            h = 1e-6
            for b in 1:3
                fd = (Molly.real_sph_harm(2, perturb(r, b, h)) .-
                      Molly.real_sph_harm(2, perturb(r, b, -h))) ./ (2h)
                for i in 1:9
                    @test isapprox(J[i, b], fd[i]; atol=1e-5, rtol=1e-4)
                end
            end
        end
    end

    @testset "e3nn convention pin (real SH values)" begin
        # Exact values from e3nn 0.6.0 o3.spherical_harmonics([0,1,2], x, normalize=true,
        # normalization="component"). Pins the axis order / normalization / l=2 basis so trained
        # e3nn/Allegro weights transfer. Regenerate with test/allegro_package_reference.py.
        cases = [
            (SVector(0.3, -0.5, 0.8),
             [1.0, 0.524890659168, -0.87481776528, 1.399708424448, 0.948485717439,
              -0.592803573399, -0.262395732054, -1.580809529064, 1.086806551232]),
            (SVector(0.1, 0.2, -0.4),
             [1.0, 0.377964473009, 0.755928946018, -1.511857892037, -0.737711113563,
              0.368855556782, -0.47915742375, -1.475422227127, 1.383208337931]),
        ]
        for (v, ref) in cases
            @test isapprox(collect(Molly.real_sph_harm(2, v)), ref; atol=1e-9)
        end
    end
end
