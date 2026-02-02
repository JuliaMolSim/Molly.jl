# Mock interaction for testing cutoffs in isolation
struct MockInter end
# V(r) = 1/r^6, F(r) = -dV/dr = 6/r^7
Molly.pairwise_pe(::MockInter, r, params) = 1.0 / r^6
Molly.pairwise_force(::MockInter, r, params) = 6.0 / r^7

@testset "Cutoffs" begin
    inter = MockInter()
    params = nothing # Mock interaction doesn't use params

    @testset "NoCutoff" begin
        cutoff = NoCutoff()
        r = 2.0
        @test Molly.pe_cutoff(cutoff, inter, r, params) == 1.0 / r^6
        @test Molly.force_cutoff(cutoff, inter, r, params) == 6.0 / r^7
    end

    @testset "DistanceCutoff" begin
        dist_cut = 3.0
        cutoff = DistanceCutoff(dist_cut)

        # Inside cutoff
        r = 2.0
        @test Molly.pe_cutoff(cutoff, inter, r, params) == 1.0 / r^6
        @test Molly.force_cutoff(cutoff, inter, r, params) == 6.0 / r^7

        # Outside cutoff
        r = 4.0
        @test Molly.pe_cutoff(cutoff, inter, r, params) == 0.0
        @test Molly.force_cutoff(cutoff, inter, r, params) == 0.0
    end

    @testset "ShiftedPotentialCutoff" begin
        dist_cut = 3.0
        cutoff = ShiftedPotentialCutoff(dist_cut)
        V_cut = 1.0 / dist_cut^6

        # Inside cutoff
        r = 2.0
        @test Molly.pe_cutoff(cutoff, inter, r, params) ≈ (1.0 / r^6) - V_cut
        @test Molly.force_cutoff(cutoff, inter, r, params) == 6.0 / r^7

        # Outside cutoff
        r = 4.0
        @test Molly.pe_cutoff(cutoff, inter, r, params) == 0.0
        @test Molly.force_cutoff(cutoff, inter, r, params) == 0.0
    end

    @testset "ShiftedForceCutoff" begin
        dist_cut = 3.0
        cutoff = ShiftedForceCutoff(dist_cut)
        V_cut = 1.0 / dist_cut^6
        F_cut = 6.0 / dist_cut^7

        # Inside cutoff
        r = 2.0
        # V_c(r) = V(r) - (r-r_c)V'(r_c) - V(r_c)
        # V'(r) = -F(r)
        # V_c(r) = V(r) + (r-r_c)F(r_c) - V(r_c)
        @test Molly.pe_cutoff(cutoff, inter, r, params) ≈ (1.0 / r^6) + (r - dist_cut) * F_cut - V_cut
        @test Molly.force_cutoff(cutoff, inter, r, params) ≈ (6.0 / r^7) - F_cut

        # Outside cutoff
        r = 4.0
        @test Molly.pe_cutoff(cutoff, inter, r, params) == 0.0
        @test Molly.force_cutoff(cutoff, inter, r, params) == 0.0
    end

    @testset "CubicSplineCutoff" begin
        dist_act = 2.0
        dist_cut = 3.0
        cutoff = CubicSplineCutoff(dist_act, dist_cut)

        @test_throws ArgumentError CubicSplineCutoff(3.0, 2.0)

        # Before activation
        r = 1.5
        @test Molly.pe_cutoff(cutoff, inter, r, params) == 1.0 / r^6
        @test Molly.force_cutoff(cutoff, inter, r, params) == 6.0 / r^7

        # After cutoff
        r = 3.5
        @test Molly.pe_cutoff(cutoff, inter, r, params) == 0.0
        @test Molly.force_cutoff(cutoff, inter, r, params) == 0.0

        # Between activation and cutoff
        r = 2.5
        V_act = 1.0 / dist_act^6
        F_act = 6.0 / dist_act^7
        dV_dr_act = -F_act
        t = (r - dist_act) / (dist_cut - dist_act)

        V_expected = (2t^3 - 3t^2 + 1) * V_act + (t^3 - 2t^2 + t) * (dist_cut - dist_act) * dV_dr_act
        F_expected = -(6t^2 - 6t) * V_act / (dist_cut - dist_act) - (3t^2 - 4t + 1) * dV_dr_act
        # Wait, in src/cutoffs.jl:
        # return -(6t^2 - 6t) * pe_act / (cutoff.dist_cutoff - cutoff.dist_activation) - (3t^2 - 4t + 1) * dpe_dr_act
        # My F_expected calculation from the code logic:
        # force_apply_cutoff returns exactly what is inside the parens (multiplied by r<=cutoff.dist_cutoff)

        @test Molly.pe_cutoff(cutoff, inter, r, params) ≈ V_expected
        @test Molly.force_cutoff(cutoff, inter, r, params) ≈ F_expected

        # Check continuity at dist_act
        r_eps_minus = dist_act - 1e-9
        r_eps_plus = dist_act + 1e-9
        @test Molly.pe_cutoff(cutoff, inter, r_eps_minus, params) ≈ Molly.pe_cutoff(cutoff, inter, r_eps_plus, params) atol = 1e-8
        @test Molly.force_cutoff(cutoff, inter, r_eps_minus, params) ≈ Molly.force_cutoff(cutoff, inter, r_eps_plus, params) atol = 1e-8

        # Check continuity at dist_cut
        r_eps_minus = dist_cut - 1e-9
        r_eps_plus = dist_cut + 1e-9
        @test Molly.pe_cutoff(cutoff, inter, r_eps_minus, params) ≈ 0.0 atol = 1e-8
        @test Molly.pe_cutoff(cutoff, inter, r_eps_plus, params) == 0.0
        @test Molly.force_cutoff(cutoff, inter, r_eps_minus, params) ≈ 0.0 atol = 1e-8
        @test Molly.force_cutoff(cutoff, inter, r_eps_plus, params) == 0.0
    end

    @testset "Operator +" begin
        c = DistanceCutoff(1.0)
        @test c + c === c
    end
end
