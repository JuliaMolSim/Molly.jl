@testset "MBAR positive infinite reduced potentials" begin
    u = [
        0.0 Inf
        0.0 1.0
        1.0 0.0
        Inf 0.0
    ]
    target_u = [0.0, 1.0, Inf, 0.0]
    windows = [1, 1, 2, 2]
    counts = [2, 2]
    f, log_counts = Molly.iterate_mbar(u, windows, counts)
    weights, target_weights, log_denominators = Molly.mbar_weights(
        u, target_u, f, counts, log_counts)

    @test all(isfinite, weights)
    @test all(isfinite, target_weights)
    @test all(isfinite, log_denominators)
    @test vec(sum(weights; dims=1)) ≈ ones(2)
    @test sum(target_weights) ≈ 1
    @test iszero(weights[1, 2])
    @test iszero(weights[4, 1])
    @test iszero(target_weights[3])

    @test_throws DomainError Molly.mbar_weights(
        [0.0 NaN; 1.0 0.0], target_u[1:2], f, counts, log_counts)
    @test_throws DomainError Molly.mbar_weights(
        [0.0 -Inf; 1.0 0.0], target_u[1:2], f, counts, log_counts)
    @test_throws DomainError Molly.mbar_weights(
        [0.0 Inf; Inf Inf], target_u[1:2], f, counts, log_counts)
    @test_throws DomainError Molly.mbar_weights(
        u, fill(Inf, 4), f, counts, log_counts)
end
