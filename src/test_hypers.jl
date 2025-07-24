#ldoc on
#=
## Testing
=#

@testset "Test NLL gradients" begin
    Zk, y = test_setup2d((x,y) -> x^2 + y)
    s, ℓ = 1e-4, 1.0
    z=log(s)

    gp_SE_nll(ℓ,z) = nll(GPPContext(KernelSE{2}(ℓ), exp(z), Zk, y))
    g = gθz_nll(GPPContext(KernelSE{2}(ℓ), s, Zk, y))

    @test g[1] ≈ diff_fd(ℓ->gp_SE_nll(ℓ,z), ℓ) rtol=1e-6
    @test g[2] ≈ diff_fd(z->gp_SE_nll(ℓ,z), z) rtol=1e-6
end

@testset "Test NLL Hessians" begin
    Zk, y = test_setup2d((x,y) -> x^2 + y)
    s, ℓ = 1e-3, 0.89
    z = log(s)

    testf(ℓ,z) = Hθ_nll(GPPContext(KernelSE{2}(ℓ), exp(z), Zk, y))
    ϕref, gref, Href = testf(ℓ, z)

    @test gref[1] ≈ diff_fd(ℓ->testf(ℓ,z)[1][1], ℓ) rtol=1e-6
    @test gref[2] ≈ diff_fd(z->testf(ℓ,z)[1][1], z) rtol=1e-6
    @test Href[1,1] ≈ diff_fd(ℓ->testf(ℓ,z)[2][1], ℓ) rtol=1e-6
    @test Href[1,2] ≈ diff_fd(ℓ->testf(ℓ,z)[2][2], ℓ) rtol=1e-6
    @test Href[2,2] ≈ diff_fd(z->testf(ℓ,z)[2][2], z) rtol=1e-6
    @test Href[1,2] ≈ Href[2,1]
end

@testset "Test reduced NLL consistency" begin
    Zk, y = test_setup2d((x,y) -> x^2 + y)
    gp = GPPContext(KernelSE{2}(1.0), 0.0, Zk, y)

    # Form scaled kernel Cholesky and weights
    KC = Cholesky(sqrt(getCopt(gp))*getKC(gp).U)
    c = KC\y

    @test nll(KC, c, y) ≈ nllr(gp)
end

@testset "Test reduced NLL gradients" begin
    Zk, y = test_setup2d((x,y) -> x^2 + y)
    s, ℓ = 1e-4, 1.0
    z=log(s)

    gp_SE_nllr(ℓ,z) = nllr(GPPContext(KernelSE{2}(ℓ), exp(z), Zk, y))
    g = gθz_nllr(GPPContext(KernelSE{2}(ℓ), s, Zk, y))
    @test g[1] ≈ diff_fd(ℓ->gp_SE_nllr(ℓ,z), ℓ) rtol=1e-6
    @test g[2] ≈ diff_fd(z->gp_SE_nllr(ℓ,z), z) rtol=1e-6
end

@testset "Test reduced NLL Hessian" begin
    Zk, y = test_setup2d((x,y) -> x^2 + y)
    s, ℓ = 1e-3, 0.89
    z = log(s)

    testf(ℓ,z) = Hθ_nllr(GPPContext(KernelSE{2}(ℓ), exp(z), Zk, y))
    ϕref, gref, Href = testf(ℓ, z)

    @test gref[1] ≈ diff_fd(ℓ->testf(ℓ,z)[1][1], ℓ) rtol=1e-6
    @test gref[2] ≈ diff_fd(z->testf(ℓ,z)[1][1], z) rtol=1e-6
    @test Href[1,1] ≈ diff_fd(ℓ->testf(ℓ,z)[2][1], ℓ) rtol=1e-6
    @test Href[1,2] ≈ diff_fd(ℓ->testf(ℓ,z)[2][2], ℓ) rtol=1e-6
    @test Href[2,2] ≈ diff_fd(z->testf(ℓ,z)[2][2], z) rtol=1e-6
    @test Href[1,2] ≈ Href[2,1]
end

@testset "Test tridiagonal manipulations" begin
    n = 10
    Zk, y = test_setup2d((x,y) -> x^2 + y, n)
    ctx = KernelSE{2}(1.0)
    η = 1e-3

    # Ordinary reduced NLL computation (full and pieces)
    K = kernel(ctx, Zk)
    KC = cholesky(K+η*I)
    c = KC\y
    data1 = c'*y
    logdet1 = sum(log.(diag(KC.U)))
    trinv1 = tr(KC\I)
    ϕ̄1 = nllr(KC, c, y)

    # Reduced NLL computation
    tridiag_reduce!(K)
    tridiag_applyQT!(K, y)
    alpha, beta = tridiag_params(K)
    alpha[:] .+= η
    LAPACK.pttrf!(alpha, beta)
    c = copy(y)
    LAPACK.pttrs!(alpha, beta, c)
    data2 = c'*y
    logdet2 = sum(log.(alpha))/2
    trinv2 = tridiag_tr_invLDLt(alpha, beta)
    ϕ̄2, dϕ̄ = nllrT!(K, y, η, alpha, beta, c)

    @test data1 ≈ data2
    @test logdet1 ≈ logdet2
    @test trinv1 ≈ trinv2
    @test ϕ̄1 ≈ ϕ̄2
end

@testset "Test reduced NLL gradient with tridiag" begin
    Zk, y = test_setup2d((x,y) -> x^2 + y)
    ctx = KernelSE{2}(1.0)
    η = 1e-3
    K = kernel(ctx, Zk)
    tridiag_reduce!(K)
    tridiag_applyQT!(K, y)
    @test nllrT(K, y, η)[2] ≈ diff_fd(η->nllrT(K, y, η)[1], η) rtol=1e-6
end
