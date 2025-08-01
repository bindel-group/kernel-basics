#ldoc on
#=
## Testing
=#

function fd_check_Dϕ(Dϕ, s; kwargs...)
    ϕ,  dϕ_div,  dϕ,  Hϕ  = Dϕ(s; kwargs...)
    @test dϕ_div*s ≈ dϕ
    @test dϕ ≈ diff_fd(s->Dϕ(s; kwargs...)[1], s) rtol=1e-6
    @test Hϕ ≈ diff_fd(s->Dϕ(s; kwargs...)[3], s) rtol=1e-6
end

@testset "Kernel function derivative checks" begin
    s = 0.89
    @testset "SE" fd_check_Dϕ(Dϕ_SE, s)
    @testset "M1" fd_check_Dϕ(Dϕ_M1, s)
    @testset "M3" fd_check_Dϕ(Dϕ_M3, s)
    @testset "M5" fd_check_Dϕ(Dϕ_M5, s)
    @testset "IQ" fd_check_Dϕ(Dϕ_IQ, s)
    @testset "IM" fd_check_Dϕ(Dϕ_IM, s)
    @testset "RQ" fd_check_Dϕ(Dϕ_RQ, s; α=0.75)
end

@testset "Kernel hyper derivatives" begin
    x, y = [0.1; 0.2], [0.8; 0.8]
    ℓ = 0.2996
    kse(ℓ) = KernelSE{2}(ℓ)
    k_kse(ℓ) = kernel(kse(ℓ),x,y)
    g_kse(ℓ) = gθ_kernel(kse(ℓ),x,y)[1]
    H_kse(ℓ) = Hθ_kernel(kse(ℓ),x,y)[1,1]
    @test g_kse(ℓ) ≈ diff_fd(k_kse,ℓ) rtol=1e-6
    @test H_kse(ℓ) ≈ diff_fd(g_kse,ℓ) rtol=1e-6
end

@testset "RQ kernel hyper derivatives" begin
    x, y = [0.1; 0.2], [0.8; 0.8]
    ℓ, α = 0.2996, 0.8253
    krq(ℓ,α) = KernelRQ{2}(ℓ,α)
    k_krq(ℓ,α) = kernel(krq(ℓ,α),x,y)
    g_krq(ℓ,α) = gθ_kernel(krq(ℓ,α),x,y)
    H_krq(ℓ,α) = Hθ_kernel(krq(ℓ,α),x,y)
    @test g_krq(ℓ,α)[1]   ≈ diff_fd(s->k_krq(ℓ+s,α)) rtol=1e-6
    @test g_krq(ℓ,α)[2]   ≈ diff_fd(s->k_krq(ℓ,α+s)) rtol=1e-6
    @test H_krq(ℓ,α)[:,1] ≈ diff_fd(s->g_krq(ℓ+s,α)) rtol=1e-6
    @test H_krq(ℓ,α)[:,2] ≈ diff_fd(s->g_krq(ℓ,α+s)) rtol=1e-6
end

@testset "Kernel spatial derivatives" begin
    x, y, dx = [0.1; 0.2], [0.8; 0.8], [0.617; 0.779]
    k = KernelM5{2}(0.5)
    k_k(x) = kernel(k,x,y)
    g_k(x) = gx_kernel(k,x,y)
    H_k(x) = Hx_kernel(k,x,y)
    @test g_k(x)'*dx ≈ diff_fd(s->k_k(x+s*dx)) rtol=1e-6
    @test H_k(x)*dx ≈ diff_fd(s->g_k(x+s*dx)) rtol=1e-6
end
