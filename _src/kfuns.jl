#ldoc on
#=
# Kernel functions

A kernel function is a symmetric positive definite function
$k : \Omega \times \Omega \rightarrow \mathbb{R}$.  Here positive
definiteness means that for any set of distinct points $x_1, \ldots,
x_n$ in $\Omega$, the matrix with entries $k(x_i, x_j)$ is positive
definite.  We usually denote the ordered list of points $x_i$ as $X$,
and write $K_{XX}$ for the kernel matrix, i.e. the matrix with entries
$k(x_i, x_j)$.  When the set of points is obvious from context, we
will abuse notation slightly and write $K$ rather than $K_{XX}$ for
the kernel matrix.

## Common kernels

We are generally interested in kernel functions that are stationary
(invariant under translation) and isotropic (invariant under
rotation).  In these cases, we can write the kernel as
$$
  k(x,y) = \phi(\|x-y\|)
$$
where $\phi$ is sometimes called a *radial basis function*.
We often define parametric families of kernels by combining a standard
radial basis function with a length scale parameter $\ell$, i.e.
$$
  k(x,y) = \phi(\|x-y\|/\ell)
$$
In terms of the scaled distance $s = \|x-y\|/\ell$, some common
positive definite radial basis functions include:

| Name                | $\phi(s)$                                | Smoothness |
|---------------------|------------------------------------------|------------|
| Squared exponential | $\exp(-s^2/2)$                           | $C^\infty$ |
| Matern 1/2          | $\exp(-s)$                               | $C^0$      |
| Matern 3/2          | $(1+\sqrt{3} s) \exp(-\sqrt{3} s)$       | $C^1$      |
| Matern 5/2          | $(1+\sqrt{5}s+5s^2/3) \exp(-\sqrt{5} s)$ | $C^2$      |
| Inv quadratic       | $(1+s^2)^{-1}$                           | $C^\infty$ |
| Inv multiquadric    | $(1+s^2)^{-1/2}$                         | $C^\infty$ |
| Rational quadratic  | $(1+s^2)^{-\alpha}$                      | $C^\infty$ |

When someone writes about "the" radial basis function, they are
usually referring to the squared exponential function.  We will
frequently default to the squared exponential function in our
examples.  When we go beyond the squared exponential function, we will
most often use the Matern 5/2, inverse quadratic, or inverse
multiquadric kernel, since we usually want more smoothness than
the Matern 1/2 and Matern 3/2 kernels provide.

In code, these functions are
=#

ϕ_SE(s) = exp(-s^2/2)
ϕ_M1(s) = exp(-s)

ϕ_M3(s) = let t = sqrt(3)*s
    (1+t)*exp(-t)
end

ϕ_M5(s) = let t = sqrt(5)*s
    (1+t*(1+t/3))*exp(-t)
end

ϕ_IQ(s) = 1/(1+s^2)
ϕ_IM(s) = 1/sqrt(1+s^2)
ϕ_RQ(s; α=1.0) = (1+s^2)^-α

#=
For later work, we will need functions for $\phi(s)$, $\phi'(s)/s$,
$\phi'(s)$, and $\phi''(s)$.  It is helpful to pack these all together
in a single function definition.
=#

function Dϕ_SE(s)
    ϕ = exp(-s^2/2)
    dϕ_div = -ϕ
    dϕ = dϕ_div*s
    Hϕ = (-1+s^2)*ϕ
    ϕ, dϕ_div, dϕ, Hϕ
end

function Dϕ_M1(s)
    ϕ = exp(-s)
    ϕ, -ϕ/s, -ϕ, ϕ
end

function Dϕ_M3(s)
    t = √3*s
    ψ = exp(-t)
    ϕ = (1+t)*ψ
    dϕ_div = -3*ψ
    dϕ = dϕ_div*s
    Hϕ = 3*(t-1)*ψ
    ϕ, dϕ_div, dϕ, Hϕ
end

function Dϕ_M5(s)
    t = √5*s
    ψ = exp(-t)
    ϕ = (1+t*(1+t/3))*ψ
    dϕ_div = -5/3*(1+t)*ψ
    dϕ = dϕ_div*s
    Hϕ = -5/3*(1+t*(1-t))*ψ
    ϕ, dϕ_div, dϕ, Hϕ
end

function Dϕ_IQ(s)
    ϕ = 1/(1+s^2)
    dϕ_div = -2*ϕ^2
    dϕ = dϕ_div*s
    Hϕ = 2*ϕ^2*(4*ϕ*s^2-1)
    ϕ, dϕ_div, dϕ, Hϕ
end

function Dϕ_IM(s)
    ϕ = 1/sqrt(1+s^2)
    dϕ_div = -ϕ^3
    dϕ = dϕ_div*s
    Hϕ = ϕ^3*(3*s^2*ϕ^2-1)
    ϕ, dϕ_div, dϕ, Hϕ
end

function Dϕ_RQ(s; α=1.0)
    ϕ = (1+s^2)^-α
    dϕ_div = -α*(1+s^2)^-(α+1)*2
    dϕ = dϕ_div*s
    Hϕ = dϕ_div + α*(α+1)*(1+s^2)^-(α+2)*4*s^2
    ϕ, dϕ_div, dϕ, Hϕ
end

#=
## Distance functions

There are several ways to compute Euclidean distance functions in
Julia.  The most obvious ones (e.g. `norm(x-y)`) involve materializing
an intermediate vector.  Since we will be doing this a lot, we will
write a loopy version that runs somewhat faster.
=#

function dist2(x :: AbstractVector{T}, y :: AbstractVector{T}) where {T}
    s = zero(T)
    for k = 1:length(x)
        dk = x[k]-y[k]
        s += dk*dk
    end
    s
end

dist(x :: AbstractVector{T}, y :: AbstractVector{T}) where {T} =
    sqrt(dist2(x,y))

#=
## Kernel contexts

We define a *kernel context* as "all the stuff you need to work with a
kernel."  This includes the type of the kernel, the dimension of the
space, and any hyperparameters.

=#

abstract type KernelContext end
(ctx :: KernelContext)(args...) = kernel(ctx, args...)

#=
All of the kernels we work with are based on radial basis functions,
and have the form
$$
  k(x,y) = \phi(\|x-y\|/\ell)
$$
where $\ell$ is a length scale parameter.  There may be other
hyperparameters as well.  We therefore define an `RBFKernelContext`
subtype that includes the dimension of the space as a type parameter,
and define a getter function to extract that information.
=#

"""
For an RBFKernelContext{d}, we should define

ϕ(ctx, s) = RBF evaluation at s
Dϕ(ctx, s) = RBF derivatives at s
nhypers(ctx) = Number of tuneable hyperparameters
getθ!(θ, ctx) = Extract the hyperparameters into a vector
updateθ(ctx, θ) = Create a new context with updated hyperparameters

We also have the predefined ndims function to extract {d}.
"""
abstract type RBFKernelContext{d} <: KernelContext end

ndims(::RBFKernelContext{d}) where {d} = d

function getθ(ctx :: KernelContext)
    θ = zeros(nhypers(ctx))
    getθ!(θ, ctx)
    θ
end

#=
## Simple RBF kernels

In most cases, the only hyperparameter we will track in the type is
the length scale hyperparameter.  For these subtypes of
`RBFKernelContext`, we have a fairly boilerplate structure and method
definition that we encode in a macro.
=#

macro rbf_simple_kernel(T, ϕ_rbf, Dϕ_rbf)
    T, ϕ_rbf, Dϕ_rbf= esc(T), esc(ϕ_rbf), esc(Dϕ_rbf)
    quote
        struct $T{d} <: $(esc(:RBFKernelContext)){d}
            ℓ :: Float64
        end
        $(esc(:ϕ))(::$T, s) = $ϕ_rbf(s)
        $(esc(:Dϕ))(::$T, s) = $Dϕ_rbf(s)
        $(esc(:nhypers))(::$T) = 1
        $(esc(:getθ!))(θ, ctx :: $T) = θ[1]=ctx.ℓ
        $(esc(:updateθ))(ctx :: $T{d}, θ) where {d} = $T{d}(θ[1])
    end
end

@rbf_simple_kernel(KernelSE, ϕ_SE, Dϕ_SE)
@rbf_simple_kernel(KernelM1, ϕ_M1, Dϕ_M1)
@rbf_simple_kernel(KernelM3, ϕ_M3, Dϕ_M3)
@rbf_simple_kernel(KernelM5, ϕ_M5, Dϕ_M5)
@rbf_simple_kernel(KernelIQ, ϕ_IQ, Dϕ_IQ)
@rbf_simple_kernel(KernelIM, ϕ_IM, Dϕ_IM)

#=
## Kernel operations

One of the reasons for defining a kernel context is that it allows us
to have a generic high-performance interface for kernel operations.
The most fundamental operation, of course, is evaluating the kernel
on a pair of points.
=#

kernel(ctx :: RBFKernelContext, x :: AbstractVector, y :: AbstractVector) =
    ϕ(ctx, dist(x, y)/ctx.ℓ)

#=
The interface for computing derivatives will involve two functions:
one for computing the gradient with respect to the hypers, the other
for computing the Hessian.  In the case of radial basis functions
where the only intrinsic hyperparameter is the length scale,
we have
$$\begin{aligned}
\nabla_\theta k(x,y) &=
    \begin{bmatrix} -\phi'(s) s/\ell \end{bmatrix} \\
H_{\theta} k(x,y) &=
    \begin{bmatrix} (\phi''(s) s + 2 \phi'(s)) s / \ell^2 \end{bmatrix}
\end{aligned}$$
This gives us the following generic code:
=#

function gθ_kernel!(g :: AbstractVector, ctx :: RBFKernelContext,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    ℓ = ctx.ℓ
    s = dist(x,y)/ℓ
    _, _, dϕ, _ = Dϕ(ctx, s)
    g[1] -= c*dϕ*s/ℓ
    g
end

function Hθ_kernel!(H :: AbstractMatrix, ctx :: RBFKernelContext,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    ℓ = ctx.ℓ
    s = dist(x,y)/ℓ
    _, _, dϕ, Hϕ = Dϕ(ctx, s)
    H[1,1] += c*(Hϕ*s + 2*dϕ)*s/ℓ^2
    H
end

#=
For spatial derivatives of kernels based on radial basis functions, it is
useful to write $r = x-y$ and $\rho = \|r\|$, and to write the
derivative formulae in terms of $r$ and $\rho$.  Using $f_{,i}$ to
denote the $i$th partial derivative of a function $f$, we have first
derivatives
$$
  [\phi(\rho)]_{,i} = \phi'(\rho) \rho_{,i}
$$
and second derivatives
$$
  [\phi(\rho)]_{,ij} = \phi''(\rho) \rho_{,i} \rho_{,j} + \phi'(\rho) \rho_{,ij}
$$
The derivatives of $\rho = \|r\|$ are given by
$$
  \rho_{,i} = \rho^{-1} r_i = u_i
$$
and
$$
  \rho_{,ij} = \rho^{-1} \left( \delta_{ij} - u_i u_j \right)
$$
where $u = r/\rho$ is the unit length vector in the $r$ direction.
Putting these calculations together, we have
$$
  \nabla \phi = \phi'(\rho) u
$$
and
$$
  H \phi =
  \frac{\phi'(\rho)}{\rho} I +
  \left( \phi''(\rho) - \frac{\phi'(\rho)}{\rho} \right) uu^T
$$
This gives us the following generic code.
=#

function gx_kernel!(g :: AbstractVector, ctx :: RBFKernelContext,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    ℓ = ctx.ℓ
    d = ndims(ctx)
    ρ = dist(x,y)
    s = ρ/ℓ
    _, _, dϕ, _ = Dϕ(ctx, s)
    if ρ != 0.0
        dϕ /= ctx.ℓ
        C = c*dϕ/ρ
        for i = 1:d
            g[i] += C*(x[i]-y[i])
        end
    end
    g
end

function Hx_kernel!(H :: AbstractMatrix, ctx :: RBFKernelContext,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    ℓ = ctx.ℓ
    d = ndims(ctx)
    ρ = dist(x,y)
    s = ρ/ℓ
    _, dϕ_div, _, Hϕ = Dϕ(ctx, s)
    Hϕ /= ℓ^2
    dϕ_div /= ℓ^2
    for j = 1:d
        H[j,j] += c*dϕ_div
    end
    if ρ != 0.0
        C = c*(Hϕ-dϕ_div)/ρ^2
        for j = 1:d
            xj, yj = x[j], y[j]
            for i = 1:d
                xi, yi = x[i], y[i]
                H[i,j] += C*(xj-yj)*(xi-yi)
            end
        end
    end
    H
end

#=
## RQ kernel

The rational quadratic case is a little more complicated, with two
adjustable hyperparameters (the length scale $\ell$ and the exponent $\alpha$).
=#

struct KernelRQ{d} <: RBFKernelContext{d}
    ℓ :: Float64
    α :: Float64
end

ϕ(ctx :: KernelRQ, s) = ϕ_RQ(s; α=ctx.α)
Dϕ(ctx :: KernelRQ, s) = Dϕ_RQ(s; α=ctx.α)

nhypers(:: KernelRQ) = 2
function getθ!(θ, ctx :: KernelRQ)
    θ[1] = ℓ
    θ[2] = α
end
updateθ(ctx :: KernelRQ{d}, θ) where {d} = KernelRQ{d}(θ[1], θ[2])

#=
Here we also want to compute the gradients and Hessians with respect
to both $\ell$ and $\alpha$.
=#

function gθ_kernel!(g :: AbstractVector, ctx :: KernelRQ,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    ℓ, α = ctx.ℓ, ctx.α
    s  = dist(x,y)/ℓ
    s2 = s^2
    z  = 1.0 + s2
    ϕ     = z^-α
    gℓ    = 2*α*ϕ/z * s2/ℓ
    gα    = -ϕ * log(z)
    g[1] += c*gℓ
    g[2] += c*gα
    g
end

function Hθ_kernel!(H :: AbstractMatrix, ctx :: KernelRQ,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    ℓ, α = ctx.ℓ, ctx.α
    s  = dist(x,y)/ℓ
    s2 = s^2
    z  = 1.0 + s2
    logz  = log(z)
    ϕ     = z^-α
    Hℓℓ   = 2*ϕ/z*α*s^2/ℓ*( 2*(α+1)/z*s^2/ℓ - 3/ℓ )
    Hℓα   = 2*ϕ/z*(1-α*logz) * s2/ℓ
    Hαα   = ϕ * logz^2
    H[1,1] += c*Hℓℓ
    H[1,2] += c*Hℓα
    H[2,1] += c*Hℓα
    H[2,2] += c*Hαα
    H
end

#=
## Convenience functions

While it is a little more efficient to use a mutating function to
compute kernel gradients and Hessians, it is also convenient to have a
version available to allocate an outupt vector or matrix.  We note
that these convenience functions do not need to be specialized for the
rational quadratic case.
=#

gθ_kernel(ctx :: RBFKernelContext,
          x :: AbstractVector, y :: AbstractVector) =
              gθ_kernel!(zeros(nhypers(ctx)), ctx, x, y)

Hθ_kernel(ctx :: RBFKernelContext,
          x :: AbstractVector, y :: AbstractVector) =
              Hθ_kernel!(zeros(nhypers(ctx), nhypers(ctx)), ctx, x, y)

gx_kernel(ctx :: RBFKernelContext{d},
          x :: AbstractVector, y :: AbstractVector) where {d} =
              gx_kernel!(zeros(d), ctx, x, y)

Hx_kernel(ctx :: RBFKernelContext{d},
          x :: AbstractVector, y :: AbstractVector) where {d} =
              Hx_kernel!(zeros(d,d), ctx, x, y)

#=
## Testing

```{julia}
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

nothing
```
=#
