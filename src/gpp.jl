#ldoc on
#=
# GPP context

As with kernels, it is helpful to define a context object that
includes the fundamental data needed for working with a GP posterior (GPP).
This includes the kernel information, data points, kernel matrix,
function values, weights, and some scratch space useful for computing
the posterior variance and its derivatives.  We will typically
allocate some extra space so that we can change the number of points
without reallocating all our storage.
=#

struct GPPContext{T <: KernelContext}
    ctx :: T
    η :: Float64
    Xstore  :: Matrix{Float64}
    Kstore  :: Matrix{Float64}
    cstore  :: Vector{Float64}
    ystore  :: Vector{Float64}
    scratch :: Matrix{Float64}
    n :: Integer
end

getX(gp :: GPPContext) = view(gp.Xstore,:,1:gp.n)
getc(gp :: GPPContext) = view(gp.cstore,1:gp.n)
gety(gp :: GPPContext) = view(gp.ystore,1:gp.n)
getK(gp :: GPPContext) = view(gp.Kstore,1:gp.n,1:gp.n)
getKC(gp :: GPPContext) = Cholesky(UpperTriangular(getK(gp)))
capacity(gp :: GPPContext) = length(gp.ystore)
getXrest(gp :: GPPContext) = view(gp.Xstore,:,gp.n+1:capacity(gp))
getyrest(gp :: GPPContext) = view(gp.ystore,gp.n+1:capacity(gp))
getXrest(gp :: GPPContext, m) = view(gp.Xstore,:,gp.n+1:gp.n+m)
getyrest(gp :: GPPContext, m) = view(gp.ystore,gp.n+1:gp.n+m)

function GPPContext(ctx :: KernelContext, η :: Float64, capacity)
    d = ndims(ctx)
    Xstore  = zeros(d, capacity)
    Kstore  = zeros(capacity, capacity)
    cstore  = zeros(capacity)
    ystore  = zeros(capacity)
    scratch = zeros(capacity,max(d+1,3))
    GPPContext(ctx, η, Xstore, Kstore, cstore, ystore, scratch, 0)
end

#=
For internal use, we want to be able to regularly refactor the current
kernel matrix and resolve the coefficient problem.
=#

refactor!(gp :: GPPContext) = kernel_cholesky!(getK(gp), gp.ctx, getX(gp), gp.η)
resolve!(gp :: GPPContext) = ldiv!(getKC(gp), copyto!(getc(gp), gety(gp)))

#=
The basic operations we need are to add or remove data points,
evaluate the predictive mean and variance (and gradients), and update
the kernel hyperparameters.

## Adding points

We start with adding data points, which is maybe the most complicated
operation (since it involves extending a Cholesky factorization).
=#

function add_points!(gp :: GPPContext, m)
    n = gp.n + m
    if gp.n > capacity(gp)
        error("Added points exceed GPPContext capacity")
    end

    # Create new object (same storage)
    gpnew = GPPContext(gp.ctx, gp.η, gp.Xstore, gp.Kstore,
                       gp.cstore, gp.ystore, gp.scratch, n)

    # Refactor (if start from 0) or extend Cholesky (if partly done)
    if gp.n == 0
        refactor!(gpnew)
    else
        X1, X2 = getX(gp), getXrest(gp,m)
        R11 = getK(gp)
        K12 = view(gp.Kstore,1:gp.n,gp.n+1:n)
        K22 = view(gp.Kstore,gp.n+1:n,gp.n+1:n)
        kernel!(K12, gp.ctx, X1, X2)
        kernel!(K22, gp.ctx, X2, gp.η)
        ldiv!(UpperTriangular(R11)', K12)         # R12 = R11'\K12
        BLAS.syrk!('U', 'T', -1.0, K12, 1.0, K22) # S = K22-R12'*R12
        cholesky!(Symmetric(K22))                 # R22 = chol(S)
    end

    # Update c
    resolve!(gpnew)

    gpnew
end

function add_points!(gp :: GPPContext,
                     X :: AbstractMatrix, y :: AbstractVector)
    m = length(y)
    if size(X,2) != m
        error("Inconsistent number of points and number of values")
    end
    copy!(getXrest(gp,m), X)
    copy!(getyrest(gp,m), y)
    add_points!(gp, m)
end

function add_point!(gp :: GPPContext, x :: AbstractVector, y :: Float64)
    add_points!(gp, reshape(x, length(x), 1), [y])
end

function GPPContext(ctx :: KernelContext, η :: Float64,
                    X :: Matrix{Float64}, y :: Vector{Float64})
    d, n = size(X)
    if d != ndims(ctx)
        error("Mismatch in dimensions of X and kernel")
    end
    gp = GPPContext(ctx, η, n)
    copy!(gp.Xstore, X)
    copy!(gp.ystore, y)
    add_points!(gp, n)
end

#=
## Removing points

Removing points is rather simpler.
=#

function remove_points!(gp :: GPPContext, m)
    if m > gp.n
        error("Cannot remove $m > $(gp.n) points")
    end
    gpnew = GPPContext(gp.ctx, gp.η, gp.Xstore, gp.Kstore,
                       gp.cstore, gp.ystore, gp.scratch, gp.n-m)
    resolve!(gpnew)
    gpnew
end

#=
## Changing kernels

Changing the kernel is also simple, though it involves a complete
refactorization.
=#

function change_kernel_nofactor!(gp :: GPPContext, ctx :: KernelContext, η :: Float64)
    GPPContext(ctx, η, gp.Xstore, gp.Kstore,
               gp.cstore, gp.ystore, gp.scratch, gp.n)
end

function change_kernel!(gp :: GPPContext, ctx :: KernelContext, η :: Float64)
    gpnew = change_kernel_nofactor!(gp, ctx, η)
    refactor!(gpnew)
    resolve!(gpnew)
    gpnew
end

#=
## Predictive mean

And now we compute the predictive mean and its derivatives.
=#

function mean(gp :: GPPContext, z :: AbstractVector)
    ctx, X, c = gp.ctx, getX(gp), getc(gp)
    d, n = size(X)
    sz = 0.0
    for j = 1:n
        xj = @view X[:,j]
        sz += c[j]*kernel(ctx, z, xj)
    end
    sz
end

function gx_mean!(gsz :: AbstractVector, gp :: GPPContext, z :: AbstractVector)
    ctx, X, c = gp.ctx, getX(gp), getc(gp)
    d, n = size(X)
    for j = 1:n
        xj = @view X[:,j]
        gx_kernel!(gsz, ctx, z, xj, c[j])
    end
    gsz
end

function gx_mean(gp :: GPPContext, z :: AbstractVector)
    d = ndims(gp.ctx)
    gx_mean!(zeros(d), gp, z)
end

function Hx_mean!(Hsz :: AbstractMatrix, gp :: GPPContext, z :: AbstractVector)
    ctx, X, c = gp.ctx, getX(gp), getc(gp)
    d, n = size(X)
    for j = 1:n
        xj = @view X[:,j]
        Hx_kernel!(Hsz, ctx, z, xj, c[j])
    end
    Hsz
end

function Hx_mean(gp :: GPPContext, z :: AbstractVector)
    d = ndims(gp.ctx)
    Hx_mean!(zeros(d,d), gp, z)
end

#=
## Predictive variance

The predictive variance is a little less straighforward than the
predictive mean, but only a little.  The formula is
$$
  v(z) = k(z,z) - k_{zX} K_{XX}^{-1} k_{Xz},
$$
where we will assume $k(z,z)$ is constant.  Differentiating once with
respect to $z$ gives
$$
  \nabla v(x) = -2 \nabla k_{zX} (K_{XX}^{-1} k_{Xz}),
$$
and differentiating a second time gives
$$
  H v(x) =
  -2 \sum_j H k(z, x_j) (K_{XX}^{-1} k_{Xz})_j
  -2 (\nabla k_{zX}) K_{XX}^{-1} (\nabla k_{zX})^T.
$$
=#

function var(gp :: GPPContext, z :: AbstractVector)
    kXz = view(gp.scratch,1:gp.n,1)
    kernel!(kXz, gp.ctx, getX(gp), z)
    L = getKC(gp).L
    v = ldiv!(L, kXz)
    kernel(gp.ctx,z,z) - v'*v
end

function gx_var!(g :: AbstractVector, gp :: GPPContext, z :: AbstractVector)
    X, KC, ctx = getX(gp), getKC(gp), gp.ctx
    d, n = size(X)
    kXz  = view(gp.scratch,1:n,1)
    gkXz = view(gp.scratch,1:n,2:d+1)
    gkXz[:] .= 0.0
    for j = 1:n
        xj = @view X[:,j]
        kXz[j] = kernel(ctx, z, xj)
        gx_kernel!(view(gkXz,j,:), ctx, z, xj)
    end
    w = ldiv!(KC,kXz)
    mul!(g, gkXz', w, -2.0, 0.0)
end

function gx_var(gp :: GPPContext, z :: AbstractVector)
    d = ndims(gp.ctx)
    gx_var!(zeros(d), gp, z)
end

function Hx_var!(H :: AbstractMatrix, gp :: GPPContext, z :: AbstractVector)
    X, KC, ctx = getX(gp), getKC(gp), gp.ctx
    d, n = size(X)
    kXz  = view(gp.scratch,1:n,1)
    gkXz = view(gp.scratch,1:n,2:d+1)
    gkXz[:] .= 0.0
    for j = 1:n
        xj = @view X[:,j]
        kXz[j] = kernel(ctx, z, xj)
        gx_kernel!(view(gkXz,j,:), ctx, z, xj)
    end
    w = ldiv!(KC,kXz)
    invL_gkXz = ldiv!(KC.L, gkXz)
    H[:] .= 0.0
    for j = 1:n
        xj = @view X[:,j]
        Hx_kernel!(H, ctx, z, xj, w[j])
    end
    mul!(H, invL_gkXz', invL_gkXz, -2.0, -2.0)
end

function Hx_var(gp :: GPPContext, z :: AbstractVector)
    d = ndims(gp.ctx)
    Hx_var!(zeros(d,d), gp, z)
end

#=
## Demo

We did an earlier demo of the basic subroutines for computing a GP
predictive mean and variance; now let's do the same demo with the
`GPPContext` object.

```{julia}
let
    testf(x,y) = x^2+y
    Zk, y = test_setup2d(testf)
    ctx = KernelSE{2}(1.0)
    gp = GPPContext(ctx, 0.0, Zk, y)

    z = [0.456; 0.456]
    fz = testf(z...)
    μz, σz = mean(gp, z), sqrt(var(gp,z))
    zscore = (fz-μz)/σz
    println("""
        True value:       $fz
        Posterior mean:   $μz
        Posterior stddev: $σz
        z-score:          $zscore
        """)
end
```
=#
