#ldoc on
#=
# Acquisition functions

We are interested in optimizing some objective function $f$ via
Bayesian optimization, which means that at each step of the algorithm
we will choose a new point $x_j$ at which to sample $f$ by optimizing
an *acquisition function* that balances exploration with exploitation.
We will focus on two such functions: the lower confidence bound, and
the expected improvement.

## Lower confidence bound

The lower confidence bound acquisition function is
$$
  \alpha(x) = \mu(x) - \lambda \sigma(x).
$$
We have already worked out spatial derivatives of the predictive mean
and variance, so all we need to do to finish is differentiating
$\sigma(x) = \sqrt{v(x)}$.  Formulas are given below; derivation is an
exercise for the reader:
$$\begin{aligned}
  \nabla \sigma &= \frac{1}{2\sigma} \nabla v \\
  H \sigma &= \frac{1}{2 \sigma} Hv -
  \frac{1}{4\sigma^3} (\nabla v) (\nabla v)^T
\end{aligned}$$
=#

function Hgx_αLCB(gp :: GPPContext, x :: AbstractVector, λ :: Float64)
    Copt = getCopt(gp)
    μ, gμ, Hμ = mean(gp, x), gx_mean(gp, x), Hx_mean(gp, x)
    v, gv, Hv = Copt*var(gp, x), Copt*gx_var(gp, x), Copt*Hx_var(gp, x)
    σ = sqrt(v)
    α = μ-λ*σ
    gα = gμ - λ/(2σ)*gv
    Hα = Hμ - λ/(2σ)*Hv + λ/(4*σ*v)*gv*gv'
    α, gα, Hα
end

#=
```{julia}
let
    Zk, y = test_setup2d((x,y)->x^2+y)
    gp = GPPContext(KernelSE{2}(0.5), 1e-8, Zk, y)
    z = [0.47; 0.47]
    dz = randn(2)
    λ = 2.3
    g(s)  = Hgx_αLCB(gp, z+s*dz, λ)[1]
    dg(s) = Hgx_αLCB(gp, z+s*dz, λ)[2]
    Hg(s) = Hgx_αLCB(gp, z+s*dz, λ)[3]
    @test dg(0)'*dz ≈ diff_fd(g) rtol=1e-6
    @test Hg(0)*dz ≈ diff_fd(dg) rtol=1e-6
end
```

## Expected improvement

We will follow the usual perspective of deriving expected improvement
as a method of *maximizing* an objective rather than *minimizing* the
objective.  We can (and will) switch perspectives later by multipling
by minus one.

### The standard derivation

The expected improvement is
$$
  \alpha(x) = \mathbb{E}[I(x)],
$$
where the improvement (for the minimization problem) is
$I(x) = (f(x)-f_*)_+$ where $f_*$ is the largest function value found
so far.  Equivalently,
$$
  \alpha(x) = \int_{f_*}^\infty (f-f_*) p(f) \, df
$$
where $p(f)$ is the predictive probability density for $f(x)$.
It is convenient to change variables to $z = (f-\mu)/\sigma$, which is
a standard normal random variable, giving us
$$\begin{aligned}
\alpha
&= \int_{z_*}^\infty \sigma (z-z_*) \phi(z) \, dz = \sigma G(z_*), \\
G(z_*) &:= \int_{z_*}^\infty (z-z_*) \phi(z) \, dz,
\end{aligned}$$
where $\phi(z)$ is the pdf or the standard normal.  We note that
$$\begin{aligned}
-\phi'(z) &= z \phi(z), & \Phi'(z) &= \phi(z)
\end{aligned}$$
where $\Phi(z)$ is the standard normal cdf.  Therefore,
$$\begin{aligned}
G(z_*)
&= \left. \left( -\phi(z)-z_*\Phi(z) \right) \right|_{z=z_*}^\infty \\
&= \phi(z_*)-z_* (1-\Phi(z_*)) \\
&= \phi(z_*)-z_* Q(z_*),
\end{aligned}$$
where $Q(z_*) = 1-\Phi(z_*) = \Phi(-z_*)$ is the complementary cdf for
the standard normal.  Equivalently, if we let
$u = -z_* = (\mu-f_*)/\sigma$, we have
$$
  G(z_*) = \phi(u) + u \Phi(u),
$$
which is the form that we see most often for the expected improvement.

A plot of the $G$ is perhaps useful:
```{julia}
let
    z = range(-4.0, 4.0, step=1e-3)
    G(z) = normpdf(z) - z*normccdf(z)
    plot(z, G.(z))
end
```
For sufficiently negative values, we have $G(z) \approx -z$, and for
sufficiently positive values we have $G(z) \approx 0$.  Near zero,
there is a smoothed-out transition region.

For optimization, it's useful to have derivatives as well as a formula
for the expected improvement.  Fortunately, they are pretty simple:
the first and second derivatives of $G$ are
$$\begin{aligned}
  G'(z) &= -Q(z) \\
  G''(z) &= \phi(z).
\end{aligned}$$

Per usual, a finite difference check gives us confidence in our
ability to do calculus.
```{julia}
let
    z = 0.123
    G(z) = normpdf(z) - z*normccdf(z)
    dG(z) = -normccdf(z)
    HG(z) = normpdf(z)
    @test dG(z) ≈ diff_fd(G,z) rtol=1e-6
    @test HG(z) ≈ diff_fd(dG,z) rtol=1e-6
end
```

### Negative log EI

One of the challenges of working with expected improvement is that
when $z_*$ gets larger than zero, the expected improvement becomes
exponentially small.  To analyze this, it is helpful to introduce the
*Mills ratio*
$$
  R(z) = Q(z)/\phi(z),
$$
so that we can rewrite the scaled EI function $G(z)$ as
$$
  G(z) = H(z) \phi(z), \quad H(z) = 1-zR(z).
$$
Using asymptotics of the Mills ratio for large $z$, we have
$$\begin{aligned}
  R(z) &= z^{-1} - z^{-3} + O(z^{-5}), \\
  H(z) &= z^{-2} - O(z^{-4}),
\end{aligned}$$
and so $G(z)$ decays asymptotically like $z^{-2} \phi(z)$ --- which is
rapid decay indeed.

Because it gets so close to zero, it's numerically advantageous to
work not with expected improvement, but with the negative log of the expected
improvement (NLEI).
$$
  \psi_{NLEI}
  = -\log \alpha_{EI}
  = -\frac{1}{2} \log(\sigma^2) - \log G(z_*).
$$
We will write $\psi_{NLG}(z) = -\log G(z)$ for the second term.

We evaluate $\psi_{NLG}$ differently depending on the argument.  For
$z$ negative (or not too positive), the standard formulas are entirely
reasonable.  That is,
$$\begin{aligned}
\psi_{NLG}(z) &= -\log\left( \phi(z) - z Q(z) \right) \\
\psi_{NLG}'(z) &= \frac{Q(z)}{G(z)} \\
\psi_{NLG}''(z) &= \frac{-\phi(z) G(z) + Q(z)^2}{G(z)^2}
\end{aligned}$$
We implement these formulas as "version zero" of the computation.
=#

function DψNLG0(z)
    ϕz = normpdf(z)
    Qz = normccdf(z)
    Gz = ϕz-z*Qz
    ψz = -log(Gz)
    dψz = Qz/Gz
    Hψz = (-ϕz*Gz + Qz^2)/Gz^2
    ψz, dψz, Hψz
end

#=
```{julia}
let
    z = 0.123
    @test DψNLG0(z)[2] ≈ diff_fd(z->DψNLG0(z)[1], z) rtol=1e-6
    @test DψNLG0(z)[3] ≈ diff_fd(z->DψNLG0(z)[2], z) rtol=1e-6
end
```

If we believe our model, we are unlikely to have to evaluate
$\psi_{NLG}$ for very large arguments.  Such an occurrence would mean
a very poorly calibrated model!  Nonetheless, very poor calibration
could potentially happen at some point, and it is worth knowing how to
deal with it.

For values of $z$ that are larger than about 26, the standard formulas
get perilously close to underflow.  In this case, it is more
reasonable to write $G(z) = H(z) \phi(z)$ and its derivatives in terms
of the Mills ratio $R(z)$ and the related $H(z) = 1-zR(z)$:
$$\begin{aligned}
\psi_{NLG}(z) &= -\log H(z) + \frac{z^2}{2} + \frac{1}{2} \log(2\pi) \\
\psi_{NLG}'(z) &= \frac{R(z)}{H(z)} \\
\psi_{NLG}''(z) &= \frac{-H(z)+R(z)^2}{H(z)^2}.
\end{aligned}$$
We can write the Mills ratio as $Q(z)/\phi(z)$ when $z$ is not too enormous;
alternately, we can write
$$
  R(z) = \sqrt{\frac{\pi}{2}}
  \operatorname{erfcx}\left( \frac{z}{\sqrt{2}} \right)
$$
where erfcx is the scaled complementary error function (assuming one
has a library of special functions that includes erfcx).
=#

function DψNLG1(z)
    Rz = sqrt(π/2)*erfcx(z/√2)
    Hz = 1-z*Rz
    ψz = -log1p(-z*Rz) + 0.5*(z^2 + log(2π))
    dψz = Rz/Hz
    Hψz = (-Hz+Rz^2)/Hz^2
    ψz, dψz, Hψz
end

#=
```{julia}
let
    z = 0.123
    z1 = 30.456
    @test DψNLG0(z)[1] ≈ DψNLG1(z)[1]
    @test DψNLG0(z)[2] ≈ DψNLG1(z)[2]
    @test DψNLG0(z)[3] ≈ DψNLG1(z)[3]
    @test DψNLG1(z)[2] ≈ diff_fd(z->DψNLG1(z)[1], z) rtol=1e-6
    @test DψNLG1(z)[3] ≈ diff_fd(z->DψNLG1(z)[2], z) rtol=1e-6
    @test DψNLG1(z1)[2] ≈ diff_fd(z->DψNLG1(z)[1], z1, h=1e-4) rtol=1e-6
    @test DψNLG1(z1)[3] ≈ diff_fd(z->DψNLG1(z)[2], z1, h=1e-4) rtol=1e-6
end
```

Were we unwilling to find a library for computing the scaled
complementary error function, another approach would be to use
Laplace's continued fraction expansion for the Mills ratio:
$$
  R(z) = \frac{1}{z+} \frac{1}{z+} \frac{2}{z+} \frac{3}{z+ \ldots},
$$
or we can write
$$
  R(z) = \frac{1}{z+W(z)}, \quad
  W(z) = \frac{1}{z+} \frac{2}{z+} \frac{3}{z+ \ldots},
$$
where $WR = H$.  Then we can put everything in terms of $W$:
$$\begin{aligned}
R(z) &= \frac{1}{z+W(z)} \\
H(z) &= W(z) R(z) \\
\psi_{NLG}(z) &= \log \left(1+\frac{z}{W(z)}\right) + \frac{z^2}{2} + \frac{1}{2} \log(2\pi) \\
\psi_{NLG}'(z) &= W(z)^{-1} \\
\psi_{NLG}''(z) &= \frac{1-W(z)(z+W(z))}{W(z)^2}.
\end{aligned}$$
=#

function DψNLG2(z)

    # Approximate W by 20th convergent
    W = 0.0
    for k = 20:-1:1
        W = k/(z+W)
    end

    ψz = log1p(z/W) + 0.5*(z^2 + log(2π))
    dψz = 1/W
    Hψz = (1-W*(z+W))/W^2

    ψz, dψz, Hψz
end

#=
```{julia}
let
    z = 5.23
    @test DψNLG0(z)[1] ≈ DψNLG2(z)[1]
    @test DψNLG0(z)[2] ≈ DψNLG2(z)[2]
    @test DψNLG0(z)[3] ≈ DψNLG2(z)[3]
end
```

The continued fraction doesn't converge super-fast, but that is almost
surely fine for what we're doing here.  By $z$ values of 5 or so, 20
terms is quite adequate to get good accuracy --- and this version does
not need any more exotic special functions in our library.  If needed,
we could do a similar manipulation to get an optimized rational
approximation to $W$ from Cody's 1969 rational approximation to the
complementary error function (erfc).  Or we could use a less accurate
approximation --- the point of getting the tails right is really to
give us enough information to climb out of the flat regions for EI.

Putting everything together, we have
=#

# By z = 6, a 20-term convergent of the rational approx is plenty
DψNLG(z) = if z < 6.0 DψNLG0(z) else DψNLG2(z) end

#=
```{julia}
let
    zs = range(-20, 20, step=1e-3)
    ψzs = [DψNLG(z)[1] for z in zs]
    plot(zs, ψzs)
end
```

Of course, if we are interested in minimization rather than
maximization, then we want to minimize $\psi_{NLG}(-z_*(x))$ rather than
$\psi_{NLG}(z_*(x))$.

### Gradients and Hessians

To choose a next point for a BO minimization problem using an EI
acquisition function, we minimize $\psi_{NLG}(u(x))$ where
$$
  u(x) = -z_*(x) = \frac{\mu(x)-f_*}{\sigma(x)}.
$$
Taking first and second derivatives, we have
$$\begin{aligned}
  u_{,i} &= \frac{\mu_{,i}}{\sigma} - u \frac{\sigma_{,i}}{\sigma} \\
  u_{,ij} &=
  \frac{\mu_{,ij}}{\sigma}
  -\frac{\mu_{,i}}{\sigma} \frac{\sigma_{,j}}{\sigma}
  -\frac{\mu_{,j}}{\sigma} \frac{\sigma_{,i}}{\sigma}
  -u \frac{\sigma_{,ij}}{\sigma}
  +u \frac{\sigma_{,i} \sigma_{,j}}{\sigma^2}
\end{aligned}$$
Let $v(x) = \sigma^2(x)$ be the variance; then note that
$$\begin{aligned}
  v_{,i} &= 2 \sigma \sigma_{,i} \\
  v_{,ij} &= 2 \sigma_{,j} \sigma_{,j} + 2 \sigma \sigma_{,ij},
\end{aligned}$$
which we can rearrange to
$$\begin{aligned}
\frac{\sigma_{,i}}{\sigma} &= \frac{v_{,i}}{2v} \\
\frac{\sigma_{,ij}}{\sigma} &=
\frac{v_{,ij}}{2v} - \frac{v_{,i}}{2v} \frac{v_{,j}}{2v}.
\end{aligned}$$
Substituting back in, we have
$$\begin{aligned}
u_{,i} &= \frac{\mu_{,i}}{\sigma} - u \frac{v_{,i}}{2v} \\
u_{,ij} &=
\frac{\mu_{,ij}}{\sigma} -
\frac{1}{2} \left(
\frac{\mu_{,i}}{\sigma} \frac{v_{,j}}{v} +
\frac{\mu_{,j}}{\sigma} \frac{v_{,i}}{v} \right) +
\frac{1}{2} u \left(
\frac{3}{2} \frac{v_{,i}}{v} \frac{v_{,j}}{v}
-\frac{v_{,ij}}{v}
\right)
\end{aligned}$$
In vector notation, we have
$$\begin{aligned}
\nabla u &=
\frac{\nabla \mu}{\sigma} -
\frac{u}{2} \frac{\nabla v}{v}
\\
H u &=
\frac{H \mu}{\sigma} -
\frac{1}{2} \left(
\frac{\nabla \mu}{\sigma} \frac{(\nabla v)^T}{v} +
\frac{\nabla v}{v} \frac{(\nabla \mu)^T}{\sigma}
\right) +
\frac{u}{2} \left(
\frac{3}{2}
\frac{\nabla v}{v} \frac{(\nabla v)^T}{v} -
\frac{H v}{v} \right).
\end{aligned}$$
=#

function Hgx_u(gp :: GPPContext, x :: AbstractVector, fopt :: Float64)
    Copt = getCopt(gp)
    μ, gμ, Hμ = mean(gp, x), gx_mean(gp, x), Hx_mean(gp, x)
    v, gv, Hv = Copt*var(gp, x), Copt*gx_var(gp, x), Copt*Hx_var(gp, x)

    σ = sqrt(v)
    gμs, Hμs = gμ/σ, Hμ/σ
    gvs, Hvs = gv/v, Hv/v

    u = (μ-fopt)/σ
    gu = gμs - 0.5*u*gvs
    Hu = Hμs - 0.5*(gμs*gvs' + gvs*gμs') + 0.5*u*(1.5*gvs*gvs' - Hvs)
    u, gu, Hu
end

#=
The derivatives of $\psi(u)$ are
$$\begin{aligned}
  \nabla \psi &= \psi'(u) \nabla u \\
  H \psi &= \psi''(u) \nabla u (\nabla u)^T + \psi'(u) H u.
\end{aligned}$$
=#

function Hgx_ψNLG(gp :: GPPContext, x :: AbstractVector, fopt :: Float64)
    u, gu, Hu = Hgx_u(gp, x, fopt)
    ψ, dψ, Hψ = DψNLG(u)
    ψ, dψ*gu, Hψ*gu*gu' + dψ*Hu
end

#=
Getting to the actual acquisition function, we note that
$$
\alpha_{NLEI} = -\log \alpha_{EI} = -\frac{1}{2} \log v + \psi_{NLG}(u)
$$
and the relevant derivatives of the leading $-(\log v)/2$ are
$$\begin{aligned}
-\frac{1}{2} \nabla (\log v) &= -\frac{1}{2} \frac{\nabla v}{v} \\
-\frac{1}{2} H (\log v) &=
-\frac{1}{2} \frac{Hv}{v}
+ \frac{1}{2} \frac{\nabla v}{v} \frac{(\nabla v)^T}{v}.
\end{aligned}$$
=#

function Hgx_αNLEI0(gp :: GPPContext, x :: AbstractVector, fopt :: Float64)
    Copt = getCopt(gp)
    v, gv, Hv = Copt*var(gp, x), Copt*gx_var(gp, x), Copt*Hx_var(gp, x)
    gvs, Hvs = gv/v, Hv/v

    u, gu, Hu = Hgx_u(gp, x, fopt)
    ψ, dψ, Hψ = DψNLG(u)

    -0.5*log(v) + ψ,
    -0.5*gvs + dψ*gu,
    -0.5*Hvs + 0.5*gvs*gvs' + Hψ*gu*gu' + dψ*Hu
end

#=
And we finally put everything together by combining all the algebra
above.
=#

function Hgx_αNLEI(gp :: GPPContext, x :: AbstractVector, fopt :: Float64)
    Copt = getCopt(gp)
    μ, gμ, Hμ = mean(gp, x), gx_mean(gp, x), Hx_mean(gp, x)
    v, gv, Hv = Copt*var(gp, x), Copt*gx_var(gp, x), Copt*Hx_var(gp, x)

    σ = sqrt(v)
    gμs, Hμs = gμ/σ, Hμ/σ
    gvs, Hvs = gv/(2v), Hv/v

    u = (μ-fopt)/σ
    ψ, dψ, Hψ = DψNLG(u)

    α = -log(σ) + ψ
    dα = dψ*gμs - (1+u*dψ)*gvs
    Hα = (-0.5*(1.0+u*dψ)*Hvs + dψ*Hμs + Hψ*gμs*gμs' +
        (2.0 + u^2*Hψ + 3.0*u*dψ)*gvs*gvs' +
        -(u*Hψ+dψ)*(gμs*gvs' + gvs*gμs'))

    α, dα, Hα
end

#=
Since we have several functions coming with a similar signature, we
put together a common tester.

```{julia}
function check_derivs3_NLEI(f)
    Zk, y = test_setup2d((x,y)->x^2+y)
    gp = GPPContext(KernelSE{2}(0.5), 1e-8, Zk, y)
    z = [0.47; 0.47]
    dz = randn(2)
    fopt = -0.1
    g(s)  = f(gp, z+s*dz, fopt)[1]
    dg(s) = f(gp, z+s*dz, fopt)[2]
    Hg(s) = f(gp, z+s*dz, fopt)[3]
    @test dg(0)'*dz ≈ diff_fd(g) rtol=1e-6
    @test Hg(0)*dz ≈ diff_fd(dg) rtol=1e-6
end

check_derivs3_NLEI(Hgx_u)
check_derivs3_NLEI(Hgx_ψNLG)
check_derivs3_NLEI(Hgx_αNLEI0)
check_derivs3_NLEI(Hgx_αNLEI)
```
=#
