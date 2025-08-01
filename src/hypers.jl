#ldoc on
#=
# Hyperparameter tuning

Kernel hyperparameters are things like the diagonal variance, length
scale, and noise variance.  When we don't have a good prior guess for
their values (which is usually the case), we need to find some way to
choose them automatically.  We generally denote the vector of
hyperparameters as $\theta$.  In the interest of minimizing visual
clutter, we will mostly write the kernel matrix as $K$ (rather than
$K_{XX}$) in this section.

A standard approach is to compute the
maximum (marginal) likelihood estimator; equivalently, we minimize the
negative log likelihood (NLL)
$$
  \phi(\theta) =
    \frac{1}{2} \log \det K + \frac{1}{2} y^T K^{-1} y +
    \frac{n}{2} \log(2\pi).
$$
The first term (the log determinant) penalizes model complexity;
the second term captures data fidelity; and the third term is simply
a dimension-dependent normalizing constant.  Of these, the first term
is generally the most tricky to work with numerically.

As a starting point, we note that after computing the Cholesky
factorization of $K$ and solving $c = K^{-1} y$, evaluating the
NLL can be done in $O(n)$ time, using the fact that
$$
  \frac{1}{2} \log \det K = \log \det R = \sum_{j} \log r_{jj}.
$$
=#

function nll(KC :: Cholesky, c :: AbstractVector, y :: AbstractVector)
    n = length(c)
    ϕ = (dot(c,y) + n*log(2π))/2
    for k = 1:n
        ϕ += log(KC.U[k,k])
    end
    ϕ
end

nll(gp :: GPPContext) = nll(getKC(gp), getc(gp), gety(gp))

#=
## Differentiating the NLL

Putting together the results of the previous section, we have
$$
  \delta \phi =
  \frac{1}{2} \tr(K^{-1} \, \delta K) -
  \frac{1}{2} c^T (\delta K) c
$$
where $c = K^{-1} y$.  For the second derivative, we have
$$\begin{aligned}
  \Delta \delta \phi =&
  \frac{1}{2} \tr(K^{-1} \Delta \delta K) -
  \frac{1}{2} \tr(K^{-1} \Delta K \, K^{-1} \delta K) \\
  &-\frac{1}{2} c^T (\Delta \delta K) c + c^T (\Delta K) K^{-1} (\delta K) c.
\end{aligned}$$

Now consider the case of $n$ data points and $d$ tunable hyperparameters.
In general, we can assume that $n$ is significantly larger than $d$;
if this is not the case, we probably need more data or fewer
hyperparameters!  Cholesky factorization of the kernel matrix in order
to compute a mean field or the negative log likelihood takes $O(n^3)$
time.  How long does it take to compute gradients and Hessians with
respect to the hyperparameters?

Just computing the matrix of derivatives of the kernel with respect to
a hyperparameter will generally take $O(n^2)$ time; so, with a few
exceptions, we do not expect to be able to compute any derivative term
in less than $O(n^2)$.  But how much more than $O(n^2)$ time might we
need?

#### Fast gradients

The tricky piece in computing gradients is the derivative of the log
determinant term.  If we are willing to form $K^{-1}$ explicitly, we
can write
$$
  \delta \phi =
  \left\langle \delta K, \frac{1}{2} \left( K^{-1}-cc^T \right) \right\rangle_F.
$$
Computing this way costs an additional fixed $O(n^3)$ cost to form
$K^{-1}$ explicitly, followed by $O(n^2)$ time to compute each of the
derivatives, for an overall cost of $O(n^3 + dn^2)$.  The danger here
is that $K^{-1}$ will generally have some very large entries with
alternating sign, and so the inner product here is somewhat
numerically sensitive.  Of course, this is associated with
ill-conditioning of $K$, which can be a problem for other methods of
computation as well.

If we compute the gradient this way, we do not ever need to
materialize the full $\delta K$ matrices.  We can also take advantage
of symmetry.
=#

function gθ_nll!(g :: AbstractVector, gp :: GPPContext,
                 invK :: AbstractMatrix)
    ctx, X, c = gp.ctx, getX(gp), getc(gp)
    d, n = size(X)
    for j = 1:n
        xj = @view X[:,j]
        cj = c[j]
        gθ_kernel!(g, ctx, xj, xj, (invK[j,j]-cj*cj)/2)
        for i = j+1:n
            xi = @view X[:,i]
            ci = c[i]
            gθ_kernel!(g, ctx, xi, xj, (invK[i,j]-ci*cj))
        end
    end
    g
end

gθ_nll(gp :: GPPContext, invK) =
    let nθ = nhypers(gp.ctx)
        gθ_nll!(zeros(nθ), gp, invK)
    end

#=
We treat the hyperparameter associated with the noise variance as a
special case.  If $z = \log \eta$ is the log-scale version of the
noise variance, we also want to compute the extended gradient with the
derivative with respect to $z$ at the end.
=#

function gθz_nll!(g, gp :: GPPContext, invK)
    ctx, X, c, s = gp.ctx, getX(gp), getc(gp), gp.η
    gθ_nll!(g, gp, invK)
    g[nhypers(ctx)+1] = (tr(invK) - c'*c)*s/2
    g
end

gθz_nll(gp :: GPPContext, invK) =
    gθz_nll!(zeros(nhypers(gp.ctx)+1), gp, invK)

gθz_nll(gp :: GPPContext) = gθz_nll(gp, getKC(gp)\I)

#=
#### Fast Hessians

If we want to compute with Newton methods, it is useful to
rewrite these expressions in a more symmetric way.  Let $K = LL^T$ be
the (lower triangular) Cholesky factorization of $K$, and define
$$\begin{aligned}
  \tilde{c} &= L^{-1} y \\
  \delta \tilde{K} &= L^{-1} (\delta K) L^{-T} \\
  \Delta \tilde{K} &= L^{-1} (\Delta K) L^{-T}
\end{aligned}$$

This is something we want to do systematically across hyperparameters.
=#

function whiten_matrix!(δK, KC :: Cholesky)
    ldiv!(KC.L, δK)
    rdiv!(δK, KC.U)
    δK
end

function whiten_matrices!(δKs, KC :: Cholesky)
    n, n, d = size(δKs)
    for k=1:d
        whiten_matrix!(view(δKs,:,:,k), KC)
    end
    δKs
end

#=
Then we can rewrite the gradient and Hessian as
$$\begin{aligned}
  \delta \phi &= \frac{1}{2} \tr(\delta \tilde{K}) - \frac{1}{2} \tilde{c}^T
  (\delta \tilde{K}) \tilde c \\
  \Delta \delta \phi &=
  \left\langle \frac{1}{2} \left( K^{-1} - cc^T \right), \Delta \delta K \right\rangle_F -
  \frac{1}{2} \langle \Delta \tilde{K}, \delta \tilde{K} \rangle_F
  + \langle \Delta \tilde{K} \tilde{c}, \delta \tilde{K} \tilde{c} \rangle
\end{aligned}$$
The cost here is an initial factorization and computation of $K^{-1}$,
an $O(n^3)$ factorization for computing each whitened perturbation
($\delta \tilde{K}$ or $\Delta \tilde{K}$) and then $O(n^2)$ for each
derivative component (gradient or Hessian) for a total cost of
$O((d+1)n^3 + d^2 n^2)$.

We would like to make a special case for the noise variance term.
However, after several attempts, I still do not have a method that
avoids forming $K^{-2}$ or $L^{-1} L^{-T}$, either of which requires
$O(n^3)$ time.  Hence, our current code treats $z = \log \eta$ like
the other hyperparameters, save that the kernel functions are not
called to compute the derivative.
=#

function mul_slices!(result, As, b)
    m, n, k = size(As)
    for j=1:k
        mul!(view(result,:,j), view(As,:,:,j), b)
    end
    result
end

function Hθ_nll(gp :: GPPContext)
    ctx, X, y, c, s = gp.ctx, getX(gp), gety(gp), getc(gp), gp.η
    d, n = size(X)
    nθ = nhypers(ctx)

    # Factorization and initial solves
    KC = getKC(gp)
    invK = KC\I
    c̃ = KC.L\y
    ϕ = nll(gp)
    ∂z_nll = (tr(invK)-(c'*c))*s/2

    # Set up space for NLL, gradient, and Hessian (including wrt z)
    g = zeros(nθ+1)
    H = zeros(nθ+1,nθ+1)

    # Add Hessian contribution from kernel second derivatives
    d, n = size(X)
    for j = 1:n
        xj = @view X[:,j]
        cj = c[j]
        Hθ_kernel!(H, ctx, xj, xj, (invK[j,j]-cj*cj)/2)
        for i = j+1:n
            xi = @view X[:,i]
            ci = c[i]
            Hθ_kernel!(H, ctx, xi, xj, (invK[i,j]-ci*cj))
        end
    end
    H[nθ+1,nθ+1] = ∂z_nll

    # Set up whitened matrices δK̃ and products δK*c and δK̃*c̃
    δKs = zeros(n, n, nθ+1)
    dθ_kernel!(δKs, ctx, X)
    for j=1:n  δKs[j,j,nθ+1] = s  end
    δK̃s = whiten_matrices!(δKs, KC)
    δK̃c̃s = mul_slices!(zeros(n,nθ+1), δK̃s, c̃)
    δK̃r = reshape(δK̃s, n*n, nθ+1)

    # Add Hessian contributions involving whitened matrices
    mul!(H, δK̃r', δK̃r, -0.5, 1.0)
    mul!(H, δK̃c̃s', δK̃c̃s, 1.0, 1.0)

    # And put together gradient gradient
    for j=1:nθ
        g[j] = tr(view(δK̃s,:,:,j))/2
    end
    mul!(g, δK̃c̃s', c̃, -0.5, 1.0)
    g[end] = ∂z_nll

    ϕ, g, H
end

#=
#### Fast approximate Hessians

As a final note, we can *estimate* second derivatives using stochastic
trace estimation, i.e.
$$
  \tr(A) = \mathbb{E}[Z^T A Z]
$$
where $Z$ is a probe vector with independent entries of mean zero and
variance 1.  Taking $W = L^{-T} Z$ gives us the *approximate* first
and second derivatives
$$\begin{aligned}
  \delta \phi =&
  \frac{1}{2} \tr(L^{-1} (\delta K) L^{-T}) - \frac{1}{2} c^T (\delta K) c \\
  \approx&
  \frac{1}{2} W^T (\delta K) W - \frac{1}{2} c^T (\delta K) c
  \\
  \Delta \delta \phi =&
  \frac{1}{2} \tr(L^{-1} (\Delta \delta K) L^{-T}) -
  \frac{1}{2} \tr(L^{-1} (\Delta K) L^{-T} L^{-1} (\delta K) L^{-T})
  \\
  & -\frac{1}{2} c^T (\Delta \delta K) c + c^T (\Delta K) L^{-T} L^{-1} (\delta K)
  c \\
  \approx&
  \frac{1}{2} W^T (\Delta \delta K) W -
  \frac{1}{2} \langle L^{-1} (\Delta K) W, L^{-1} (\delta K) W \rangle_F
  \\
  & -\frac{1}{2} c^T (\Delta \delta K) c + \langle L^{-1} (\Delta K) c, L^{-1}
  (\delta K) c \rangle.
\end{aligned}$$
The approximate gradient and approximate Hessian are consistent with
each other, which may be helpful if we want to do Newton on an
approximate objective rather than approximate Newton on the true NLL.
Or we can get an approximate Hessian and an exact gradient at the same
cost of $O(n^3 + d^2 n^2)$.

## Scale factors

We will frequently write our kernel matrix as $K = C \bar{K}$, where
$\bar{K}$ is a reference kernel matrix and $C$ is a scaling factor.
If $\bar{K}$ has diagonal equal to one, we can interpret it as a
correlation matrix; however, this is not necessary.
The scaling factor $C$ is a hyperparameter, but it is simple enough
that it deserves special treatment.  We therefore will separate out
the the hyperparameters into the scaling ($C$) and the rest of the
hypers ($\theta'$).  The key observation is that given the other
hyperparameters, we can easily compute the optimum value
$C_{\mathrm{opt}}(\theta')$ for the scaling.  This lets us work with a
reduced negative log likelihood
$$
  \bar{\phi}(\theta') = \phi(C_{\mathrm{opt}}(\theta'), \theta').
$$
This idea of eliminating the length scale is reminiscent of the
*variable projection* approaches of Golub and Pereira for nonlinear
least squares problems.

The critical point equation for optimizing $\phi$ with respect to $C$ yields
$$
  \frac{1}{2} C_{\mathrm{opt}}^{-1} \tr(I) -
  \frac{1}{2} C_{\mathrm{opt}}^{-2} y^T \bar{K}^{-1} y = 0,
$$
which, with a little algebra, gives us
$$
  C_{\mathrm{opt}} = \frac{y^T \bar{K}^{-1} y}{n}.
$$
Note that if $z = L^{-1} y$ is the "whitened" version of the $y$
vector, then $C_{\mathrm{opt}} = \|z\|^2/n$ is roughly the sample
variance.  This scaling of the kernel therefore corresponds
to trying to make the sample variance of the whitened signal equal to one.

If we substitute the formula for $C_{\mathrm{opt}}$ into the negative
log likelihood formula, we have the reduced negative log likelihood
$$\begin{aligned}
  \bar{\phi} &=
  \frac{1}{2} \log \det (C_{\mathrm{opt}} \bar{K}) +
  \frac{1}{2} y^T (C_{\mathrm{opt}} \bar{K})^{-1} y +
  \frac{n}{2} \log(2\pi) \\
  &=
  \frac{1}{2} \log \det \bar{K} +
  \frac{n}{2} \log C_{\mathrm{opt}} +
  \frac{1}{2} \frac{y^T \bar{K}^{-1} y}{C_{\mathrm{opt}}} + \frac{n}{2} \log(2\pi) \\
  &=
  \frac{1}{2} \log \det \bar{K} +
  \frac{n}{2} \log(y^T \bar{K}^{-1} y) +
  \frac{n}{2} \left( \log(2\pi) + 1 - \log n \right).
\end{aligned}$$
=#

function nllr(K̄C :: Cholesky, c̄ :: AbstractVector, y :: AbstractVector)
    n = length(c̄)
    ϕ = n*(log(dot(c̄,y)) + log(2π) + 1 - log(n))/2
    for k = 1:n
        ϕ += log(K̄C.U[k,k])
    end
    ϕ
end

nllr(gp :: GPPContext) = nllr(getKC(gp), getc(gp), gety(gp))
getCopt(gp :: GPPContext) = ( getc(gp)'*gety(gp) )/gp.n

#=
Differentiating $\bar{\phi}$ with respect to hyperparameters of $\bar{K}$
(i.e. differentiating with respect to any hyperparameter but the
scaling factor), we have
$$\begin{aligned}
  \delta \bar{\phi}
  &=
  \frac{1}{2} \tr(\bar{K}^{-1} \delta \bar{K})
  -\frac{n}{2} \frac{y^T \bar{K}^{-1} (\delta \bar{K}) \bar{K}^{-1} y}{y^T \bar{K}^{-1} y} \\
  &=
  \frac{1}{2} \tr(\bar{K}^{-1} \delta \bar{K})
  -\frac{1}{2} \frac{\bar{c}^T (\delta \bar{K}) \bar{c}}{C_{\mathrm{opt}}},
\end{aligned}$$
where $\bar{c} = \bar{K}^{-1} y$.  Alternately, we can write this as
$$
\delta \bar{\phi} =
  \frac{1}{2} \left\langle
  \bar{K}^{-1} - \frac{\bar{c}\bar{c}^T}{C_{\mathrm{opt}}}, \delta \bar{K}
  \right\rangle_F.
$$

For the second derivative, we have
$$\begin{aligned}
  \Delta \delta \bar{\phi} =&
  \frac{1}{2} \tr(\bar{K}^{-1} \Delta \delta \bar{K}) -
  \frac{1}{2} \tr(\bar{K}^{-1} \Delta \bar{K} \,
                  \bar{K}^{-1} \delta \bar{K}) \\
  &-\frac{1}{2} C_{\mathrm{opt}}^{-1}
    \bar{c}^T (\Delta \delta \bar{K}) \bar{c} +
  C_{\mathrm{opt}}^{-1}
    \bar{c}^T (\Delta \bar{K}) \bar{K}^{-1} (\delta \bar{K}) \bar{c} \\
  &-\frac{1}{2n}
    C_{\mathrm{opt}}^{-2}
    \bar{c}^T (\delta \bar{K}) \bar{c}
    \bar{c}^T (\Delta \bar{K}) \bar{c}.
\end{aligned}$$
This is very similar to the formula for the derivative of the
unreduced likelihood, except that (a) there is an additional factor of
$C_{\mathrm{opt}}^{-1}$ for the data fidelity derivatives in the
second line, and (b) there is an additional term that comes from the
derivative of $C_{\mathrm{opt}}$.  Consequently, we can rearrange for
efficiency here the same way that we did for the unreduced NLL.
If we want the gradient alone, we compute
$$
  \delta \phi =
  \left \langle
  \frac{1}{2} \left(
    \bar{K}^{-1} - C_{\mathrm{opt}}^{-1} \bar{c}\bar{c}^T
  \right),
  \delta \bar{K} \right \rangle_F.
$$
As before, our code has slightly special treatment for the case where
we also want derivatives with respect to $z = \log \eta$.
=#

function gθ_nllr!(g, gp :: GPPContext, invK, Copt)
    ctx, X, c = gp.ctx, getX(gp), getc(gp)
    d, n = size(X)
    for j = 1:n
        xj = @view X[:,j]
        cj = c[j]
        cj_div_Copt = cj/Copt
        gθ_kernel!(g, ctx, xj, xj, (invK[j,j]-cj*cj_div_Copt)/2)
        for i = j+1:n
            xi = @view X[:,i]
            ci = c[i]
            gθ_kernel!(g, ctx, xi, xj, (invK[i,j]-ci*cj_div_Copt))
        end
    end
    g
end

function gθz_nllr!(g, gp :: GPPContext, invK, Copt)
    ctx, X, c, s = gp.ctx, getX(gp), getc(gp), gp.η
    gθ_nllr!(g, gp, invK, Copt)
    g[nhypers(ctx)+1] = (tr(invK) - c'*c/Copt)*s/2
    g
end

gθ_nllr(gp :: GPPContext, invK, Copt) =
    gθ_nllr!(zeros(nhypers(gp.ctx)), gp, invK, Copt)

gθz_nllr(gp :: GPPContext, invK, Copt) =
    gθz_nllr!(zeros(nhypers(gp.ctx)+1), gp, invK, Copt)

gθ_nllr(gp :: GPPContext) = gθ_nllr(gp, getKC(gp)\I, getCopt(gp))
gθz_nllr(gp :: GPPContext) = gθz_nllr(gp, getKC(gp)\I, getCopt(gp))

#=
If we want the gradient and the Hessian, we compute
$$\begin{aligned}
  \delta \phi =&
  \tr(\delta \tilde{K})
  - \frac{1}{2} C_{\mathrm{opt}}^{-1} \tilde{c}^T \delta \tilde{K} \tilde{c}
\\
  \Delta \delta \phi =&
  \left \langle
  \frac{1}{2} \left(
    \bar{K}^{-1} - C_{\mathrm{opt}}^{-1} \bar{c}\bar{c}^T
  \right),
  \Delta \delta \bar{K} \right \rangle_F \\
  &-\frac{1}{2}
  \langle
  \Delta \tilde{K}, \delta \tilde{K}
  \rangle_F
  +C_{\mathrm{opt}}^{-1}
  \langle
  \Delta \tilde{K} \tilde{c}, \delta \tilde{K} \tilde{c}
  \rangle \\
  &
  -\frac{1}{2n} C_{\mathrm{opt}}^{-2}
  \left( \tilde{c}^T (\Delta \tilde{K}) \tilde{c} \right)
  \left( \tilde{c}^T (\delta \tilde{K}) \tilde{c} \right)
\end{aligned}$$
This differs from the unscaled version primarily in the last term,
which comes from differentiating $C_{\mathrm{opt}}^{-1}$ in the
gradient.  Hence, our Hessian code looks extremely similar to what we
wrote before.  We note that the last term can be rewritten in terms of
the first derivatives of the data fidelity term from before:
$$
  -\frac{1}{2n} C_{\mathrm{opt}}^{-2}
  \left( \tilde{c}^T (\Delta \tilde{K}) \tilde{c} \right)
  \left( \tilde{c}^T (\delta \tilde{K}) \tilde{c} \right)
=
  -\frac{2}{n}
  \left( -\frac{1}{2} C_{\mathrm{opt}}^{-1}
  \tilde{c}^T (\Delta \tilde{K}) \tilde{c} \right)
  \left( -\frac{1}{2} C_{\mathrm{opt}}^{-1}
  \tilde{c}^T (\delta \tilde{K}) \tilde{c} \right).
$$
This means the code for the Hessian of the reduced NLL is a very small
rearrangement of our code for the unreduced NLL.  In addition, we add
a small tweak to deal with the case where we don't care about the
derivatives with respect to $z = \log(\eta)$.
=#

function Hθ_nllr(gp :: GPPContext; withz=true)
    ctx, X, y, s = gp.ctx, getX(gp), gety(gp), gp.η
    d, n = size(X)
    nθ = nhypers(ctx)

    # Factorization and initial solves
    KC = getKC(gp)
    invK = KC\I
    c̃ = KC.L\y
    c = getc(gp)
    Copt = getCopt(gp)
    ϕ = nllr(gp)
    ∂z_nllr = (tr(invK)-(c'*c)/Copt)*s/2

    # Set up space for NLL, gradient, and Hessian (including wrt z)
    nt = withz ? nθ+1 : nθ
    g = zeros(nt)
    H = zeros(nt,nt)

    # Add Hessian contribution from kernel second derivatives
    for j = 1:n
        xj = @view X[:,j]
        cj = c[j]
        cj_div_Copt = cj/Copt
        Hθ_kernel!(H, ctx, xj, xj, (invK[j,j]-cj*cj_div_Copt)/2)
        for i = j+1:n
            xi = @view X[:,i]
            ci = c[i]
            Hθ_kernel!(H, ctx, xi, xj, (invK[i,j]-ci*cj_div_Copt))
        end
    end
    if withz
        H[nt,nt] = ∂z_nllr
    end

    # Set up matrices δK
    δKs = zeros(n, n, nt)
    dθ_kernel!(δKs, ctx, X)
    if withz
        for j=1:n
            δKs[j,j,nt] = s
        end
    end

    # Set up whitened matrices δK̃ and products δK*c and δK̃*c̃
    δK̃s = whiten_matrices!(δKs, KC)
    δK̃c̃s = mul_slices!(zeros(n,nt), δK̃s, c̃)
    δK̃r = reshape(δK̃s, n*n, nt)

    # Add Hessian contributions involving whitened matrices
    mul!(H, δK̃r', δK̃r, -0.5, 1.0)
    mul!(H, δK̃c̃s', δK̃c̃s, 1.0/Copt, 1.0)

    # Last term of the Hessian written via data fidelity part of gradient
    mul!(g, δK̃c̃s', c̃, -0.5/Copt, 1.0)
    if withz
        g[end] = -(c'*c)/Copt*s/2
    end
    mul!(H, g, g', -2.0/n, 1.0)

    # And finish the gradient
    for j=1:nθ
        g[j] += tr(view(δK̃s,:,:,j))/2
    end
    if withz
        g[end] = ∂z_nllr
    end

    ϕ, g, H
end

#=
The vector $c = \bar{c}/C_{\mathrm{opt}}$ can be computed later if
needed.  However, we usually won't need it, as we can write the the
posterior mean at a new point $z$ as $\bar{k}_{Xz}^T \bar{c}$.  The
posterior variance is
$C_{\mathrm{opt}} (\bar{k}_{zz} -
\bar{k}_{Xz}^T \bar{K}_{XX}^{-1} \bar{k}_{Xz}).$

## Newton solve

We are now in a position to tune the hyperparameters for a GP example.
The problematic bit, which we will deal with in the next section, is
the noise variance.  Once we have a reasonable estimate for the
optimized noise variance, we can get the expected quadratic
convergence of Newton iteration.  But the basin of convergence is
rather small.

This iteration is meant to demonstrate this point -- if we were trying
to do this for real, of course, we would do a line search!  As it is,
we just cut the step if we are changing the noise variance by more
than a factor of 20 or so.

```{julia}
function newton_hypers0(ctx :: KernelContext, η :: Float64,
                        X :: AbstractMatrix, y :: AbstractVector;
                        niters = 12, max_dz=3.0,
                        monitor = (ctx, η, ϕ, gϕ)->nothing)
    θ = getθ(ctx)
    for j = 1:niters
        ϕ, gϕ, Hϕ = Hθ_nllr(ctx, X, y, η)
        monitor(ctx, η, ϕ, gϕ)
        u = -Hϕ\gϕ
        if abs(u[end]) > max_dz
            u *= max_dz/abs(u[2])
        end
        θ[:] .+= u[1:end-1]
        ctx = updateθ(ctx, θ)
        η *= exp(u[end])
    end
    ctx, η
end

function newton_hypers0(gp :: GPPContext;
                        niters = 12, max_dz=3.0,
                        monitor = (ctx, η, ϕ, gϕ)->nothing)
    ctx, η = gp.ctx, gp.η
    θ = getθ(ctx)
    for j = 1:niters
        ϕ, gϕ, Hϕ = Hθ_nllr(gp)
        monitor(ctx, η, ϕ, gϕ)
        u = -Hϕ\gϕ
        if abs(u[end]) > max_dz
            u *= max_dz/abs(u[2])
        end
        θ[:] .+= u[1:end-1]
        ctx = updateθ(ctx, θ)
        η *= exp(u[end])
        gp = change_kernel!(gp, ctx, η)
    end
    gp
end

let
    Zk, y = test_setup2d((x,y) -> x^2 + cos(3*y) + 5e-4*cos(100*y), 40)
    gp = GPPContext(KernelSE{2}(0.7), 1e-4, Zk, y)

    normgs = []
    function monitor(ctx, η, ϕ, gϕ)
        ℓ = ctx.ℓ
        println("($ℓ,$η)\t$ϕ\t$(norm(gϕ))")
        push!(normgs, norm(gϕ))
    end

    gp = newton_hypers0(gp, monitor=monitor)
    println("Finished with ($(gp.ctx.ℓ), $(gp.η))")
    plot(normgs, yscale=:log10)
end
```

## Noise variance

Unfortunately, the reduced NLL function (viewed as a function of
$\eta$ alone) is not particularly nice.  It has a large number of
log-type singularities on the negative real axis ($2n-1$ of them),
with some getting quite close to zero.  So Newton-type iterations are
likely to be problematic unless we have a good initial guess,
particularly when we are looking for optima close to zero.
Fortunately, working with $\log \eta$ rather than $\eta$ mitigates
some of the numerical troubles; and, furthermore, we can be clever
about our use of factorizations in order to make computing the reduced
NLL and its derivatives with respect to $\eta$ sufficiently cheap so
that we don't mind doing several of them.

We could evaluate the reduced NLL and its derivatives
quickly by solving an eigenvalue problem -- an up-front $O(n^3)$ cost
followed by an $O(n)$ cost for evaluating the NLL and derivatives.
Better yet, we do not need to get all the way to an eigenvalue
decomposition -- a tridiagonal reduction is faster and gives the same
costs.  That is, we write
$$
  T = Q^T K Q, \tilde{y} = Q^T y
$$
and note that
$$\begin{aligned}
  Q^T (K + \eta I) Q &= T + \eta I, \\
  \log \det (K + \eta I) &= \log \det (T + \eta I), \\
  y^T (K+\eta I)^{-1} y &= \tilde{y}^T (T + \eta I)^{-1} \tilde{y}.
\end{aligned}$$
Therefore, all the computations that go into the reduced NLL and its
derivatives can be rephrased in terms of $T$ and $\tilde{y}$.  Using
the factorization $T + \eta I = LDL^T$, we can compute
$\log \det(K + \eta I) = \sum_i \log d_i$, and we previously also wrote
a short code to compute $\tr((K+\eta I)^{-1}) = \tr((T+\eta I)^{-1})$
using the $LDL^T$ factorization.

Putting this all together, we have the following code for computing
the reduced negative log likelihood and its derivative after a
tridiagonal reduction.
=#

function nllrT!(T, y, s, alpha, beta, c)
    n = length(y)
    tridiag_params!(T, alpha, beta)
    alpha[:] .+= s
    c[:] .= y
    LAPACK.pttrf!(alpha, beta)
    LAPACK.pttrs!(alpha, beta, c)
    cTy = c'*y
    Copt = cTy/n

    # Compute NLL
    ϕ̄ = n*(log(cTy) + log(2π) + 1 - log(n))
    for j = 1:n
        ϕ̄ += log(alpha[j])
    end
    ϕ̄ /= 2.0

    # Compute NLL derivative
    dϕ̄ = (tridiag_tr_invLDLt(alpha, beta) - (c'*c)/Copt)/2

    ϕ̄, dϕ̄
end

function nllrT(T, y, s)
    n = length(y)
    alpha = zeros(n)
    beta = zeros(n-1)
    c = zeros(n)
    nllrT!(T, y, s, alpha, beta, c)
end

#=
The main cost of anything to do with noise variance is the initial
tridiagonal reduction.  After that, each step costs only $O(n)$, so we
don't need to be too penny-pinching about the cost of doing
evaluations.  Therefore, we use an optimizer that starts with a brute
force grid search (in $\log \eta$) to find a bracketing interval for a
best guess at the global minimum over the range, and then does a few
steps of secant iteration to refine the result.
=#

function min_nllrT!(T, y, ηmin, ηmax, alpha, beta, c; nsamp=10, niter=5)

    # Sample on an initial grid
    logmin = log(ηmin)
    logmax = log(ηmax)
    logx   = range(logmin, logmax, length=nsamp)
    nllx   = [nllrT!(T,y,exp(s),alpha,beta,c) for s in logx]

    # Find min sample index and build bracketing interval
    i = argmin(t[1] for t in nllx)
    if ((i == 1 && nllx[i][2] > 0) ||
        (i == nsamp && nllx[i][2] < 0))
        return exp(logx[i]), nllx[i]...
    end
    ilo = i
    ihi = i
    if nllx[i][2] < 0
        ihi = i+1
    else
        ilo = i-1
    end

    # Do a few steps of secant iteration
    a,  b  = logx[ilo], logx[ihi]
    fa, fb = nllx[ilo], nllx[ihi]
    for k = 1:niter
        dfa, dfb = exp(a)*fa[2], exp(b)*fb[2]
        d = (a*dfb-b*dfa)/(dfb-dfa)
        fd = nllrT!(T,y,exp(d),alpha,beta,c)
        a, b = b, d
        fa, fb = fb, fd
    end

    exp(b), nllrT!(T,y,exp(b),alpha,beta,c)...
end

#=
It's useful to use the GP context objects to wrap this up.
=#

function tridiagonalize!(gp :: GPPContext)
    K = getK(gp)
    y = view(gp.scratch,1:gp.n,1)
    kernel!(K, gp.ctx, getX(gp))
    copy!(y, gety(gp))
    tridiag_reduce!(K)
    tridiag_applyQT!(K, y)
end

function nllrT!(gp :: GPPContext, η :: Float64)
    K     = getK(gp)
    y     = view(gp.scratch,1:gp.n,1)
    alpha = view(gp.scratch,1:gp.n,2)
    beta  = view(gp.scratch,1:gp.n-1,3)
    c     = getc(gp)
    nllrT!(K,y,η,alpha,beta,c)
end

function min_nllrT!(gp :: GPPContext, ηmin, ηmax;
                    tridiagonalized=false, nsamp=10, niter=5)
    if !tridiagonalized
        tridiagonalize!(gp)
    end
    K     = getK(gp)
    y     = view(gp.scratch,1:gp.n,1)
    alpha = view(gp.scratch,1:gp.n,2)
    beta  = view(gp.scratch,1:gp.n-1,3)
    c     = getc(gp)
    ηopt, ϕ, dϕ = min_nllrT!(K,y,ηmin,ηmax,alpha,beta,c,
                             nsamp=nsamp, niter=niter)
    change_kernel!(gp, gp.ctx, ηopt), ϕ, dϕ
end

#=
An example in this case is useful.

```{julia}
let
    # Set up sample points and test function
    n = 40
    Zk, y = test_setup2d((x,y) -> x^2 + cos(3*y + 5e-4*cos(100*x)), n)
    ℓ = 1.2

    gp = GPPContext(KernelSE{2}(ℓ), 1e-10, Zk, y)
    tridiagonalize!(gp)
    ss = 10.0.^(-10:0.1:-2)
    ϕss = [nllrT!(gp,s)[1] for s in ss]
    gp, ϕ̄, dϕ̄ = min_nllrT!(gp,1e-10,1e-2,tridiagonalized=true)

    plot(ss, ϕss, xscale=:log10, legend=:false)
    plot!([gp.η], [ϕ̄], marker=:circle)
end
```

## Putting it together

With the fast tuning of the noise variance parameter ready, we can
tweak our Newton iteration so that all the other hyperparameters are
updated by Newton, and then the noise variance is updated with our
tridiagonal solve.

```{julia}
function newton_hypers1(gp :: GPPContext;
                        ηmin = 1e-10, ηmax = 1e-2,
                        niters = 12, max_dz=3.0,
                        monitor = (ctx, η, ϕ, gϕ)->nothing)
    ctx, η = gp.ctx, gp.η
    θ = getθ(ctx)
    for j = 1:niters
        ϕ, gϕ, Hϕ = Hθ_nllr(gp)
        monitor(ctx, gp.η, ϕ, gϕ)
        u = -Hϕ\gϕ
        if abs(u[end]) > max_dz
            u *= max_dz/abs(u[2])
        end
        θ[:] .+= u[1:end-1]
        ctx = updateθ(ctx, θ)
        gp = change_kernel_nofactor!(gp, ctx, η)
        gp, _, _ = min_nllrT!(gp, ηmin, ηmax)
    end
    gp
end

let
    n = 40
    Zk, y = test_setup2d((x,y) -> x^2 + cos(3*y) + 1e-3*cos(100*x), n)
    gp = GPPContext(KernelSE{2}(1.2), 1e-10, Zk, y)

    normgs = []
    function monitor(ctx, η, ϕ, gϕ)
        println("$(ctx.ℓ)\t$η\t$ϕ\t$(norm(gϕ))")
        push!(normgs, norm(gϕ))
    end
    newton_hypers1(gp, monitor=monitor)
    plot(normgs, yscale=:log10)
end
```

An alternate approach is perhaps appropriate if we want to use a
third-party Newton solver (e.g. something from `Optim.jl` on all the
hyperparameters other than the noise variance.  If we split the
hyperparameters into the $\theta = (\gamma, z)$ where $z$ is the
log noise variance and $\gamma$ is everything else, then
the Newton step at an optimal $z_{\mathrm{opt}}(\gamma)$ satisfies
$$
  \begin{bmatrix}
    H_{\gamma\gamma} & H_{\gamma z} \\
    H_{z \gamma} & H_{zz}
  \end{bmatrix}
  \begin{bmatrix}
    u_\gamma \\
    u_z
  \end{bmatrix} =
  \begin{bmatrix} g_\gamma \\ 0 \end{bmatrix}
$$
or
$$
  (H_{\gamma\gamma} - H_{\gamma z} H_{zz}^{-1} H_{z \gamma}) u_\gamma
  = -g_\gamma.
$$
assuming that $z$ is free to move.  If $z$ is at one of the
constraints, then we have just $H_{\gamma \gamma} u_\gamma = -g_\gamma$.
=#
