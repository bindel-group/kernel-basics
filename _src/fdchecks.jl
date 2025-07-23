#ldoc on
#=
# Finite difference checks

A finite-difference tester is a useful thing to have.
=#

"""
    diff_fd(f, x=0.0; h=1e-6)

Compute a centered difference estimate of f'(x) with step size h.
"""
diff_fd(f, x=0.0; h=1e-6) = (f(x+h)-f(x-h))/(2h)
