#ldoc on
#=
## Testing
=#

@testset "Check Cholesky extension" begin

    # Pre-generated random matrix
    A = [ 1.91951   1.97238  1.02416   0.823799  2.14757;
          1.97238   2.40571  1.01815   1.24571   2.49423;
          1.02416   1.01815  0.956358  0.396442  1.378;
          0.823799  1.24571  0.396442  1.29615   1.25933;
          2.14757   2.49423  1.378     1.25933   3.00076]
    AC = cholesky(A)

    AC1 = extend_cholesky!(A, 0, 3)
    @test AC1.U ≈ AC.U[1:3,1:3]

    AC2 = extend_cholesky!(A, 3, 5)
    @test AC2.U ≈ AC.U
end

@testset "Check tridiagonalization" begin

    # Pre-generated random matrix and SPD
    A = [ 1.91951   1.97238  1.02416   0.823799  2.14757;
          1.97238   2.40571  1.01815   1.24571   2.49423;
          1.02416   1.01815  0.956358  0.396442  1.378;
          0.823799  1.24571  0.396442  1.29615   1.25933;
          2.14757   2.49423  1.378     1.25933   3.00076]
    y = [ 0.9390474198483851
          0.6376073515292209
          0.6127075058931962
          0.06289877680571387
          0.4328065454686951 ]
    Ay = A*y
    invAy = A\y
    tr_invA = tr(inv(A))

    # Check we have properties of trace correct.
    tridiag_reduce!(A)
    T = get_tridiag(A)
    tr_invT = tr(inv(T))
    @test tr_invA ≈ tr_invT

    # Check matrix multiply
    Ay2 = tridiag_applyQ!(A, T*tridiag_applyQT!(A, copy(y)))
    @test Ay ≈ Ay2

    # Check solve with LAPACK routines
    invAy2 = tridiag_applyQT!(A, copy(y))
    d, l = tridiag_params(A)
    LAPACK.pttrf!(d, l)
    LAPACK.pttrs!(d, l, invAy2)
    tridiag_applyQ!(A, invAy2)
    @test invAy ≈ invAy2

    # Check inverse trace
    @test tr_invA ≈ tridiag_tr_invLDLt(d, l)
end
