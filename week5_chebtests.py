# Tests on the Chebyshev differentiation matrix routines

from numpy import *


def testeq(M, N, name):
    """
    testeq(M, N, name)

    Test for equality of two matrices (M and N) allowing for some floating point
    error. The parameter name is the name of the test.
    """
    D = abs(M - N)
    if all(D < 1e-6):
        print("Passed " + name)
    else:
        print("Failed " + name)


def runtests_cheb(cheb):
    """
    runtests_cheb(cheb)

    Run a small suite of tests on the Chebyshev differentiation matrix code
    provided. E.g.,

        from week5_chebtests import runtests_cheb
        runtests_cheb(mychebcode)
    """
    # Test 1
    M = cheb(1)
    N = array([[0.5, -0.5], [0.5, -0.5]])
    testeq(M, N, "test 1")

    # Test 2
    M = cheb(2)
    N = array([[1.5, -2.0, 0.5], [0.5, 0.0, -0.5], [-0.5, 2.0, -1.5]])
    testeq(M, N, "test 2")

    # Test 3
    M = cheb(5)
    N = array([[8.5,-10.472136,2.8944272,-1.527864,1.1055728,-0.5],
               [2.618034,-1.1708204,-2,0.89442719,-0.61803399,0.2763932],
               [-0.7236068,2,-0.17082039,-1.618034,0.89442719,-0.38196601],
               [0.38196601,-0.89442719,1.618034,0.17082039,-2,0.7236068],
               [-0.2763932,0.61803399,-0.89442719,2,1.1708204,-2.618034],
               [0.5,-1.1055728,1.527864,-2.8944272,10.472136,-8.5]])
    testeq(M, N, "test 3")
