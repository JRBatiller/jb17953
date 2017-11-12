# Tests on the collocation method routines

from numpy import *
import Duffing

def ode1(x, t, par):
    """
    ode1(x, t, par)

    Return the right-hand-side of the ODE

        x' = sin(pi*t) - x
    """
    return sin(pi*t) - x


def ode2(x, t, par):
    """
    ode2(x, t, par)

    Return the right-hand-side of the ODE

        x'' + par[0]*x' + par[1]*x = sin(pi*t)
    """
    return [x[1], sin(pi*t) - par[0]*x[1] - par[1]*x[0]]

def ode3(x,t,par):
    return cos(pi*t)-x

def ode4(x,t,par):
    return cos(t)

def runtests_ode(collocation):
    """
    runtests_ode(collocation)

    Run a small suite of tests on the collocation code provided. E.g.,

        from week5_odetests import runtests_ode
        runtests_ode(mycollocationcode)

    The collocation function should take the form

        collocation(ode, n, x0, pars)

    where ode is the right-hand-side of the ODE, n is the order of the
    polynomial to use, x0 is the initial guess at the solution, and pars is the
    parameters (if any, use [] for none).
    """
    # Solve the first ODE on the interval [-1, 1] with no parameters and zeros
    # as the starting guess; use 21 points across the interval
    n = 20  # 21 - 1 for the degree of polynomial needed
    x = cos(pi*arange(0, n+1)/n)  # the Chebyshev collocation points
    soln1 = collocation(ode1, n, zeros(n+1), [])
    exactsoln1 = 1/(1+pi**2)*sin(pi*x) - pi/(1+pi**2)*cos(pi*x)
    if linalg.norm(soln1 - exactsoln1) < 1e-6:
        print("ODE test 1 passed")
    else:
        print("ODE test 1 failed")
    
    l=zeros((n+1,2))
    pars = 0.1 , 1
    soln2 = collocation(ode2, n, l, pars)
    time=[-1,1]
    z=Duffing.shooting(ode2, np.array([0,0]), time, pars)
    exactsoln2 = odeint(ode2,z,x, args=(pars,))
    if linalg.norm(soln2 - exactsoln2) < 1e-6:
        print("ODE test 2 passed")
    else:
        print("ODE test 2 failed")
    
    soln3 = collocation(ode3, n, zeros(n+1), [])
    exactsoln3 = pi/(1+pi**2)*sin(pi*x) + 1/(1+pi**2)*cos(pi*x)
    if linalg.norm(soln3 - exactsoln3) < 1e-6:
        print("ODE test 3 passed")
    else:
        print("ODE test 3 failed")
    
    soln4 = collocation(ode4, n, zeros(n+1), [])
    exactsoln4 = cos(x)
    if linalg.norm(soln3 - exactsoln3) < 1e-6:
        print("ODE test 4 passed")
    else:
        print("ODE test 4 failed")

