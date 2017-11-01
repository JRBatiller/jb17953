# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:28:02 2017

@author: Joby
"""
"""
Write a general collocation code to enable a user to easily solve periodic boundary
value problems. Take as input
1. an arbitrary ordinary differential equation (in first-order form),
2. the parameters of ODE, and
3. a guess at the state variables over a whole period.
The code should try to solve the differential equation with periodic boundary
conditions (i.e., u(0) = u(T)) and return the corrected state variables over a
whole period.
"""
#hoooooooookaaaaaaaayyyyyyyyyyy


"""
1. Write a function that constructs a Chebyshev differentiation matrix and
ensure that it passes the specified tests.
• Specifically, use Chebyshev points of the second kind.
• You can use the Matlab code on Blackboard as a starting point or
equations (4.2) and (5.4) from “Barycentric Lagrange Interpolation”
by Berrut and Trefethen, SIAM Review 2004. What are the pros and
cons of either starting point?
• (You could also use the standard definition of Lagrange polynomials
if so desired; again what are the pros and cons?)
"""

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import week5_chebtests
import week5_odetests

def diff(f, u,t):
    #this shall be our function to zero
    
    
    func=f(u,t,[]) # this is the last element
    diff=u-func
    #df=np.array([f[1],du_dt(uinitial,t,gamma,epsilon)[1]-du_dt(ufinal,t+T,gamma,epsilon)[1]])
    #Ans=np.array([f, df])
    return diff

def cheb(N):
    if(not isinstance(N,int)):
        raise ValueError("N must be a positive integer")
        return 0
    if(N<=0):
        raise ValueError("N must be a positive integer")
        return 0
    #we need an x matrix
    x=np.array([np.cos(np.pi*i/N) for i in range (N+1)])
    #we need a c matrix =/
    c=np.ones(N+1)
    c[0]=2
    c[N]=2
    
    #we want a square matrix of side N+1
    D=np.zeros((N+1,N+1))
    #fill in the values
    for i in range(N+1):
        for j in range(N+1):
            if( i==0 and j==0):
                D[i][j]=(2*N*N+1)/6
            elif(i==N and j==N):
                D[i][j]=-(2*N*N+1)/6
            elif(i==j):
                D[i][j]=-x[i]/(2*(1-x[i]**2))
            else:
                D[i][j]=(c[i]/c[j])*((-1)**(i+j))/(x[i]-x[j])
    return D
    #improve on this later
    
def cheb2(N):
    #lets try doing the matlab way
    if(not isinstance(N,int)):
        raise ValueError("N must be a positive integer")
        return 0
    if(N<=0):
        raise ValueError("N must be a positive integer")
        return 0
    x=np.array([np.cos(np.pi*i/N) for i in range (N+1)])
    c=np.ones(N+1)
    c[0]=2
    c[N]=2
    d=np.array([(-1)**n for n in range(N+1)])
    c=c*d
    X=tile(x,(N+1,1)).transpose()
    dX=X-X.transpose()
    #D  = (c*(1./c)')./(dX+(eye(N+1)));
    D=c.reshape(N+1,1)*(1/c)/(dX+np.identity(N+1))
    D=D-np.diag(sum(D,1))
    return D
    #D  = D - diag(sum(D'));
    
"""
2. Use the Chebyshev differentiation matrix to differentiate known functions
over some interval (e.g., sin(x) for −1 ≤ x ≤ 1).
• Write a number of tests that your code should satisfy and ensure that
your code does satisfy them.
"""
def collocation(ode, n, x0, pars):
    t=np.array([np.cos(np.pi*i/n) for i in range (n+1)])
    f=ode(x0,t,pars)
    D=cheb(n)
    Dx=np.sum(D*x0,1)
    x, infodict, ier, mesg=sp.optimize.fsolve(diff, Dx, args=(f, t), full_output=1 )
    
    return x


if __name__ == "__main__":
    cheb(6) 
    runtests_cheb(cheb)
    runtests_cheb(cheb2)
    runtests_ode(collocation)
    #PASSED!!!!
    
    