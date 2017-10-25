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


if __name__ == "__main__":
    print(cheb(3))
    
    runtests_cheb(cheb)
    #PASSED!!!!