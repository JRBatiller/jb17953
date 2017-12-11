# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:59:07 2017

@author: Joby
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:24:15 2017

@author: Joby
"""
from operator import xor
import numpy as np
import pylab as pl
import scipy as sp
from numpy import pi
from scipy.sparse import spdiags

"""
Modify these as needed
"""
def u_exact(x,t, c):
    return np.cos(np.pi*c*t)*np.sin(np.pi*x)

def u_I(x):
    # initial temperature distribution
    y = np.array([np.sin(pi*x), 0*x])
    return y

def zero_bound(u,t):
    #takes a vector and adds boundary condition
    u[0]=0 #left bound
    u[-1]=0 #right bound
    return u


"""
Matrix Generating Functions
"""

def tridiag(v,n):
    #returns sparse tridiagonal array of size n given vector v
    v=np.tile(v,(n,1)).transpose()
    diags=np.array([-1,0,1])
    #A=spdiags(v,diags, n,n).toarray()
    A=spdiags(v,diags, n,n)
    return A

def aew(ld, n):
    v=np.array([ld**2, 2-2*ld**2, ld**2])
    A=tridiag(v,n)
    return A

def aiw(ld, n):
    v=np.array([-0.5*ld**2,1+ld**2, -0.5*ld**2 ])
    A=tridiag(v,n)
    return A

def biw(ld,n):
    v=np.array([0.5*ld**2,-1-ld**2, 0.5*ld**2])
    A=tridiag(v,n)
    return A

"""
Finite Diff methods
"""
def explicit(ld, n):
    A=sp.sparse.identity(n)
    B=aew(ld, n)
    return A, B/2

def implicit(ld,n):
    A=aiw(ld,n)
    B=sp.sparse.identity(n)
    #C=biw(ld,n)
    return A, B

def fde_hyperbolic(c,L,T,xinit,mx, mt, bdry, method=explicit, neumann=0):
    #the main solver
    xs=np.linspace(0, L, mx+1)
    ts=np.linspace(0,T,mt+1)
    dx = xs[1] - xs[0]            # gridspacing in x
    dt = ts[1] - ts[0]            # gridspacing in t
    ld = c*dt/dx                  # courant number
    print("lambda=",ld)
    u0=xinit(xs) #initialize u0
    matrix=method(ld,mx+1) #this gives you the left and right matrixes
    #A=modify_evo(A) 
    #B=modify_evo(B) 
    if neumann:
        u=neumann_solve(u0,xs,ts,ld,matrix,bdry)
    else:
        u=dirichlet_solve(u0,xs,ts,ld,matrix,bdry)
    return u

def dirichlet_solve(u0,xs,ts,ld,mtrx,bdry):
    dt=ts[1]-ts[0]
    dx=xs[1]-xs[0]
    A,B=mtrx
    A=truncate_matrix(A)
    B=truncate_matrix(B)
    C=-A
    u, v=u0
    addends=np.zeros(A.shape[0])
    u=np.tile(u, (3,1))
    explicit=is_identity(A)
    #FIRST STEP
    if(explicit):
        addend=ld**2*bdry(addends,ts[1])
    else:
        addend=0.5*ld**2*(bdry(addends,ts[1]+dt)+bdry(addends,ts[1]-dt))
    rhs=B.dot(u[0][1:-1])-dt*C.dot(v[1:-1])+addend
    u[1][1:-1]=sp.sparse.linalg.spsolve(A,rhs)
    u[1]=bdry(u[1],ts[1])    
    for t in ts[2:]:
        if(explicit):
            addend=ld**2*bdry(addends,t)
        else:
            addend=0.5*ld**2*(bdry(addends,t+dt)+bdry(addends,t-dt))
        rhs=2*B.dot(u[1][1:-1])+C.dot(u[0][1:-1])+addend
        u[2][1:-1]=sp.sparse.linalg.spsolve(A,rhs)
        u[2]=bdry(u[2], t)
        u[:2]=u[1:] #copy array down
    return u[2]
"""
def neumann_solve(u0, xs, time, ld, lmatrix, rmatrix, nbdry, F):
    dt=time[1]-time[0]
    dx=xs[1]-xs[0] 
    u=np.zeros(len(u0))
    addends=np.zeros(len(u0))
    #we do this out of for loop so its only done once!!!
    left = is_identity(lmatrix) 
    right = is_identity(rmatrix) 
    for t in time[1:]:
        raddend=ld*dx*modify_bound(nbdry(addends,t-dt))+dt*F(xs,t-dt)
        laddend=ld*dx*modify_bound(nbdry(addends,t))+dt*F(xs, t)
        #think of a way to do htis better
        if xor(left,right):
            if left:
                laddend=raddend
            else:
                raddend=laddend           
        rhs=rmatrix.dot(u0)+(laddend+raddend)
        u=sp.sparse.linalg.spsolve(lmatrix,rhs)
        u0=u
    return u
"""
def truncate_matrix(A):
    #this gets the inner matrix for dirichlet
    A=A.toarray()
    A=A[1:-1,1:-1]
    A=sp.sparse.dia_matrix(A)
    return A

def is_identity(A):
    #to find out if matrix is an identity matrix
    B=np.identity(A.shape[0])
    ans=(A.toarray()==B) #can't get not equals to work properly
    return ans.all()   

if __name__ == "__main__":
    #initialize values here
    c = 1   # diffusion constant
    L=1         # length of spatial domain
    T=2     # total time to solve for
    mx = 10     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time
    x=np.linspace(0, L, mx+1)
    t=np.linspace(0, T, mt+1)
    ue=fde_hyperbolic(c,L,T,u_I, mx, mt, zero_bound)
    ui=fde_hyperbolic(c,L,T,u_I, mx, mt, zero_bound, implicit)
    exact=u_exact(x,T,c)
    print(ue-ui)
    
    """
    fig1=pl.figure(figsize=(10,5))
    ax1=fig1.add_subplot(1,1,1)
    ax1.plot(x,u, 'r-', x, exact,'go')
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    """