# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:03:57 2017

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
def u_exact(x,t):
    # the exact solution
    y = np.exp(-(pi**2)*t)*np.sin(pi*x)
    return y

def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x)
    return y

def zero_bound(u,t):
    #takes a vector and adds boundary condition
    u[0]=0 #left bound
    u[-1]=0 #right bound
    return u


def heat_source(x,t):
    #heat source is some function f(x,t)
    f=x*0
    return f

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

def afe(ld, n):
    v=np.array([ld,1-2*ld,ld])
    A=tridiag(v,n)
    return A

def abe(ld, n):
    v=np.array([(-ld),1+2*ld,(-ld)])
    A=tridiag(v,n)
    return A

def acn(ld, n):
    v=np.array([(-ld/2),1+ld,(-ld/2)])
    A=tridiag(v,n)
    return A

def bcn(ld, n):
    v=np.array([ld/2,1-ld,ld/2])
    A=tridiag(v,n)
    return A

"""
Finite Diff methods
"""

def f_euler(ld, n):
    #forward euler A is identity matrix, B is AFE matrix
    A=sp.sparse.identity(n)
    B=afe(ld, n)    
    return A, B 
    
def b_euler(ld, n):
    #backward euler A is ABE matrix B is identity
    A=abe(ld,n)
    B=sp.sparse.identity(n)
    return A, B 

def crank_nich(ld, n):
    #A is ACN matrix and B is BCN matrix
    A=acn(ld, n)
    B=bcn(ld,n)
    return A, B

def fde_parabolic(K,L,T,xinit,mx, mt, bdry,method=f_euler, neumann=0, h_source=heat_source):
    #the main solver
    xs=np.linspace(0, L, mx+1)
    ts=np.linspace(0,T,mt+1)
    dx = xs[1] - xs[0]            # gridspacing in x
    dt = ts[1] - ts[0]            # gridspacing in t
    ld = K*dt/(dx**2)             # mesh fourier number
    print("lambda=",ld)
    u0=xinit(xs) #initialize u0
    A,B=method(ld,mx+1) #this gives you the left and right matrixes
    if neumann:
        u=neumann_solve(u0,xs,ts,ld,A,B,bdry, h_source)
    else:
        u=dirichlet_solve(u0,xs,ts,ld,A,B,bdry, h_source)
    return u

def dirichlet_solve(u0, xs, time, ld, lmatrix, rmatrix, bdry, F):
    lmatrix=truncate_matrix(lmatrix) #remove outermost elements of matrix
    rmatrix=truncate_matrix(rmatrix) #modification of matrixes don't matter
    #this might slow things down but at least it reduces if statements
    dt=time[1]-time[0]
    u=np.zeros(len(u0))
    addends=np.zeros(lmatrix.shape[0])
    #we do this out of for loop so its only done once!!!
    left = is_identity(lmatrix) 
    right = is_identity(rmatrix)
    for t in time[1:]:
        #addends from the rhs and lhs (transposed)
        raddend=ld/2*bdry(addends,t-dt)+dt*F(xs[1:-1],t-dt)
        laddend=ld/2*bdry(addends,t)+dt*F(xs[1:-1],t) 
        #think of a way to do this better
        if xor(left,right):
            #there is an identity matrix present
            if left:
                laddend=raddend
            else:
                raddend=laddend      
        rhs=rmatrix.dot(u0[1:-1])+(laddend+raddend)
        u[1:-1]=sp.sparse.linalg.spsolve(lmatrix,rhs)
        u=bdry(u,t)        
        u0=u
    return u

def neumann_solve(u0, xs, time, ld, lmatrix, rmatrix, nbdry, F):
    lmatrix=modify_evo(lmatrix) 
    rmatrix=modify_evo(rmatrix) 
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
  
def modify_evo(A):
    #this multiplies the appropriate elements by 2
    A=A.toarray() #so it can handle sparse matrixes
    A[0,1]=2*A[0,1]
    A[-1,-2]=2*A[-1,-2]
    A=sp.sparse.dia_matrix(A)
    return A

def modify_bound(brdr):
    #this changes the boundary to negative
    brdr[0]=-brdr[0]
    return brdr

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

def abs_errorx(K,L,T,xinit,mx,mt,bdry, method, steps, neumann=0,F=heat_source, exact=u_exact):
    elist=[]
    for a in range(steps):
        u1=fde_parabolic(K, L,T,xinit, mx,mt,bdry, method, neumann, F)
        soln=exact(mx,T)
        error=abs(u1-soln)**2
        error=np.sqrt(sum(error))
        if np.isfinite(error):
            elist.append(error)
        else:
            elist.append(0)
        mx=mx*2
    return elist

def abs_errort(K,L,T,xinit,mx,mt,bdry, method, steps, neumann=0,F=heat_source, exact=u_exact):
    elist=[]
    for a in range(steps):
        u1=fde_parabolic(K, L,T,xinit, mx,mt,bdry, method, neumann, F)
        soln=exact(mx,T)
        error=abs(u1-soln)**2
        error=np.sqrt(sum(error))
        if np.isfinite(error):
            elist.append(error)
        else:
            elist.append(0)
        mt=mt*2
    return elist

def get_errors(K,L,T,xinit,mx,mt,bdry, method,steps, neumann=0,F=heat_source):
    elist=[]
    u1=fde_parabolic(K, L,T,xinit, mx,mt,bdry, method, neumann, F)
    for a in range(steps):
        u2=fde_parabolic(K, L,T,xinit, 2*mx,2*mt,bdry, method, neumann, F)
        u2half=u2[::2]
        error=abs(u1-u2half)**2
        error=np.sqrt(sum(error))
        if np.isfinite(error):
            elist.append(error)
        else:
            elist.append(0)
        u1=u2
        mt=mt*2
        mx=mx*2
    return elist
    
if __name__ == "__main__":
    #initialize values here
    kappa = 1   # diffusion constant
    L=1         # length of spatial domain
    T=0.5       # total time to solve for
    mx = 10     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time
    
       
    u=fde_parabolic(kappa, L,T,u_I, mx,mt,zero_bound, crank_nich, neumann=0)
    
    #print(u)
    #pl.plot(x,u,'r-',x,u_exact(x,T),'go')
    x=np.linspace(0, L, mx+1) #for plotting
    t=np.linspace(0, L, mt+1)
    dx=x[1]-x[0]
    dt=t[1]-t[0]
    
    fig1=pl.figure(figsize=(10,5))
    ax1=fig1.add_subplot(1,1,1)
    ax1.plot(x,u)
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    #fig2.savefig('parabolic plot.svg')
    
    """
    errorx=get_errorx(kappa, L,T,u_I, mx,100000,zero_bound, crank_nich, neumann=0)
    xlist=np.array([2**a for a in range(8)])
    xlist=xlist*mx
    """
    
    steps=8
    error=get_errors(kappa, L,T,u_I, mx,mt,zero_bound, crank_nich, steps, neumann=0)
    #error=abs_errorx(kappa, L,T,u_I, mx,mt,zero_bound, crank_nich, steps, neumann=0)
    error=np.log(error)
    dh=np.array([2**a for a in range(steps)])
    dh=dh*dx*dx/dt
    dh=np.log(dh)
    fig2=pl.figure(figsize=(10,5))
    ax2=fig2.add_subplot(1,1,1)
    ax2.plot(dh,error)
    ax2.set_xlabel("logx")
    ax2.set_ylabel("log Error")
    #fig2.savefig('Phase Plot 2.svg')
    z = np.polyfit(dh, error, 1)
    p = np.poly1d(z)
    ax2.plot(dh,p(dh))
    # the line equation:
    print("y={}x+{}".format(z[0],z[1]))
    
    
    