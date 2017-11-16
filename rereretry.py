# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:52:16 2017

@author: Joby
"""

"""
Overall aim
Build a software package that performs numerical continuation on a given ODE
with respect to a given parameter.
"""
"""
Steps to get there
This is an exercise in defining appropriate interfaces and abstractions. Here you
will need to join together the different pieces of code that you have produced
over the last two weeks. In principle you could simply cut and paste all the code
together as one big function and use if statements, however, this would be poor
program design. Ideally, your program should have a modular structure with
the ability to use different discretisations.

An example top-level interface could be
results = continuation(myode, # the ODE to use
                       x0, # the initial state
                       par0, # the initial parameters
                       vary_par=0, # the parameter to vary
                       step_size=0.1, # the size of the steps to take
                       max_steps=100, # the number of steps to take
                       discretisation=shooting, # the discretisation to use
                       solver=scipy.optimize.fsolve) # the solver to use

Make (lots of) notes in your code about the decisions you are making! This is
particularly important this week!
For each of the exercises, consider in turn the following problems.




1. Write a code that performs natural parameter continuation, i.e., it simply
increments the a parameter by a set amount and attempts to find the
solution for the new parameter value using the last found solution as an
initial guess.
Write a code that performs pseudo-arclength continuation, i.e., the pseudoarclength
equation is added to the system of equations to solve for.

• Use your code to investigate the behaviour of the Duffing equation
for different values of Γ, e.g., 0.1, 0.2, 0.4, 0.8, etc.
Extensions

At certain points your code might fail because the step size is too large, but
if the step size is set to be too small the computations will take a long time.
Adaptive step sizes are a way to deal with this. A simple strategy is as follows.
If the root finder fails to converge, try again with half the step size; repeat
until the root finder converges and you can carry on as normal, or until you hit
a prespecified minimum step size (e.g. 0.000001) at which point end with an
error. If the root finder converges, increase the stepsize by 20% until you hit a
prespecified maximum step size
"""

import numpy as np
import scipy as sp
import math

import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve


     
"""
ODE LIST
"""

"""
• The algebraic cubic equation 
x3 − x + c = 0. In this case a discretisation
(shooting or collocation) is not needed (in the interface above you might 
have an option discretisation=lambda x: x, i.e., the equations are just
passed straight through to the solver). Vary c between -2 and 2.
"""
def simple_cubic(x, t, pars, *args):
    k=pars[0]
    y=(x[0]**3)-x[0]+k
    return y

"""
• The mass-spring-damper equation x¨ + 2ξx˙ + kx = sin(πt) (note that the
period is always 2 and can be defined on the domain [0, 2] or [−1, 1]
equivalently). Use ξ = 0.05 and vary k between 0.1 and 20.
"""
def mass_spring(u,t,pars, *args):
    k, epsilon, gamma=pars
    omega=np.pi #for now
    du=[u[1],gamma*np.sin(omega*t)-2*epsilon*u[1]-k*u[0]]
    return du

"""
• The Duffing equation x¨ + 2ξx˙ + kx + βx3 = Γ sin(πt) with ξ = 0.05, β = 1,
and Γ = 0.5 and vary k between 0.1 and 20.
"""
def duffy(u,t, pars, *args):
    k, epsilon, gamma=pars
    omega=np.pi
    du=[u[1], gamma*np.sin(omega*t)-2*epsilon*u[1]-k*u[0]-u[0]**3]
    return du    


"""
SHOOTING FUNCTIONS
"""
def shooting(ode, z0, time, pars, index, oldz=[], ds=1, mode="NPC", *args):
    data =ode, time, pars, index, oldz, ds, mode
    z, infodict, ier, mesg=fsolve(start_end_diff, z0, args=data, full_output=1 )
    if(ier==1):
        #print(mesg)
        #print("The roots are {}".format(x))
        return z # This returns array of same shape as x0 This will be a problem
    else:
        print(mesg)
        return float('Nan')

def start_end_diff(z,f, time, pars, index, oldz, ds, mode, *args):
    #this shall be our function to zero
    k, *x = z 
    
    if mode=="NPC":
        xs=odeint(f,x,time, args=(pars,))
        end=np.append(k,xs[-1])
        ans=z-end
    else:
        pars[index]=k
        ans=is_tan(z,oldz,ds)
        xs=odeint(f,x,time, args=(pars,))
        end=x-xs[-1]
        ans=np.append(ans, end)    
    return ans

"""
CONTINUATION Functions
"""
def islambda(v):
    LAMBDA = lambda:0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

def integrate(f, x0, time, pars):
    t, T= time
    ts=np.linspace(t, T, 10000)
    xs=odeint(f,x0,ts, args=(pars,))        
    return xs     
    
def predictor(z,ds):
    z0=z[0,:]
    z1=z[1,:]
    z2=z1+(z1-z0)*ds
    return z2
 
def is_tan(z, oldz, ds):
    z0=oldz[0,:]
    z1=oldz[1,:]
    #z0=oldz[0,0:2]
    #z1=oldz[1,0:2]
    z2=predictor(oldz,ds)
    return np.dot(z-z2,z1-z0)

def continuation(myode,x0,par0,vary_par=0,step_size=0.1,max_steps=100,disc=shooting,solver=sp.optimize.fsolve):
    time, end =par0.pop(-1)
    ds=1
    z0=np.append(par0[vary_par],x0)
    par0[vary_par]=z0[0]
    z=initialize(myode, z0, time, par0, vary_par, step_size, disc)
    runs=2
    
    while runs<=max_steps and z[-1,0]<=end:
        old_z=z[[-2,-1]]
        pred_z=predictor(old_z, ds)
        par0[vary_par]=z[-1,0]
        next_z=corrector(myode, pred_z,time, par0, vary_par, old_z, ds, disc)
        if(runs%10000==0):
            print(z[-1,0])
        #print(check(myode,next_z,old_z,par0,ds))
        z=np.vstack((z,next_z))
        runs +=1
        #print(runs, next_z)
    return z

def check(f, z, oldz, pars, ds):
    k, *x=z
    ans=is_tan(z,oldz,ds)
    xs=integrate(f,x,time,pars)
    diffx=xs[-1]-x
    ans=np.append(ans,diffx)
    return ans
    

def disc_step(f, z0, time, pars, index, disc, oldz=[], ds=1, mode="NPC"):
    if islambda(disc):
        return f(z0,time,pars)
    else:
        z=disc(f,z0,time,pars, index, oldz, ds, mode)
        if z.ndim==1: # This means shooting was used 
            return z
        else: #this means collocation and we need the first values, integrate later
            return z[0]

def initialize(f, z0, time, pars, index, ss, disc):
    z=np.zeros((2,len(z0)))
    for a in range(2):
        newz=disc_step(f, z0, time, pars,index, disc)
        z[a]=newz
        z0[0]+= ss
        pars[index]=z0[0]
    return z


def corrector(f, z, t, pars, index, oldz, ds, disc):
    znew=disc_step(f,z,t,pars,index, disc, oldz, ds, mode="PAL")
    return znew    
    


if __name__ == "__main__":
    
    """
    Initialize here
    """
    
    tstart=-1
    T=2
    k=0.1
    end=20
    par0=[k, 0.05, 1]
    vary_par=0
    step_size=0.1
    disc=shooting
    x0=np.array([1,0])
    
    
    
    time=[tstart,tstart+T]
    timepar=[time, end]
    par0.append(timepar)    
    results = continuation(mass_spring,x0,par0,vary_par,step_size,1000,disc,fsolve)
    #results = continuation(duffy,x0,par0,vary_par,step_size,1000000,disc,fsolve)
    #NOTE! This produces a matrix in the form [k,x]
    print("I finished!")
    
    ks=results[:,0]
    xs=results[:,1:]
    xmax=[]
    
    
    for a in range(len(ks)):
        par0[vary_par]=ks[a]
        Xs=integrate(mass_spring,xs[a],time,par0)
        xmax.append(max(Xs[0]))
    
    fig1=plt.figure(figsize=(10,5))
    ax1=fig1.add_subplot(1,1,1)
    ax1.set_xlabel("k")
    ax1.set_ylabel("||x||")
    #plt.title("Duffing")
   
    ax1.plot(ks,xmax)
    fig1.savefig('Mass Spring x vs k.svg')
    
        