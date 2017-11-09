# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:06:57 2017

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

• The algebraic cubic equation 
x3 − x + c = 0. In this case a discretisation
(shooting or collocation) is not needed (in the interface above you might 
have an option discretisation=lambda x: x, i.e., the equations are just
passed straight through to the solver). Vary c between -2 and 2.

• The mass-spring-damper equation x¨ + 2ξx˙ + kx = sin(πt) (note that the
period is always 2 and can be defined on the domain [0, 2] or [−1, 1]
equivalently). Use ξ = 0.05 and vary k between 0.1 and 20.

• The Duffing equation x¨ + 2ξx˙ + kx + βx3 = Γ sin(πt) with ξ = 0.05, β = 1,
and Γ = 0.5 and vary k between 0.1 and 20.
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

import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.optimize import fsolve


"""
• The algebraic cubic equation 
x3 − x + c = 0. In this case a discretisation
(shooting or collocation) is not needed (in the interface above you might 
have an option discretisation=lambda x: x, i.e., the equations are just
passed straight through to the solver). Vary c between -2 and 2.
"""
def simple_cubic(x, k, *args):
    y=(x**3)-x+k
    return y

def dot_prod(z, oldz, step_size, *args):
    #NO MORE ELEGANCE I WILL MAKE THIS WORK
    k,x = z #THIS IS JUST A GUESS
    k0, x0 =oldz[0]
    k1, x1 =oldz[1]
    
    estk,estx=predictor(oldz,step_size)
    v1=np.array([k-estk, x-estx])
    v2=np.array([k1-k0, x1-x0])
    y=np.dot(v1,v2)
    return y

def F(z,oldz,step_size,*args):
    k,x=z
    f=np.zeros(2)
    f[0]=simple_cubic(x,k)
    f[1]=dot_prod(z,oldz, step_size)
    return f

def predictor(z, step_size, *args):
    k0, x0=z[0]
    k1, x1=z[1]
    x2=x1+(x1-x0)*step_size
    k2=k1+(k1-k0)*step_size
    z2=np.array([k2, x2])
    return z2
    
def corrector(z, oldz,*args):
    data= oldz, step_size
    znew=fsolve(F, z, args=data)
    return znew

def initialize(f,z0,increment):
    z=np.zeros((2,2))
    c, x0=z0
    for a in range(2):
        data=c
        xnew, info, ier, mesg = fsolve(simple_cubic,x0,args=data,full_output=1)
        z[a]=[c,xnew]
        c+=increment    
    return z

n=100
start=-2
end=2

step_size=1
x0=1
c=start
increment=(end-start)/n
z0=np.array([c,x0])


#Generate initial 2 points
z=initialize(simple_cubic, z0,increment)


runs=0
while z[-1][0]<end:
    prevz=z[[-2,-1]]
    predz=predictor(prevz,step_size)
    nextz=corrector(predz,prevz)
    z=np.vstack((z,nextz))
    if runs%10000==0:
        print(z[-1][0])
    runs+=1

#ks=z[:,0]
#xs=z[:,1]
#plt.plot(ks,xs)

plt.plot(*z.transpose())
#FUCK YEAH IT WORKS



