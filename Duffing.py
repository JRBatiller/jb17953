# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:03:47 2017

@author: Joby
"""
"""
Simulate the nonlinear Duffing equation u¨+2ξ u˙+u+u^3 = Γsin(ωt) 
for ξ = 0.05, Γ = 0.2 and 0.5 ≤ ω ≤ 1.5.
• What behaviour do you see in the long-time limit?
• Isolate a periodic orbit. What are its starting conditions? What is its period?
1
• This will provide testing data for your numerical methods.
– Always test your code against known results or you will come unstuck!
"""

"""
we will let u1=u'
and u2=u''
"""

import numpy as np
import scipy as sp
import time

import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import stats


#def du1dt(u,t): this is a bad name

def du1_dt(u1,u2,t):
    du1=u2
    return du1

def du2_dt(u1,u2,t):
    du2=gamma*np.sin(omega*t)-2*epsilon*u2-u1**3    
    return du2

def du_dt(u,t):
    du=[u[1], gamma*np.sin(omega*t)-2*epsilon*u[1]-u[0]**3]
    return du    
    

epsilon=0.05
gamma=0.2
#Do i just set any value of omega
omega=1.0
#Do I just set initial conditions
U0=np.array([0,0])
ts=np.linspace(0,50*np.pi,200)
Us=odeint(du_dt,U0,ts)
ys=Us[:,0]    

plt.xlabel("t")
plt.ylabel("u")
plt.title("Duffing")
plt.plot(ts,ys);