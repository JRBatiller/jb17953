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


#def dudt(u,t): this is a bad name



def du_dt(u,t):
    #takes a vector,returns a vector
    du=[u[1], gamma*np.sin(omega*t)-2*epsilon*u[1]-u[0]-u[0]**3]
    return du    
    
if __name__ == "__main__":
    epsilon=0.05
    gamma=0.2
    #Do i just set any value of omega
    omega=1.0
    #Do I just set initial conditions
    U0=np.array([0,0])
    ts=np.linspace(0,1000*np.pi,200)
    Us=odeint(du_dt,U0,ts)
    us=Us[:,0]    
    dus=Us[:,1]


    fig1=plt.figure()
    ax1=fig1.add_subplot(1,1,1)
    ax1.set_xlabel("t")
    ax1.set_ylabel("u")
    #plt.title("Duffing")
    ax1.set_xticks([np.pi*100*t for t in range(20)])
    ax1.set_xticklabels([
        r'$0$',
        r'$100\pi$',
        r'$200\pi$',
        r'$300\pi$',
        r'$400\pi$',
        r'$500\pi$',
        r'$600\pi$',
        r'$700\pi$',
        r'$800\pi$',
        r'$900\pi$',
        r'$1000\pi$',
    ], fontsize='medium')
    ax1.plot(ts,us)
    ax1.plot(ts,dus)
    fig1.savefig('Duffing Plot.svg')