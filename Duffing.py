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
import math

import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy import stats


#def dudt(u,t): this is a bad name


#used to have seperate dudt functions combined into one
def duffy(u0,t, pars):
    #takes a vector,returns a vector
    gamma, epsilon=pars
    u=u0
    du=[u[1], gamma*np.sin(omega*t)-2*epsilon*u[1]-u[0]-u[0]**3]
    return du    

def start_end_diff(u,*data):
    #this shall be our function to zero
    t, T, f, pars =data
    uinitial=u
    us=odeint(f,uinitial,[t,t+T], args=(pars,))
    ufinal=us[-1] # this is the last element
    f=uinitial-ufinal
    #df=np.array([f[1],du_dt(uinitial,t,gamma,epsilon)[1]-du_dt(ufinal,t+T,gamma,epsilon)[1]])
    #Ans=np.array([f, df])
    return f

def plot_stuff(f, u, t, pars):
    Us=odeint(f,u,t, args=(pars,))
    us=Us[:,0]    
    dus=Us[:,1]

    
    fig1=plt.figure(figsize=(10,5))
    ax1=fig1.add_subplot(1,1,1)
    ax1.set_xlabel("t")
    ax1.set_ylabel("u")
    #plt.title("Duffing")
    """
    ax1.set_xticks([np.pi*100*t for t in range(200)])
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
    """
    ax1.plot(ts,us)
    ax1.plot(ts,dus)
    #fig1.savefig('Duffing Plot.svg')
    
    fig2=plt.figure(figsize=(10,5))
    ax2=fig2.add_subplot(1,1,1)
    ax2.plot(us,dus)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Velocity")
    #fig2.savefig('Phase Plot.svg')
 
    

if __name__ == "__main__":
    epsilon=0.05
    gamma=0.2
    #Do i just set any value of omega
    omega=1.0
    #Do I just set initial conditions
    U0=np.array([0.5, 0]) # max amplitude 0 velocity seem like good starting points
    t0=0
    pars= [gamma, epsilon]
    #WELL SHIT
    Us=odeint(duffy,U0,[0,1000], args=(pars,))
    
    
    ts=np.linspace(0, 1000, 1000000)
    # the time spacings have to be very small too. I think the sin screws things up a lot
    #changed time scale to not depend on pi its easier this way
    #ts=np.array([np.pi*z/1000 for z in range(1000000)])
    plot_stuff(duffy, U0, ts, pars)
   
    
    T=2*np.pi/omega
    
    
    u=U0
    u=[1,-1]
    t=t0
    
    data = t, T, duffy, pars
    x, infodict, ier, mesg=sp.optimize.fsolve(start_end_diff, u, args=data, full_output=1 )
    
    if(ier==1):
        print(mesg)
        print("The roots are {}".format(x))
    else:
        print(mesg)
    
    ts=np.linspace(t, T, 1000000)
    
    plot_stuff(duffy,x,ts,pars)
    
    """   
    
    #Isolate a period T
    periods=[]
    for index in range(len(Us)):
        A=Us[index]-Us[750000]
        if(np.sqrt(np.dot(A,A))/2<=0.00005):
            periods.append(index)
    
    print(periods)
    
    for tm in periods:
        print(Us[tm])
    last=0
    for tm in periods[:]:
        print(ts[tm]-last)
        last=ts[tm]
    #bad but okay way to check period
    #it does change to 2pi/omega when I changed omega YES IT WORKS
    
    fig3=plt.figure(figsize=(10,5))
    ax3=fig3.add_subplot(1,1,1)
    ax3.plot(ts[742000:758000],us[742000:758000])
    ax3.plot(ts[742000:758000],dus[742000:758000])
    ax3.set_xlabel("t")
    ax3.set_ylabel("u")    
    """