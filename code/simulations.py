#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 21:11:01 2023

@author: julio
"""

from IsingModel2D_H import isingModel
import matplotlib.pyplot as plt
import numpy as np
import time
#def __init__(self,N,iterations,minTemp, maxTemp, JKb, HKb):
def time_size_simulation(min_lattice_size=10,
                         max_lattice_size=1000,
                         step=10,
                         iterations=10000,
                         minTemp=1,
                         maxTemp=20,
                         JKb=1,
                         HKb=1,
                         fixed_T=10):
    #min_lattice_size=10
    #max_lattice_size=1000
    #step=10
    #iterations=10000
    #minTemp=1
    #maxTemp=20
    #JKb=HKb=1
    
    times=[]
    final_magnetizations=[]
    size_range=range(min_lattice_size, max_lattice_size+step,step)
    for N in size_range:
        ising_model=isingModel(N, iterations, minTemp, maxTemp ,JKb, HKb )
        #initial_magnetization=ising_model.latticeMagnetization()
        start_time=time.time()
        ising_model.runMonteCarlo(fixed_T)
        final_magnetizations.append(ising_model.latticeMagnetization())
        end_time=time.time()
        delta_time=end_time-start_time
        times.append(delta_time)
    return list(size_range), times, final_magnetizations


def temperature_simulation(N=50,
                         num_steps=1000,
                         iterations=10000,
                         minTemp=1,
                         maxTemp=100,
                         JKb=1,
                         HKb=1):

    times=[]
    final_magnetizations=[]
    temperature_range=np.linspace(minTemp,maxTemp,num=num_steps)
    for T in temperature_range:
        ising_model=isingModel(N, iterations, minTemp, maxTemp ,JKb, HKb )
        #initial_magnetization=ising_model.latticeMagnetization()
        start_time=time.time()
        ising_model.runMonteCarlo(T)
        final_magnetizations.append(ising_model.latticeMagnetization())
        end_time=time.time()
        delta_time=end_time-start_time
        times.append(delta_time)
    return temperature_range, times, final_magnetizations


def MC_iterations_simulation(N=50,
                         min_iterations=100,
                         max_iterations=10000,
                         T=5,
                         #maxTemp=100,
                         JKb=1,
                         HKb=1):

    times=[]
    final_magnetizations=[]
    iteration_range=range(min_iterations, max_iterations, 100)
    for iterations in iteration_range:
        ising_model=isingModel(N, iterations, 10, 100 ,JKb, HKb )
        #initial_magnetization=ising_model.latticeMagnetization()
        start_time=time.time()
        ising_model.runMonteCarlo(T)
        final_magnetizations.append(ising_model.latticeMagnetization())
        end_time=time.time()
        delta_time=end_time-start_time
        times.append(delta_time)
    return iteration_range, times, final_magnetizations


"""
# Compute time vrs. lattice size simulation
lattice_size, times, magnetization=time_size_simulation(fixed_T=1,max_lattice_size=500)
plt.xlabel('Tamaño del retículo')
plt.ylabel('Tiempo de cómputo')
plt.plot(lattice_size,times)"""

"""
# Compute time vrs. Temperature

temperatures, times, magnetizations=temperature_simulation()    
plt.xlabel('Temperatura')
#plt.ylabel('Tiempo de cómputo')
#plt.plot(temperatures,times)
plt.ylabel('Magnetización promedio por sitio')
plt.plot(temperatures, magnetizations)    """

"""        
# Compute time vrs. MC iterations

iterations, times, magnetizations=MC_iterations_simulation()    
plt.xlabel('Número de iteraciones de Monte Carlo')
plt.ylabel('Tiempo de cómputo')
plt.plot(iterations,times)
#plt.ylabel('Magnetización promedio por sitio')
#plt.plot(temperatures, magnetizations)    """

"""
## J=1, H=0, T->[0.1,10] N=50, n=10000, parallel magnetization
simulation=isingModel(50,10000,0.1,10,1,0,simulation_name='sim1');
simulation.plotMvrsT()"""
"""
## J=1, H=0, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
simulation=isingModel(50,10000,0.1,10,1,0,initial_state=1.0,simulation_name='sim2');
simulation.plotMvrsT()"""

"""
## J=1, H=1, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
simulation=isingModel(50,10000,0.1,10,1,1,initial_state=1.0,simulation_name='sim3');
simulation.plotMvrsT()"""

"""
## J=1, H=-10, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
simulation=isingModel(50,10000,0.1,50,1,-10,initial_state=1.0,simulation_name='sim4');
simulation.plotMvrsT()#"""

"""
#J=1, H=0, T->[0.1,10] N=50, n=10000, parallel magnetization, plotTc=True
simulation=isingModel(50,10000,0.1,10,1,0,simulation_name='sim1_v02',plotTc=True);
simulation.plotMvrsT()#"""
"""
#J=1, H=0, T->[0.1,10] N=500, n=10000, parallel magnetization, plotTc=True
simulation=isingModel(500,10000,0.1,10,1,0,simulation_name='sim1_v03',plotTc=True);
simulation.plotMvrsT()#"""
"""
#J=1, H=0, T->[0.1,10] N=500, n=10000, parallel magnetization, plotTc=True
simulation=isingModel(500,10000,0.1,4,0.5,0,simulation_name='sim1_v04',plotTc=True);
simulation.plotMvrsT()#"""
"""
## J=10, H=-1, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
simulation=isingModel(1000,1000000,0.1,50,5,0,initial_state=-1.0,simulation_name='sim5',plotTc=True);
simulation.plotMvrsT()#"""
"""
## J=10, H=-1, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
simulation=isingModel(1000,100000,0.1,50,2,0,initial_state=-1.0,simulation_name='sim6',plotTc=True);
simulation.plotMvrsT()#"""
"""
#J=10, H=-1, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
simulation=isingModel(5000,5000000,0.1,50,2,0,initial_state=-1.0,simulation_name='sim7',plotTc=True);
simulation.plotMvrsT()#"""
"""
#J=10, H=-1, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
simulation=isingModel(5000,5000000,0.1,12,2,0,initial_state=-1.0,simulation_name='sim8',plotTc=True);
simulation.plotMvrsT()#"""
"""
#J=10, H=-1, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
start_time=time.time()
simulation=isingModel(5000,5000000,0.1,15,3,0,initial_state=-1.0,simulation_name='sim9',plotTc=True);
simulation.plotMvrsT()
end_time=time.time()
delta_time=end_time-start_time
times.append(delta_time)
#"""
"""
#J=10, H=-1, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
start_time=time.time()
simulation=isingModel(5000,5000000,0.1,20,4,0,initial_state=-1.0,simulation_name='sim10',plotTc=True);
simulation.plotMvrsT()
end_time=time.time()
delta_time=end_time-start_time
times.append(delta_time)
#"""
"""
#J=10, H=-1, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
start_time=time.time()
simulation=isingModel(10000,8000000,0.1,20,4,0,initial_state=-1.0,simulation_name='sim11',plotTc=True);
simulation.plotMvrsT()
end_time=time.time()
delta_time=end_time-start_time
times.append(delta_time)
#"""

## J=1, H=-10, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
simulation=isingModel(50,10000,0.1,300,1,30,initial_state=-1.0,simulation_name='sim12_v01');
simulation.plotMvrsT()#"""
"""
## J=1, H=-10, T->[0.1,10] N=50, n=10000, initial state=1(different magnetization pole, parallel)
simulation=isingModel(50,10000,0.1,50,2,1,initial_state=-1.0,simulation_name='sim13');
simulation.plotMvrsT()#"""
