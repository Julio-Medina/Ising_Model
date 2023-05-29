#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 21:11:01 2023

@author: julio
"""

from IsingModel2D_H import isingModel
import matplotlib.pyplot as plt
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
        initial_magnetization=ising_model.latticeMagnetization()
        start_time=time.time()
        ising_model.runMonteCarlo(fixed_T)
        final_magnetizations.append(ising_model.latticeMagnetization)
        end_time=time.time()
        delta_time=end_time-start_time
        times.append(delta_time)
    return list(size_range), times, final_magnetizations

lattice_size, times, magnetization=time_size_simulation(fixed_T=1,max_lattice_size=200)
plt.plot(lattice_size,times)


        
        
    
    

