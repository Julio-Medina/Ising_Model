#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:11:27 2023

@author: julio
This is a modified version of IsingModel2D.
In this version the external field of the Hamiltonian is considered.
Some optimization is done using numpy vectorization functions.
Animation of the MC process
"""


import matplotlib
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class isingModel(object): # Ising Model object

    def __init__(self,N,iterations,minTemp, maxTemp, JKb, HKb):
        self.N=N#Dimension del reticulo(red) 
        self.iterations=iterations#Numero de iteraciones
        #self.lattice=[]#np.ones([self.N,self.N])*-1.0#matriz N*N
        #self.initialLattice=[]#np.ones([self.N,self.N])*-1.0#inital lattice red inicial
        self.minTemp=minTemp
        self.maxTemp=maxTemp
        self.JKb=JKb
        self.HKb=HKb
        self.setInitialState()
        import matplotlib.pyplot as plotMvT
        self.plotMvT=plotMvT
        self.snapshots=[]

    def setInitialState(self):#establece el estado aleotorio inicial del reticulo (lattice)
        #self.intialLattice=self.lattice=[[random.choice([-1,1]) for x in range(self.N)] for y in range(self.N)]
        #self.initialLattice=self.lattice=[[-1.0 for x in range(self.N)] for y in range(self.N)] 
        self.lattice=np.ones([self.N, self.N])*1.0
        self.initialLattice=np.ones([self.N, self.N])*1.0
        ''' for i in range(self.N):
        for j in range(self.N):
        self.lattice[i][j]='''

    #calcula la energia de los vecinos mas cercanos
    # computes the energy of the nearest neighbors,
    # this computes the periodic 1D periodic version of the model
    def nearestNeighborEnergy(self,x,y):
        x1=x2=x3=x4=x
        y1=y2=y3=y4=y
        y1= self.N-1    if y==0         else y-1
        y2= 0           if y==self.N-1  else y+1
        x3= self.N-1    if x==0         else x-1
        x4= 0           if x==self.N-1  else x+1
        energy=-(self.lattice[x1][y1]+self.lattice[x2][y2]+self.lattice[x3][y3]+self.lattice[x4][y4])*self.lattice[x][y];
        return energy
    
    
    # computes the Hamiltonian for the Ising Model of an open Lattice in 2D
    def latticeEnergy(self):# 

        energy=0
        
        for i in range(self.N):
            for j in range(self.N):
                energy+=self.nearestNeighborEnergy(i,j)
        
        #energy+=self.lattice.sum()*self.HKb
        
        
             
        return energy

    # computes average magnetization per site
    def latticeMagnetization(self):
        M=0.0
        """
        for i in range(self.N):
            for j in range(self.N):
                M+=self.lattice[i][j]"""
                
        
        M=self.lattice.sum()
        return M/(self.N**2)

    def using_shifts(self):
        N=self.N
        shifter = np.arange(N*N).reshape( (N, N) )
        print(shifter)
        print([((1 >> shifter) % 2)*2-1 ])
        return [((x >> shifter) % 2)*2-1 for x in range(2 ** (N*N))]

    # runs the Monte Carlo simulation for the lattice at the given parameters
    def runMonteCarlo(self,T):
        #initial_energy=self.latticeEnergy()
        #deltas_E=[initial_energy]
        #snapshots=[]
        for i in range(self.iterations):
            randomX=random.randrange(self.N)
            randomY=random.randrange(self.N)
            #self.snapshots.append(self.lattice)
            energy=1*self.nearestNeighborEnergy(randomX,randomY)+1*self.lattice[randomX][randomY]*self.HKb
            #total_energy=self.latticeEnergy()
            #delta_energy=deltas_E[i]-energy
            #deltas_E.append(delta_energy)
            #if deltas_E[i+1]>0:
            if energy>0:
                self.lattice[randomX][randomY]*=-1
            elif math.exp(energy/T*self.JKb)>random.random() :
            #elif math.exp(deltas_E[i+1]/(T))>random.random() :
            #print(deltas_E[i+1])
            #if math.exp(-deltas_E[i+1]/(T))>random.random() :
                self.lattice[randomX][randomY]*=-1
                
                
    def animate_MC(self):
        import matplotlib.pyplot as plt2
        import matplotlib.animation as animation
        
        fig=plt2.figure(figsize=(self.N+10,self.N+10))
        a=self.snapshots[0]
        im=plt2.imshow(a)
        
        def animate_func(i):
            im.set_array(self.snapshots[i])
            return [im]
        fps=10
        anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = len(self.snapshots),
                               interval =1000/fps, # in ms
                               )
        anim.save('MC_animation_v04.mp4' ,fps=fps, extra_args=['-vcodec', 'libx264'])
        
    def plotLattice(self):
        s=1

    def plotMvrsT(self):
        X=[]
        Y=[]
        step=(self.maxTemp-self.minTemp)/1000.0
        T=self.minTemp
        #self.snapshots.append(self.lattice.copy())
        while (T<=self.maxTemp):
            #self.lattice=self.initialLattice.copy()
            aux=self.lattice.copy()
            self.snapshots.append(aux)
            self.runMonteCarlo(T)
            Y.append(self.latticeMagnetization())
            X.append(T)            
            T+=step
            
        self.plotMvT.cla()# limpia el interfaz de graficos
        #self.plotMvT.plot(X,Y,marker='o',color='blue',linewidth=2)
        self.plotMvT.ylim(-1.2,1.0)
        self.plotMvT.plot(X,Y,color='blue',linewidth=1)
        self.plotMvT.xlabel('T ')
        self.plotMvT.ylabel('M ')
        self.plotMvT.title('Magnetizacion(M) vrs. Temperatura(T)')
        self.plotMvT.savefig('PlotMvT_v04.png')
        
        
                
            
            
                
        
        
test=isingModel(50,10000,0.1,10,1,0);
#print(test.lattice)
'''print(test.nearestNeighborEnergy(0,0))
print(test.nearestNeighborEnergy(3,0))
print(test.nearestNeighborEnergy(0,3))
print(test.nearestNeighborEnergy(3,3))'''
#print('Energia: ',test.latticeEnergy())
#test.using_shifts()
#print(test.latticeMagnetization())
#test.runMonteCarlo(10)

#print(test.lattice)
#print(test.latticeMagnetization())
#init_lattice=test.lattice
test.plotMvrsT()     
#test.animate_MC()
#final_lattice=test.lattice
#print('Finished')
'''plt.arrow(10,10,10,10)
plt.draw()
plt.show()'''
snapshots=test.snapshots.copy()
fps = 10
nSeconds = 100

fig = plt.figure( figsize=(60,60) )

a = snapshots[0]
im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)

def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(snapshots[i])
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

anim.save('../animations/MC_animation_v04.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

print('Done!')

