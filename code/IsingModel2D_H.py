#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 18:31:03 2023

@author: julio
This is a modified version of IsingModel2D.
In this version the external field of the Hamiltonian is considered.
Some optimization is done using numpy vectorization functions.
"""


import matplotlib
import random
import numpy as np
import math
import matplotlib.pyplot as plt

class isingModel(object): # Ising Model object

    def __init__(self,N,iterations,minTemp, maxTemp, JKb, HKb):
        self.N=N#Dimension del reticulo(red) 
        self.iterations=iterations#Numero de iteraciones
        self.lattice=np.ones([self.N,self.N])#matriz N*N
        self.initialLattice=np.ones([self.N,self.N])#inital lattice red inicial
        self.minTemp=minTemp
        self.maxTemp=maxTemp
        self.JKb=JKb
        self.HKb=HKb
        self.setInitialState()
        import matplotlib.pyplot as plotMvT
        self.plotMvT=plotMvT

    def setInitialState(self):#establece el estado aleotorio inicial del reticulo (lattice)
        #self.intialLattice=self.lattice=[[random.choice([-1,1]) for x in range(self.N)] for y in range(self.N)]
        #self.intialLattice=self.lattice=[[-1 for x in range(self.N)] for y in range(self.N)] 
        self.intialLattice=np.ones([self.N, self.N])*-1.0
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
        
        energy+=self.lattice.sum()*self.HKb
        
        
             
        return energy

    # computes average magnetization per site
    def latticeMagnetization(self):
        M=0.0
        """
        for i in range(self.N):
            for j in range(self.N):
                M+=self.lattice[i][j]
                
        """
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
        initial_energy=self.latticeEnergy()
        deltas_E=[initial_energy]
        
        for i in range(self.iterations):
            randomX=random.randrange(self.N)
            randomY=random.randrange(self.N)
            energy=-1*self.nearestNeighborEnergy(randomX,randomY)-1*self.lattice[randomX][randomY]*self.HKb
            #energy=self.latticeEnergy()
            delta_energy=deltas_E[i]-energy
            deltas_E.append(delta_energy)
            if deltas_E[i+1]>0:
                self.lattice[randomX][randomY]*=-1
            #elif math.exp(energy/(T*self.JKb))>random.random() :
            elif math.exp(deltas_E[i+1]/(T))>random.random() :
            #print(deltas_E[i+1])
            #if math.exp(-deltas_E[i+1]/(T))>random.random() :
                self.lattice[randomX][randomY]*=-1
                
    def plotLattice(self):
        s=1

    def plotMvrsT(self):
        X=[]
        Y=[]
        step=(self.maxTemp-self.minTemp)/950.0
        T=self.minTemp
        while (T<=self.maxTemp):
            self.lattice=self.intialLattice
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
        self.plotMvT.savefig('PlotMvT.png')
        
        
                
            
            
                
        
        
#test=isingModel(100,15000,1,20,5,10);
#print(test.lattice)
'''print(test.nearestNeighborEnergy(0,0))
print(test.nearestNeighborEnergy(3,0))
print(test.nearestNeighborEnergy(0,3))
print(test.nearestNeighborEnergy(3,3))'''
#print('Energia: ',test.latticeEnergy())
#test.using_shifts()
#print(test.latticeMagnetization())
#test.runMonteCarlo(100)
#print(test.lattice)
#print(test.latticeMagnetization())
#test.plotMvrsT()     
#print('Finished')
'''plt.arrow(10,10,10,10)
plt.draw()
plt.show()'''
