# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:40:08 2015

@author: oyvind
"""

import numpy as np
import matplotlib.pyplot as plt
    
def totalEnergy(lattice, Jx, Jy, h):
    #Calculates total energy and total magnetization of a lattice configuration
    #lattice is an Nx x Ny shaped array consisting solely of -1s and 1s (spin down and spin up)
    #Jx is the coupling constant in the x direction
    #Jy is the coupling constant in the y direction
    #h is external magnetic field. Positive value is along the spin up direction
    #Note that x direction is in columns, y direction is in rows
    Nx = len(lattice[0])
    Ny = len(lattice)
    Etot = 0
    spinTot = 0
    for i in np.arange(Nx):
        for j in np.arange(Ny):
            Etot -= lattice[j, i] * (Jx * (lattice[j, i-1] + lattice[j, (i+1)%Nx]) + Jy * (lattice[j-1, i] + lattice[(j+1)%Ny, i]) + h)
            spinTot += lattice[j, i]
    Etot = Etot/2 #Divide by two to account for double counting
    return Etot, spinTot
    
def deltaE(lattice, Jx, Jy, h, x, y):
    #This function calculates the change of energy by flipping the spin at site (x, y)
    Nx = len(lattice[0])
    Ny = len(lattice)
    Ebefore = -lattice[y, x] * (Jx * (lattice[y, x-1] + lattice[y, (x+1)%Nx]) + Jy * (lattice[y-1, x] + lattice[(y+1)%Ny, x]) + h)
    return -2*Ebefore
    
def MonteCarlo_Sweep(lattice, Jx, Jy, h, T):
    #Makes a trial perturbation on the lattice
    #Based on the energy change of the perturbation, either accepts or rejects the new system
    Nx = len(lattice[0])
    Ny = len(lattice)
    x = np.random.randint(0, Nx)
    y = np.random.randint(0, Ny)
    delE = deltaE(lattice, Jx, Jy, h, x, y)
    randnr=np.random.rand()
    if randnr < np.exp(-delE/T):
        lattice[y, x] = -lattice[y, x]
        return lattice
    else:
        return lattice
        
def Metropolis_Hastings(lattice, Jx, Jy, h, T, N_iterations, Plot=True):
    #Does N_iterations Monte Carlo sweeps
    #Returns the total energy and magnetization of the final state if Plot is false
    #If Plot is true, plots the initial & final system, as well as magnetization as a function of iterations
    new_lattice = lattice.copy()    
    spinTot = []
    if Plot:
        spinTot.append(sum([sum(new_lattice[i]) for i in np.arange(len(lattice))]))
    else:
        totalE = []
        Etot_i, spinTot_i = totalEnergy(new_lattice, Jx, Jy, h)
        totalE.append(Etot_i)
        spinTot.append(spinTot_i)
    for i in np.arange(N_iterations):
        new_lattice = MonteCarlo_Sweep(new_lattice, Jx, Jy, h, T)
        if Plot:
            spinTot.append(sum([sum(new_lattice[i]) for i in np.arange(len(lattice))]))
        else:
            Etot_i, spinTot_i = totalEnergy(new_lattice, Jx, Jy, h)
            totalE.append(Etot_i)
            spinTot.append(spinTot_i)   
    if Plot:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(211)
        ax1.imshow(lattice, cmap = "Greys" , vmin=-1, vmax=1, interpolation='None')
        ax1.axes.get_xaxis().set_ticks([])
        ax1.axes.get_yaxis().set_ticks([])
        ax1.set_title('Initial lattice configuration')
        ax2 = fig1.add_subplot(212)
        ax2.imshow(new_lattice, cmap = "Greys" , vmin=-1, vmax=1, interpolation='None')
        ax2.axes.get_xaxis().set_ticks([])
        ax2.axes.get_yaxis().set_ticks([])
        ax2.set_title('Final lattice configuration')
        plt.show()
        fig3, ax3 = plt.subplots()
        ax3.plot(np.arange(N_iterations+1), spinTot)
        ax3.set_xlabel('Number of iterations')
        ax3.set_ylabel('Total spin of system')
        ax3.set_title('Magnetization as a function of iterations')
    else:
        return np.array(totalE), np.array(spinTot)

def MagnetizationAndEnergy_TempFunc(lattice, Jx, Jy, h, N_iterations):
    meanMagn = []
    meanMagn2 = []
    meanE = []
    meanE2 = []
    Tvec = np.linspace(0.1, 5, num = 100)
    meanFrom = int((N_iterations)/2) #What part of data to take the mean of. Don't want to include beginning as it's not in equilibrium
    for T in Tvec:
        print(T) #Calculations take a long time. Ok to see where in the process it is
        totalE, spinTot = Metropolis_Hastings(lattice, Jx, Jy, h, T, N_iterations, Plot=False)
        meanMagn.append(np.mean(spinTot[-meanFrom:])/(len(lattice)*len(lattice[0])))
        meanMagn2.append(np.mean(spinTot[-meanFrom:]*spinTot[-meanFrom:])/(len(lattice)*len(lattice[0]))**2)
        meanE.append(np.mean(totalE[-meanFrom:])/(len(lattice)*len(lattice[0])))
        meanE2.append(np.mean(totalE[-meanFrom:]*totalE[-meanFrom:])/(len(lattice)*len(lattice[0]))**2)
    #Susceptibility = [(meanMagn2[i]-meanMagn[i]**2)/Tvec[i]**2 for i in np.arange(len(Tvec))]
    #Cv = [(meanE2[i]-meanE[i]**2)/Tvec[i] for i in np.arange(len(Tvec))]
    np.savetxt('MeanMagn15x15Nit225000.txt', meanMagn, delimiter=',')
    np.savetxt('Mean2Magn15x15Nit225000.txt', meanMagn2, delimiter=',')
    np.savetxt('MeanE15x15Nit225000.txt', meanE, delimiter=',')
    np.savetxt('Mean2E15x15Nit225000.txt', meanE2, delimiter=',')
    
def initialize_chessboard_lattice(Nx, Ny):
    lattice = np.ones((Ny, Nx))
    for i in np.arange(Nx):
        for j in np.arange(Ny):
            if (i+j)%2 == 1:
                lattice[j, i] = -1
    return lattice
    
def initialize_random_lattice(Nx, Ny):
    lattice = (2.0*np.random.randint(2, size=Nx*Ny)-1.0).reshape(Ny,Nx)
    return lattice
    
def initialize_groundstate_lattice(Nx, Ny):
    return np.ones((Ny, Nx))
    
chessboard = initialize_chessboard_lattice(50, 50)
random_lattice = initialize_random_lattice(30, 30)
gs_lattice = initialize_groundstate_lattice(20, 20)

#Metropolis_Hastings(random_lattice, -1, -1, 4, 0.8, 10000)
#MagnetizationAndEnergy_TempFunc(random_lattice, 1, 1, 225000)

meanMagn5x5=np.loadtxt('MeanMagn5x5Nit50000.txt', delimiter=',')
meanMagn10x10=np.loadtxt('MeanMagn10x10Nit200000.txt', delimiter=',')
meanMagn15x15=np.loadtxt('MeanMagn15x15Nit225000.txt', delimiter=',')
mean2Magn5x5=np.loadtxt('Mean2Magn5x5Nit50000.txt', delimiter=',')
mean2Magn10x10=np.loadtxt('Mean2Magn10x10Nit200000.txt', delimiter=',')
mean2Magn15x15=np.loadtxt('Mean2Magn15x15Nit225000.txt', delimiter=',')
meanE5x5=np.loadtxt('MeanE5x5Nit50000.txt', delimiter=',')
meanE10x10=np.loadtxt('MeanE10x10Nit200000.txt', delimiter=',')
meanE15x15=np.loadtxt('MeanE15x15Nit225000.txt', delimiter=',')
meanE25x5=np.loadtxt('Mean2E5x5Nit50000.txt', delimiter=',')
meanE210x10=np.loadtxt('Mean2E10x10Nit200000.txt', delimiter=',')
meanE215x15=np.loadtxt('Mean2E15x15Nit225000.txt', delimiter=',')

Tvec = np.linspace(0.1, 5, num = 100)
#Tvec2 = np.linspace(0.1, 5, num = 100000)

Cv5x5 = [(meanE25x5[i]-meanE5x5[i]**2)/Tvec[i] for i in np.arange(len(Tvec))]
Cv10x10 = [(meanE210x10[i]-meanE10x10[i]**2)/Tvec[i] for i in np.arange(len(Tvec))]
Cv15x15 = [(meanE215x15[i]-meanE15x15[i]**2)/Tvec[i] for i in np.arange(len(Tvec))]

Susc5x5 = [(mean2Magn5x5[i]-meanMagn5x5[i]**2)/Tvec[i]**2 for i in np.arange(len(Tvec))]
Susc10x10 = [(mean2Magn10x10[i]-meanMagn10x10[i]**2)/Tvec[i]**2 for i in np.arange(len(Tvec))]
Susc15x15 = [(mean2Magn15x15[i]-meanMagn15x15[i]**2)/Tvec[i]**2 for i in np.arange(len(Tvec))]

#theoryMag = [(1-1/np.sinh(2/x)**4)**(1/8) for x in Tvec2]
fig, ax = plt.subplots()
#ax.plot(Tvec, abs(meanMagn5x5), 'ro', label='5x5')
#ax.plot(Tvec, abs(meanMagn10x10), 'bo', label='10x10')
ax.plot(Tvec, abs(meanMagn15x15), 'go', label='15x15')
#ax.plot(Tvec2, theoryMag, 'b')
#ax.plot(Tvec, 25*np.array(Susc5x5), 'ro', label='5x5')
#ax.plot(Tvec, 100*np.array(Susc10x10), 'bo', label='10x10')
#ax.plot(Tvec, 225*np.array(Cv15x15), 'go', label='15x15')
ax.legend()
