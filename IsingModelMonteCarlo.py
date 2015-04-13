# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:40:08 2015

@author: oyvind
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
    
def totalEnergy(lattice, Jx, Jy):
    #lattice is an Nx x Ny shaped array consisting solely of -1s and 1s (spin down and spin up)
    #Jx is the coupling constant in the x direction
    #Jy is the coupling constant in the y direction
    #Note that x direction is in columns, y direction is in rows
    Nx = len(lattice[0])
    Ny = len(lattice)
    Etot = 0
    spinTot = 0
    for i in np.arange(Nx):
        for j in np.arange(Ny):
            Etot -= lattice[j, i] * (Jx * (lattice[j, i-1] + lattice[j, (i+1)%Nx]) + Jy * (lattice[j-1, i] + lattice[(j+1)%Ny, i]))
            spinTot += lattice[j, i]
    Etot = Etot/2 #Divide by two to account for double counting
    return Etot, spinTot
    
def deltaE(lattice, Jx, Jy, x, y):
    #This function calculates the change of energy by flipping the spin at site (x, y)
    Nx = len(lattice[0])
    Ny = len(lattice)
    Ebefore = -lattice[y, x] * (Jx * (lattice[y, x-1] + lattice[y, (x+1)%Nx]) + Jy * (lattice[y-1, x] + lattice[(y+1)%Ny, x]))
    return -2*Ebefore
    
def MonteCarlo_Sweep(lattice, Jx, Jy, T):
    Nx = len(lattice[0])
    Ny = len(lattice)
    x = np.random.randint(0, Nx)
    y = np.random.randint(0, Ny)
    delE = deltaE(lattice, Jx, Jy, x, y)
    randnr=np.random.rand()
    if randnr < np.exp(-delE/T):
        lattice[y, x] = -lattice[y, x]
        return lattice
    else:
        return lattice
    
    
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
    
def groundstate_lattice(Nx, Ny):
    return np.ones((Ny, Nx))
    
#def MonteCarlo_Animation(lattice, Jx, Jy, T, N_iterations):
#    images = []
#    new_lattice = lattice.copy()
#    #image = plt.imshow(new_lattice, cmap = "Greys" , vmin=-1, vmax=1, interpolation='None')
#    images.append(new_lattice)
#    #fig = plt.figure()
#    for i in np.arange(N_iterations):
#        new_lattice = MonteCarlo_Sweep(new_lattice, Jx, Jy, T)
#        #image = plt.imshow(new_lattice, cmap = "Greys" , vmin=-1, vmax=1, interpolation='None')
#        images.append(new_lattice)
#    return images
#    
#def MonteCarlo_Anim2(lattice, Jx, Jy, T, N_iterations):
#    new_lattice = lattice.copy()
#    plt.subplot(111)
#    for i in np.arange(N_iterations):
#        new_lattice = MonteCarlo_Sweep(new_lattice, Jx, Jy, T)
#        plt.imshow(new_lattice, cmap = "Greys" , vmin=-1, vmax=1, interpolation='None')

def Plot_Magnetization(lattice, Jx, Jy, N_iterations):
    meanMagn = []
    meanMagn2 = []
    meanE = []
    meanE2 = []
    Tvec = np.linspace(0.1, 5, num = 100)
    meanFrom = int((N_iterations)/2) #What part of data to take the mean of. Don't want to include beginning as it's not in equilibrium
    for T in Tvec:
        print(T) #Calculations take a long time. Ok to see where in the process it is
        totalE, spinTot = MonteCarlo_Plot(lattice, Jx, Jy, T, N_iterations, Plot=False)
        meanMagn.append(np.mean(spinTot[-meanFrom:])/(len(lattice)*len(lattice[0])))
        meanMagn2.append(np.mean(spinTot[-meanFrom:]*spinTot[-meanFrom:])/(len(lattice)*len(lattice[0]))**2)
        meanE.append(np.mean(totalE[-meanFrom:])/(len(lattice)*len(lattice[0])))
        meanE2.append(np.mean(totalE[-meanFrom:]*totalE[-meanFrom:])/(len(lattice)*len(lattice[0]))**2)
    Susceptibility = [(meanMagn2[i]-meanMagn[i]**2)/Tvec[i]**2 for i in np.arange(len(Tvec))]
    Cv = [(meanE2[i]-meanE[i]**2)/Tvec[i] for i in np.arange(len(Tvec))]
    fig1, ax1 = plt.subplots()
    ax1.scatter(Tvec, np.abs(meanMagn))
    ax1.set_title('Magnetization')
    fig2, ax2 = plt.subplots()
    ax2.scatter(Tvec, meanE)
    ax2.set_title('Energy')
    fig3, ax3 = plt.subplots()
    ax3.scatter(Tvec, Cv)
    ax3.set_title('Heat Capacity')
    fig4, ax4 = plt.subplots()
    ax4.scatter(Tvec, Susceptibility)
    ax4.set_title('Susceptibility')
    np.savetxt('MeanMagn5x5Nit50000.txt', meanMagn, delimiter=',')
    np.savetxt('Mean2Magn5x5Nit50000.txt', meanMagn2, delimiter=',')
    np.savetxt('MeanE5x5Nit50000.txt', meanE, delimiter=',')
    np.savetxt('Mean2E5x5Nit50000.txt', meanE2, delimiter=',')
#    fig, ax = plt.subplots()
#    ax.plot(Tvec, np.abs(meanMagn), 'bo')
    
def MonteCarlo_Plot(lattice, Jx, Jy, T, N_iterations, Plot=True):
    spinTot = []
    totalE = []
    new_lattice = lattice.copy()
    Etot_i, spinTot_i = totalEnergy(new_lattice, Jx, Jy)
    totalE.append(Etot_i)
    spinTot.append(spinTot_i)
    for i in np.arange(N_iterations):
        new_lattice = MonteCarlo_Sweep(new_lattice, Jx, Jy, T)
        Etot_i, spinTot_i = totalEnergy(new_lattice, Jx, Jy)
        totalE.append(Etot_i)
        spinTot.append(spinTot_i)
        #spinTot.append(sum([sum(new_lattice[i]) for i in np.arange(len(lattice))]))
    if Plot:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(211)
        ax1.imshow(lattice, cmap = "Greys" , vmin=-1, vmax=1, interpolation='None')
        ax2 = fig1.add_subplot(212)
        ax2.imshow(new_lattice, cmap = "Greys" , vmin=-1, vmax=1, interpolation='None')
        plt.show()
        fig3, ax3 = plt.subplots()
        ax3.plot(np.arange(N_iterations+1), spinTot)
#    meanFrom = int((N_iterations)/5)
    return np.array(totalE), np.array(spinTot)
    
chessboard = initialize_chessboard_lattice(50, 50)
random_lattice = initialize_random_lattice(5, 5)
gs_lattice = groundstate_lattice(50, 50)
#figure, axis = plt.subplots()
#plt.imshow(random_lattice, cmap='Greys', interpolation='nearest')

#fig, ax = plt.subplots()
#images=MonteCarlo_Animation(chessboard, -1, -1, 0.08, 100000)
#
#def imagesFunc(i):
#    return images[i]
#
#ani = animation.FuncAnimation(fig, imagesFunc, interval=5000, blit=True, repeat_delay=1000)
#plt.show()

#MonteCarlo_Plot(random_lattice, 1, 1, 2, 20000)
Plot_Magnetization(random_lattice, 1, 1, 50000)

#meanMagn5x5=np.loadtxt('MeanMagn5x5Nit50000.txt', delimiter=',')
