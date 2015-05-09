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
    if np.random.rand() < np.exp(-delE/T):
        lattice[y, x] = -lattice[y, x]
    return lattice
        
def Metropolis_Hastings(lattice, Jx, Jy, h, T, N_equilibrium, N_sampling, Plot=True):
    #N_equilibrium is the amount of iterations to equilibrate the initial system
    #N_sampling is the amount of iterations used to get data samples from the equilibrated system
    #Does N_iterations Monte Carlo sweeps
    #Returns the total energy and magnetization of the final state if Plot is false
    #If Plot is true, plots the initial & final system, as well as magnetization as a function of iterations
    new_lattice = lattice.copy()
    spinTot = []
    N_iterations = N_equilibrium + N_sampling
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
        elif i >= N_equilibrium:
            Etot_i, spinTot_i = totalEnergy(new_lattice, Jx, Jy, h)
            totalE.append(Etot_i)
            spinTot.append(spinTot_i)
    if Plot:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        ax1.imshow(lattice, cmap = "Greys" , vmin=-1, vmax=1, interpolation='None')
        ax1.axes.get_xaxis().set_ticks([])
        ax1.axes.get_yaxis().set_ticks([])
        ax1.set_title('Initial lattice configuration')
        ax2 = fig1.add_subplot(122)
        im = ax2.imshow(new_lattice, cmap = "Greys" , vmin=-1, vmax=1, interpolation='None')
        ax2.axes.get_xaxis().set_ticks([])
        ax2.axes.get_yaxis().set_ticks([])
        ax2.set_title('Final lattice configuration')
        fig1.subplots_adjust(right=0.8)
        cbar_ax = fig1.add_axes([0.85, 0.293, 0.05, 0.41])
        fig1.colorbar(im, cax=cbar_ax, ticks=[-1, 1])
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(np.arange(N_iterations+1), spinTot)
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('Total spin of system')
        ax.set_title('Magnetization as a function of iterations')
    else:
        return np.array(totalE), np.array(spinTot)

def MagnetizationAndEnergy_TempFunc(lattice, Jx, Jy, h, Tvec, N_equilibrium, N_sampling):
    meanMagn = []
    meanMagn2 = []
    meanE = []
    meanE2 = []
    for T in Tvec:
        print(T) #Calculations take a long time. Ok to see where in the process it is
        totalE, spinTot = Metropolis_Hastings(lattice, Jx, Jy, h, T, N_equilibrium, N_sampling, Plot=False)
        meanMagn.append(np.mean(spinTot)/(len(lattice)*len(lattice[0])))
        meanMagn2.append(np.mean(spinTot*spinTot)/(len(lattice)*len(lattice[0]))**2)
        meanE.append(np.mean(totalE)/(len(lattice)*len(lattice[0])))
        meanE2.append(np.mean(totalE*totalE)/(len(lattice)*len(lattice[0]))**2)
    #Susceptibility = [(meanMagn2[i]-meanMagn[i]**2)/Tvec[i]**2 for i in np.arange(len(Tvec))]
    #Cv = [(meanE2[i]-meanE[i]**2)/Tvec[i] for i in np.arange(len(Tvec))]
    Nx = len(lattice[0])
    Ny = len(lattice)
    N_iterations = N_equilibrium + N_sampling
    np.savetxt('MeanMagn' + str(Nx) + 'x' + str(Ny) + 'Nit' + str(N_iterations) + '.txt', meanMagn, delimiter=',')
    np.savetxt('Mean2Magn' + str(Nx) + 'x' + str(Ny) + 'Nit' + str(N_iterations) + '.txt', meanMagn2, delimiter=',')
    np.savetxt('MeanE' + str(Nx) + 'x' + str(Ny) + 'Nit' + str(N_iterations) + '.txt', meanE, delimiter=',')
    np.savetxt('Mean2E' + str(Nx) + 'x' + str(Ny) + 'Nit' + str(N_iterations) + '.txt', meanE2, delimiter=',')
    
def initialize_groundstate_lattice(Nx, Ny):
    return np.ones((Ny, Nx))

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
    
def HeatCapacity(meanEnergy, meanEnergy2, Tvec, Nsize):
    #meanEnergy is an array containing the values <E>
    #meanEnergy2 is an array containing the values <E^2>
    #Tvec is an array with the temperature corresponding to each entry in meanEnergy and meanEnergy2
    #Nsize is how many particles there are in the system
    #Returns the heat capacity per particle as a function of temperature
    mE = np.array(meanEnergy)
    mE2 = np.array(meanEnergy2)
    Cv = [Nsize*(mE2[i]-mE[i]**2)/Tvec[i]**2 for i in np.arange(len(Tvec))]
    return np.array(Cv)
    
def MagneticSusceptibility(meanMagn, meanMagn2, Tvec, Nsize):
    #meanMagnetization is an array containing the values <M>
    #meanMagnetization2 is an array containing the values <M^2>
    #Returns the magnetic susceptibility per particle as a function of temperature
    mM = np.array(meanMagn)
    mM2 = np.array(meanMagn2)
    Chi = [Nsize*(mM2[i]-mM[i]**2)/Tvec[i] for i in np.arange(len(Tvec))]
    return np.array(Chi)


#Initialization of different lattices one can use
chessboard = initialize_chessboard_lattice(10, 10)
random_lattice = initialize_random_lattice(10, 10)
gs_lattice = initialize_groundstate_lattice(20, 20)

#Initialization of temperature values to plot for
Tvec = np.linspace(0.1, 5, num = 100)

#Example of the Metropolis-Hastings algorithm at a temperature T=0.2K
#Should produce a plot of initial and final lattice, and a plot of the magnetization as a function of iterations
Metropolis_Hastings(random_lattice, 1, 1, 0, 0.2, 1000, 1000, Plot=True)

#To run the main code that produces the data necessary for plotting of energy, magnetization, heat capacity and magnetic susceptibility
#as a function of temperature, uncomment the line below. NOTE: This will take a LONG time!
#MagnetizationAndEnergy_TempFunc(random_lattice, 1, 1, 0, Tvec, 100000, 125000)

#In the github repository at https://github.com/oyvindjohansen/NE-155/tree/master/FinalProject
#one can find the following .txt files with data for 5x5, 10x10, 15x15 lattices.
#Add these files to the same directory as the .py file to load the data with the code below:
meanMagn5x5=np.loadtxt('MCDataFiles/MeanMagn5x5Nit50000.txt', delimiter=',')
meanMagn10x10=np.loadtxt('MCDataFiles/MeanMagn10x10Nit200000.txt', delimiter=',')
meanMagn15x15=np.loadtxt('MCDataFiles/MeanMagn15x15Nit225000.txt', delimiter=',')
mean2Magn5x5=np.loadtxt('MCDataFiles/Mean2Magn5x5Nit50000.txt', delimiter=',')
mean2Magn10x10=np.loadtxt('MCDataFiles/Mean2Magn10x10Nit200000.txt', delimiter=',')
mean2Magn15x15=np.loadtxt('MCDataFiles/Mean2Magn15x15Nit225000.txt', delimiter=',')
meanE5x5=np.loadtxt('MCDataFiles/MeanE5x5Nit50000.txt', delimiter=',')
meanE10x10=np.loadtxt('MCDataFiles/MeanE10x10Nit200000.txt', delimiter=',')
meanE15x15=np.loadtxt('MCDataFiles/MeanE15x15Nit225000.txt', delimiter=',')
meanE25x5=np.loadtxt('MCDataFiles/Mean2E5x5Nit50000.txt', delimiter=',')
meanE210x10=np.loadtxt('MCDataFiles/Mean2E10x10Nit200000.txt', delimiter=',')
meanE215x15=np.loadtxt('MCDataFiles/Mean2E15x15Nit225000.txt', delimiter=',')

#Create data arrays for the magnetic susceptibility:
Susc5x5 = HeatCapacity(meanMagn5x5, mean2Magn5x5, Tvec, 25)
Susc10x10 = HeatCapacity(meanMagn10x10, mean2Magn10x10, Tvec, 100)
Susc15x15 = HeatCapacity(meanMagn15x15, mean2Magn15x15, Tvec, 225)

#Create data arrays for the heat capacity:
Cv5x5 = MagneticSusceptibility(meanE5x5, meanE25x5, Tvec, 25)
Cv10x10 = MagneticSusceptibility(meanE10x10, meanE210x10, Tvec, 100)
Cv15x15 = MagneticSusceptibility(meanE15x15, meanE215x15, Tvec, 225)

#Example of how to make a plot of the magnetization:
Tvec2 = np.linspace(0.1, 5, num=1000000)
theoryMag = [(1-1/np.sinh(2/x)**4)**(1/8) for x in Tvec2]
fig, ax = plt.subplots()
ax.plot(Tvec, abs(meanMagn5x5), 'ro', label='5x5')
ax.plot(Tvec, abs(meanMagn10x10), 'bo', label='10x10')
ax.plot(Tvec, abs(meanMagn15x15), 'go', label='15x15')
ax.plot(Tvec2, theoryMag, '--g', label='Theoretical')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('$<m> / N$')
ax.set_title('Magnetization as a function of temperature for different system sizes')
ax.legend()
