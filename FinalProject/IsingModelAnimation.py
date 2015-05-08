# LatticeBoltzmannDemo.py:  a two-dimensional lattice-Boltzmann "wind tunnel" simulation
# Uses numpy to speed up all array handling.
# Uses matplotlib to plot and animate the curl of the macroscopic velocity field.

# Copyright 2013, Daniel V. Schroeder (Weber State University) 2013

# Permission is hereby granted, free of charge, to any person obtaining a copy of 
# this software and associated data and documentation (the "Software"), to deal in 
# the Software without restriction, including without limitation the rights to 
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to do 
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

# Except as contained in this notice, the name of the author shall not be used in 
# advertising or otherwise to promote the sale, use or other dealings in this 
# Software without prior written authorization.

# Credits:
# The "wind tunnel" entry/exit conditions are inspired by Graham Pullan's code
# (http://www.many-core.group.cam.ac.uk/projects/LBdemo.shtml).  Additional inspiration from 
# Thomas Pohl's applet (http://thomas-pohl.info/work/lba.html).  Other portions of code are based 
# on Wagner (http://www.ndsu.edu/physics/people/faculty/wagner/lattice_boltzmann_codes/) and
# Gonsalves (http://www.physics.buffalo.edu/phy411-506-2004/index.html; code adapted from Succi,
# http://global.oup.com/academic/product/the-lattice-boltzmann-equation-9780199679249).

# For related materials see http://physics.weber.edu/schroeder/fluids

import numpy as np
import time, matplotlib.pyplot, matplotlib.animation

def initialize_random_lattice(Nx, Ny):
    lattice = (2.0*np.random.randint(2, size=Nx*Ny)-1.0).reshape(Ny,Nx)
    return lattice

# Define constants:
Nx = 50						# lattice dimensions
Ny = 50
N_equilibrium=2000
N_sampling=2000
T=0.01
Jx=1
Jy=1
h=0

lattice=initialize_random_lattice(Nx,Ny)

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

# Here comes the graphics and animation...
theFig = matplotlib.pyplot.figure(figsize=(16,8))
#fig1=pyplot.subplot
fluidImage = matplotlib.pyplot.imshow(MonteCarlo_Sweep(lattice, Jx,Jy,h,T), origin='lower', norm=matplotlib.pyplot.Normalize(-.1,.1), 
									cmap=matplotlib.pyplot.get_cmap('Greys'), interpolation='none')
		# See http://www.loria.fr/~rougier/teaching/matplotlib/#colormaps for other cmap options
#bImageArray = np.zeros((Nx, Ny, 4), np.uint8)	# an RGBA image							# set alpha=255 only at barrier sites
#barrierImage = matplotlib.pyplot.imshow(bImageArray, origin='lower', interpolation='none')

# Function called for each successive animation frame:
startTime = time.clock()
#frameList = open('frameList.txt','w')		# file containing list of images (to make movie)
def nextFrame(arg):							# (arg is the frame number, which we don't need)
	global startTime
	fluidImage.set_array(MonteCarlo_Sweep(lattice, Jx,Jy,h,T))
	return (fluidImage)		# return the figure elements to redraw

animate = matplotlib.animation.FuncAnimation(theFig, nextFrame, interval=1, blit=False)
matplotlib.pyplot.show()
