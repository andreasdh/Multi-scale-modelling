import matplotlib.pyplot as plt
import numpy as np
import time
from lammps_logfile import get_color_value
from playsound import playsound

def generate_velocities(N,T):

    """
     Generates random velocity from the Maxwell-boltzmann distribution.
    
    Parameters
    ----------
    N : int
        Number of particles
    T:  float
        Temperature in K
        
    Returns
    -------
    v : float
        Velocity in m/s.
    """
    v = np.random.randn(N,3)*np.sqrt((kb*T)/m)
    P = np.sum(m*v,axis=0)                    # Calculating the total momentum
    v = v - P/(m*N)                           # Subtracting the total momentum/m for all particles
    return v

def generate_positions_random_molecule(N,L):
    """
    Generates random positions.

    Parameters
    ----------
    N : int
        Number of particles.

    Returns
    -------
    r : array
        Nx3 position array.
    """
    r = np.random.uniform(-L/2,L/2, (N,3))
    r = np.append(r, r+1.1*R0, axis = 0)
    return r

def forces_LJ(N, N_molecules, r,R0,epsilon):
    """
    Uses the Lennard Jones potential in 3 dimensions to calculate
    the force on every particle.

    Parameters
    ----------
    N : int
        Number of particles
    r : array
        Nx3 position array.
    R0 : float
        Equilibrium distance between particles.
        
    Returns
    -------
    F : array
        Nx3 force array.
    """
    sigma = R0/(2**(1/6))    # How close non-bonding particles can get (vdW radius) in m
    r_cutoff = 3*sigma
    F = np.zeros((N,3))
    # Loop over particle pairs
    for i in range(N_molecules):
        for j in range(i+1,N_molecules): 
            dr = r[i,:]-r[j,:] # Distance vector
            rdist_sq = np.sum(dr*dr)
            if rdist_sq >= r_cutoff:
                Fij = 0
            else:
                rdist = np.sqrt(rdist_sq) # Length of distance vector
                Fij = 24*epsilon*(2*sigma**12/rdist**13 - sigma**6/rdist**7)*dr/rdist
            F[i,:] += Fij
            F[j,:] -= Fij
    return F

def forces_vibration(N, N_molecules, r, x0, k):
    F = np.zeros((N,3))
    for i in range(N_molecules):
        dr = r[i,:]-r[i+N_molecules,:]       # Distance vector
        rdist_sq = np.sum(dr*dr)             # Bond distance
        Fij = -k*(rdist_sq - x0)*dr/rdist_sq # Vibration potential
        F[i,:] += Fij
        F[i+N_molecules,:] -= Fij
    return F

def velocity_verlet(N,m,n,r,v,R0,N_molecules,dt):
    """
    Integration of the equations of motion by the Velocity-Verlet algorithm.
    
    Parameters
    ----------
    N : int
        Number of particles.
    n : int
        Number of iterations.
    r : array
        Nx3 position array.
    v : array
        Nx3 velocity array.

    Returns
    -------
    position : list
        List of all n values of positions for all N particles in 
        3 dimensions.
    """
    position_evolution = []
    position_evolution.append(r)
    
    F_inter = forces_LJ(N, N_molecules, r, R0, epsilon)
    F_intra = forces_vibration(N, N_molecules, r, R0, k)
    a_previous = (F_inter+F_intra)/m
    for i in range(n):
        # Velocity-Verlet algorithm
        F_inter = forces_LJ(N, N_molecules, r, R0, epsilon)
        F_intra = forces_vibration(N, N_molecules, r, R0, k)
        force = np.sqrt(F_intra**2)
        a = (F_inter+F_intra)/m
        v = v + (a_previous + a)*dt/2
        #v = thermostat_Berendsen(v, m, dt, tau=0.1)
        r = r + v*dt + 0.5*a*dt**2
        a_previous = a
        # Periodic boundary conditions
        for j in range(N):
            if abs(r[j,0]) >= L/2:
                r[j,0] = r[j,0]*-1
            if abs(r[j,1]) >= L/2:
                r[j,1] = r[j,1]*-1
            if abs(r[j,2]) >= L/2:
                r[j,2] = r[j,2]*-1

        position_evolution.append(r)
    return np.array(position_evolution), v

def position_velocity_test(r,v,s):
    """
    Plots the initial position and velocity of the particles along with the final positions.
    
    Parameters
    ----------
    r : array
        Nx3 initial position array.
    v : array
        Nx3 velocity array.
    s : array
        Nx3 calculated position array.

    Returns
    -------
    None.

    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot(r[:,0], r[:,1], r[:,2],color='black')
    ax.scatter(r[:,0], r[:,1], r[:,2], color = get_color_value(np.arange(0,len(r)),0,len(r),cmap='jet'), label = 'Initial positions')
    v = v*3E-13 # Scale velocity so vectors are visible
    ax.quiver(r[:,0], r[:,1], r[:,2],v[:,0], v[:,1], v[:,2], color = 'red', label = 'Initial velocities')
    #ax.plot(s[int(n/2),:,0],s[int(n/2),:,1],s[int(n/2),:,2],color='green')
    ax.scatter(s[int(n/2),:,0],s[int(n/2),:,1],s[int(n/2),:,2], color = get_color_value(np.arange(0,len(r)),0,len(r),cmap='jet'), label = 'positions half way')
    #ax.plot(s[-1,:,0],s[-1,:,1],s[-1,:,2],color='blue')
    ax.scatter(s[-1,:,0],s[-1,:,1],s[-1,:,2], color = get_color_value(np.arange(0,len(r)),0,len(r),cmap='jet'), label = 'Final positions')
    
    #plt.legend()
    scale = 1
    ax.set_xlim3d(-L/2*scale, L/2*scale) 
    ax.set_ylim3d(-L/2*scale, L/2*scale) 
    ax.set_zlim3d(-L/2*scale, L/2*scale) 
    plt.tight_layout()
    plt.show()      
    
def write_to_file(t, s, filename):
    myfile = open(filename, 'w')
    myfile.write('Time (s)  r_x r_y r_z \n')
    for i in range(len(s)):
        for j in range(N):
            myfile.write(str(t[i]) + ', ' + str(s[i,j][0]) + ', ' + str(s[i,j][1]) + ', ' + str(s[i,j][2]) + '\n')
    myfile.close()

def thermostat_Berendsen(v, m, dt, tau):
    E_k = 0.5*m*np.sum(v**2)**2
    T_t = (2/3)*E_k/kb
    print(T_t)
    l = 1 + dt/tau * (T-T_t) # lambda
    v = v*l
    return v
    
if __name__ == "__main__":
    # Constants and inital conditions
    mass_u = 15.999                           # Mass for oxygen atoms in u
    m = mass_u*1.6605402E-27                  # Mass in kg
    eV = 1.602176634E10-19                    # Electron volt conversion to joules
    N_A = 6.022E23                            # Avogadro constant
    kb = 1.38064852E-23                       # Boltzmann constant (J/K)
    T = 298                                   # Temperature in Kelvin
    k = 4410/(1000*N_A)                       # Intramolecular potential constant (J/m^2)
    epsilon = 0.71128/(1000*N_A)              # Energy minimum (J)
    R0 = 1.21E-10                             # Equilibrium position between bonded oxygen atoms in m
    sigma = 2.96E-10                          # How close non-bonding particles can get (vdW radius) in m
    L = 1E-9                                  # Box size in m
    N_molecules = 100                          # Number of molecules
    N = N_molecules*2                         # Number of particles 

    t0 = time.time()
    # Time intervals
    dt = 5E-15               # Time step
    t_end = 1E-10           # End time in seconds
    n = int(t_end/dt)        # Number of intervals
    
    # Running functions
    t0 = time.time()
    v = generate_velocities(N, T)
    T = m*np.sum(v**2)/(3*N*kb)

    r = generate_positions_random_molecule(N_molecules, L)
    t = np.linspace(0,t_end,n+1)
    s, v_end = velocity_verlet(N,m,n,r,v,R0,N_molecules,dt)
    time_end = time.time()
    print("Time elapsed:", time_end - t0)
    position_velocity_test(r,v,s)

    playsound('mario-win.mp3') # Play sound when finished with simulation
    #write_to_file(t, s, 'oxygen.txt')
    print(T)