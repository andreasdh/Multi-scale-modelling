import matplotlib.pyplot as plt
import numpy as np
import time
from lammps_logfile import get_color_value

def generate_velocities(N,T,kb=1.38064852E-23):

    """
     Generates random velocity from the Maxwell-boltzmann distribution.
    
    Parameters
    ----------
    N : int
        Number of particles
    T:  float
        Temperature in K
    kb: float
        Boltzmann constant in m^2kgs^-2K^s-1

    Returns
    -------
    v : float
        Velocity in m/s.
    """
    velocity_paired = []
    v = np.random.randn(N,3)*np.sqrt((kb*T)/m)
    v = np.append(v, v, axis = 0)
    for i in range(N):
        velocity_paired.append([v[i],v[i+N]])
    velocity_paired = np.array(velocity_paired)
    P = np.sum(m*v,axis=0)                    # Calculating the total momentum
    v = v - P/(m*N)                           # Subtracting the total momentum/m for all particles
    return velocity_paired

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
    r = np.random.uniform(-L/10,L/10, (N,3))
    r = np.append(r, r+R0, axis = 0)
    particle_pairs = []
    for i in range(N):
        particle_pairs.append([r[i],r[i+N]])
    return r, np.array(particle_pairs)

def forces_LJ(N,r,R0,epsilon):
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
    r_cutoff = (3*sigma)**2
    F = np.zeros((N,3))
    # Loop over particle pairs
    for i in range(N):
        for j in range(i+1,N): 
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

def velocity_verlet(N,m,n,r,v,dt):
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
    for i in range(n):
        # Velocity-Verlet algorithm
        F_inter = forces_LJ(N, r, R0, epsilon)
        print(F_inter)
        F_intra = 0
        a = np.array([(F_inter + F_intra)/m])
        v = v + a*dt
        r = r + v*dt # + 0.5*a*dt**2
        # Reflective boundary conditions (bounce off wall)
        for j in range(N):
            if abs(r[j,0]) > L/2:
                r[j,0] = L*r[j,0]/abs(r[j,0]) - r[j,0]
                v[j,0] = v[j,0]*(-1)
            if abs(r[j,1]) > L/2:
                r[j,1] = L*r[j,1]/abs(r[j,1]) - r[j,1]
                v[j,1] = v[j,1]*(-1)
            if abs(r[j,2]) > L/2:
                r[j,2] = L*r[j,2]/abs(r[j,2]) - r[j,2]
                v[j,2] = v[j,2]*(-1)
        position_evolution.append(r)
    return np.array(position_evolution),F_inter

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
    ax.scatter(r[:,0], r[:,1], r[:,2], color = get_color_value(np.arange(0,len(r)),0,len(r),cmap='jet'), label = 'Initial positions')
    v = v*t_end # Scale velocity so vectors are visible
    ax.quiver(r[:,0], r[:,1], r[:,2],v[:,0], v[:,1], v[:,2], color = 'black', label = 'Initial velocities')
    #ax.scatter(s[-1,:,0],s[-1,:,1],s[-1,:,2], color = get_color_value(np.arange(0,len(r)),0,len(r),cmap='jet'), label = 'Final positions')
    plt.legend()
    plt.tight_layout()
    plt.show()      
    
def write_to_file(t, s, filename):
    myfile = open(filename, 'w')
    myfile.write('Time (s)  r_x r_y r_z \n')
    for i in range(len(s)):
        for j in range(N):
            myfile.write(str(t[i]) + ', ' + str(s[i,j][0]) + ', ' + str(s[i,j][1]) + ', ' + str(s[i,j][2]) + '\n')
    myfile.close()


if __name__ == "__main__":
    # Constants and inital conditions
    mass_u = 15.999                           # Mass for oxygen atoms in u
    m = mass_u*1.6605402E-27                  # Mass in kg
    N_A = 6.022E23                            # Avogadro constant
    kb = 1.38064852E-23                       # Boltzmann constant (m^2kgs^-2K^s-1)
    T = 298                                   # Temperature in Kelvin
    k = 4410/(1000*N_A*(1E-10)**2)            # Intramolecular potential constant (J/m^2)
    epsilon = 0.71128/(1000*6.022E23)         # Energy minimum (kcal/mol)
    R0 = 1.21E-10                             # Equilibrium position between bonded oxygen atoms in Å
    sigma = 2.96E-10                          # How close non-bonding particles can get (vdW radius) in Å
    L = 100E-10                               # Box size in Å
    N = 2                                     # Number of particles 
    N_molecules = N*2                         # Number of molecules

    t0 = time.time()
    # Time intervals
    dt = 1E-15               # Time step
    t_end = 5E-13            # End time in seconds
    n = int(t_end/dt)        # Number of intervals
    
    # Running functions
    t0 = time.time()
    v = generate_velocities(N, T)
    r, particle_pairs = generate_positions_random_molecule(N, L)
    t = np.linspace(0,t_end,n+1)
    s, F_interF = velocity_verlet(N_molecules,m,n,particle_pairs,v,dt)
    time_end = time.time()
    print("Time elapsed:", time_end - t0)
    position_velocity_test(r,v,s)
    #write_to_file(t, s, 'oxygen.txt')