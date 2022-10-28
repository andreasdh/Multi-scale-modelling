import numpy as np
import matplotlib.pyplot as plt

# Plotting to demonstrate the stability of the Kac-model.

def generate_dots(N):
    """
    Generate random Kac ring.

    Parameters
    ----------
    N : int
        Total number of spheres.

    Returns
    -------
    N_wb0 : array
        Initial configuration of spheres of type 1 (white) or -1 (black)
    markers : array
        Elements of type '*' or 'o'.
    """
    N_wb0 = np.random.choice((-1,1),N)   # Start distribution for rings
    markers = np.random.choice(('*','o'),N) # N markers
    return N_wb0, markers


def clockwise_rotation(N, R, N_wb0, markers):
    """
    Rotates elements clockwise on a Kac-ring.
    
    Parameters
    ----------
    N : int
        Number of spheres.
    R : float
        Number of full rotations
    N_wb0 : array
        Initial configuration of spheres of type 1 (white) or -1 (black)
    markers : array
        Elements of type '*' or 'o'.

    Returns
    -------
    N_wb : array
        End configuration of spheres of type 1 (white) or -1 (black)
    """
    mean_of_circles = []
    N_wb = np.copy(N_wb0)        
    for j in range(R):
        N0 = N_wb[-1]
        for i in range(N-1,0,-1):
            if markers[i-1] == '*':
                N_wb[i] = N_wb[i-1]*(-1)
            else:
                N_wb[i] = N_wb[i-1]
        if markers[-1] == '*':
            N_wb[0] = N0*-1
        else:
            N_wb[0] = N0
        mean_of_circles.append(np.mean(N_wb))
    return N_wb, mean_of_circles

#N = 10000 # Number of spheres
#R = 100  # Number of rotations

spheres_nr = [100, 1000, 10000]
rotation_nr  = [10, 50, 100]

for R in rotation_nr: 
    plt.figure(figsize=(10,10))
    for N in spheres_nr:
        N_wb0, markers = generate_dots(N)
        N_wb, mean_of_circles = clockwise_rotation(N,R, N_wb0, markers)
        rotations = np.linspace(1,R,R) # Number of rotations for plotting on x-axis
        plt.title(f'Mean of circles for {R} rotations')
        plt.plot(rotations,mean_of_circles,label=f'{N} spheres')
        plt.axhline(y=0,color='red')
    plt.legend()

plt.tight_layout()
plt.show()