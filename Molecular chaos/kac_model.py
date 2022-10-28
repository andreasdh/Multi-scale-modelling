import numpy as np

# Reversibility with functions

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
    N_wb = np.zeros(N)
    for j in range(R):
        for i in range(N):
            if markers[i-1] == '*':
                N_wb[i] = N_wb0[i-1]*(-1)
            else:
                N_wb[i] = N_wb0[i-1]
    return N_wb

def counter_clockwise_rotation(N, R, N_wb0, markers):
    """
    Rotates elements counter clockwise on a Kac-ring.
    
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
    N_back = np.zeros(N)
    for j in range(R):
        for i in range(N-2,-2,-1):
            if markers[i] == '*':
                N_back[i] = N_wb[i+1]*(-1)
            else:
                N_back[i] = N_wb[i+1]
    return N_back


N = 5 # Number of spheres
R = 1 # Number of rounds
N_wb0, markers = generate_dots(N)
N_wb = clockwise_rotation(N,R, N_wb0, markers)
N_back = counter_clockwise_rotation(N,R,N_wb0,markers)
print("Is the model time reversible?", list(N_back)==list(N_wb0))
