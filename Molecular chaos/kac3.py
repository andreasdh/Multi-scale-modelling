import numpy as np
import matplotlib.pyplot as plt
"""
Test program to verify reversibility of Kac-model.
"""
R = 7
N = 500 # Number of spheres
N_wb0 = np.random.choice((-1,1),N)   # Start distribution fo rings
markers = np.random.choice(('*','o'),N) # N markers

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

N_back = np.copy(N_wb)
for j in range(R):
    N0 = N_back[-1]
    for i in range(N-1):
        if markers[i-1] == '*':
            N_back[i-1] = N_back[i]*(-1)
        else:
            N_back[i-1] = N_back[i]
    if markers[-2] == '*':
        N_back[-2] = N0*-1
    else:
        N_back[-2] = N0
    
print(list(N_back)==list(N_wb0))