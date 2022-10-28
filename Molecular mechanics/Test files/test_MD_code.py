import numpy as np

pos = np.random.randn(natoms,3)

for i in range(3): #3D
    pos[i,:] = box[i][0] + (box[i][1] - box[i][0])*pos[:,i]
    
step = 0
output []

while step <= nsteps:
    
    step += 1
        