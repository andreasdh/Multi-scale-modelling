import numpy as np
import matplotlib.pyplot as plt

# Time intervals
dt = 1E-3      # Time step
time = 50      # End time in seconds
N = int(time/dt) # Number of intervals

# Inital values and constants
m = 1   # Mass in kg
k = 4 # Spring constant in N/m
x0 = 1  # Start position in m
v0 = 0  # Start velocity in m/s
t0 = 0  # Start time in s

# Equations of Motion (EoM)
def forces(x, x_eq = 0):
    # x_eq = equilibrium position
    return -k*(x - x_eq)

# Initialization
a = np.zeros(N+1) # Acceleration in m/s^2
v = np.zeros(N+1) # Velocity in m/s
x = np.zeros(N+1) # Position in m
t = np.zeros(N+1) # Time in s
v[0] = v0         
x[0] = x0
a[0] = forces(x[0])
t[0] = t0

# Integration with velocity-Verlet algorithm
for i in range(N):
    x[i+1] = x[i] + v[i]*dt + 0.5*a[i]*dt**2
    a[i+1] = forces(x[i+1])/m
    v[i+1] = v[i] + (a[i] + a[i+1])*dt/2
    t[i+1] = t[i] + dt

def analytical_integrated_forces(x0,t):
    omega = np.sqrt(k/m)
    x = x0*np.cos(omega*t)
    return x

analytical = analytical_integrated_forces(x0, t)
plt.plot(t,x,color='limegreen',label='Numerical solution')
plt.plot(t,analytical,color='crimson',label='Analytical solution', linestyle='--')
plt.title('Integration of harmonic ocillator by Velocity-Verlet')
plt.axhline(y=x0,color='green')
plt.axhline(y=-x0,color='red')
plt.legend(loc=0)
plt.show()

