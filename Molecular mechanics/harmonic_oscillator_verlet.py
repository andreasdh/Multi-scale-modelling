import numpy as np
import matplotlib.pyplot as plt

# Time intervals
dt = 1E-4      # Time step
time = 100     # End time in seconds
N = int(time/dt) # Number of intervals

# Inital values and constants
m = 1   # Mass in kg
k = 4 # Spring constant in N/m
x0 = 0.1  # Start position in m
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

# Calculation of additional start values by Euler method
a[0] = forces(x[0])
v[1] = v[0] + a[0]*dt
x[1] = x[0] + v[1]*dt
t[0] = t0

# Integration with Verlet algorithm
for i in range(1,N):
    a[i] = forces(x[i])/m
    x[i+1] = 2*x[i] - x[i-1] + a[i]*dt**2
    v[i+1] = (x[i+1] - x[i-1])/(2*dt)
    t[i+1] = t[i] + dt
    
def analytical_integrated_forces(x0,t):
    omega = np.sqrt(k/m)
    x = x0*np.cos(omega*t)
    return x

analytical = analytical_integrated_forces(x0, t)
plt.plot(t,x,color='limegreen',label='Numerical solution')
plt.plot(t,analytical,color='crimson',label='Analytical solution', linestyle='--')
plt.title('Integration of harmonic ocillator by Verlet algorithm')
plt.axhline(y=x0,color='green')
plt.axhline(y=-x0,color='red')
plt.legend()
plt.show()

