import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

t0 = 0
dt = 0.0001 #ms
T1 = T2 = 1000 #ms
def bloch_solve(M_init, T1, T2, omega1, d_omega, phi, T):
    n = int((T-t0)/dt)
    M = np.zeros((n, 3))
    M0 = np.linalg.norm(M_init) ### Is this right?
    B_mat = np.array([[-1/T2,                omega1              -omega1*np.sin(phi)],
                      [-omega1,             -1/T2,                omega1*np.cos(phi)],
                      [ omega1*np.sin(phi), -omega1*np.cos(phi), -1/T1]])
    for i in range(1, n):
        M[i] = M[i-1] + dt*(B_mat * M[i-1]*dt + np.array([0, 0, M0/T1]))
    return M

def bloch_solve_prime(M_init, T1, T2, omega1, d_omega, phi, T): # Solving in primed coordinate system (hopefully?)
    n = int((T-t0)/dt)
    M = np.zeros((n,3))
    M0 = np.linalg.norm(M_init) ## Maybe?
    M[0] = M_init
    R1 = 1/T1
    R2 = 1/T2
    #B_mat = np.array([[-1/T2,               -d_omega              -omega1*np.sin(phi)],
    #                  [-omega1,             -1/T2,                omega1*np.cos(phi)],
    #                  [ omega1*np.sin(phi), -omega1*np.cos(phi), -1/T1]])
    # https://www.sciencedirect.com/topics/medicine-and-dentistry/bloch-equation
    #B_mat = np.array([[-1/T2,         -d_omega,             0],
    #                  [-d_omega,      -1/T2,                2*np.pi*omega1*np.cos(phi)],
    #                  [ 0,            -2*np.pi*omega1*np.sin(phi),     -1/T1]])
    # ?? https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Spectroscopy/Magnetic_Resonance_Spectroscopies/Nuclear_Magnetic_Resonance/NMR_-_Theory/Bloch_Equations
    B_mat = np.array([[-1/T2,                   d_omega,                -omega1*np.sin(phi)],
                      [-d_omega,               -1/T2,                    omega1*np.cos(phi)],
                      [ omega1*np.cos(phi),    -omega1*np.sin(phi),     -1/T1]])
    for i in range(1,n):
        M[i] = M[i-1] + dt * (np.matmul(B_mat,M[i-1]) + np.array([0, 0, M0/T1]))
    return M

M_in = np.array([0., 0., 1.])
M_sol = bloch_solve_prime(M_in, T1, T2, np.pi/2, 0, 0, 1)

fig = plt.figure()
#ax = fig.add_subplot(projection = "3d")
#ax.scatter(*M_sol.T)
#plt.show()

#///////////////////////// Plotting
import mpl_toolkits.mplot3d.axes3d as p3
def update_line(num, data, line):
    # NOTE: there is no .set_data() for 3 dim data...
    line.set_data(data[0:2, :num])
    line.set_3d_properties(data[2, :num])
    return line

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
data = M_sol[0::3].T #Only every 3 datapoints for faster animation

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
line, = ax.plot(data[0],data[1],data[2])

# Setting the axes properties
ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_line, frames = np.shape(data)[1], fargs=(data, line),
                                   interval=1, blit=False)

plt.show()
#/////////////////////////

def RF_pulse(t, b1, t_w, n_z):
    t_rf = n_z*t_w
    n = len(t)
    B1 = np.zeros(n)
    for i in range(n):
        if -0.5*t_rf <= t[i] and t[i] <= 0.5*t_rf:
            B1[i] = b1*np.sin(2*np.pi*t[i]/t_w)/(2*np.pi*t[i]/t_w)
        else:
            B1[i] = 0
    return B1

def M_plus(M_p0, t, omega0, T2):
    return M_p0 * np.exp(-t*T2) * np.exp(-np.imag*omega0*t)

ts = np.linspace(-10,10, 1000)
#plt.plot(ts, RF_pulse(ts, 1, 1, 10))
#plt.show()