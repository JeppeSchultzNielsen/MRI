import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

### Plot definitions
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', weight='normal')
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.figure(figsize = (5.8, 5.8))
#### Plot definitions end 


t0 = 0
dt = 1e-4 #ms
T1 = T2 = 1000 #ms

def bloch_solve_prime(M_init, T1, T2, omega1, d_omega, phi, T): # Solving in primed coordinate system (hopefully?)
    n = int((T-t0)/dt)
    M = np.zeros((n,3))
    M0 = np.linalg.norm(M_init) ## Maybe?
    M[0] = M_init
    B_mat = np.array([[-1/T2,                   d_omega,                -omega1*np.sin(phi)],
                      [-d_omega,               -1/T2,                    omega1*np.cos(phi)],
                      [ omega1*np.sin(phi),    -omega1*np.cos(phi),     -1/T1]])
    for i in range(1,n):
        M[i] = M[i-1] + dt * (np.matmul(B_mat,M[i-1]) + np.array([0, 0, M0/T1]))
    return M
def bloch_solve_prime_time(M_init, T1, T2, omega1, d_omega, phi, T): # Solving in primed coordinate system (hopefully?)
    #n = int((T-t0)/dt)
    n = len(d_omega)
    M = np.zeros((n,3))
    M0 = np.linalg.norm(M_init) ## Maybe?
    M[0] = M_init
    for i in range(1,n):
        B_mat = np.array([[-1/T2,                       d_omega[i-1],                -omega1[i-1]*np.sin(phi)],
                          [-d_omega[i-1],              -1/T2,                         omega1[i-1]*np.cos(phi)],
                          [ omega1[i-1]*np.sin(phi),   -omega1[i-1]*np.cos(phi),     -1/T1]])
        M[i] = M[i-1] + dt * (np.matmul(B_mat,M[i-1]) + np.array([0, 0, M0/T1]))
    return M

gamma = 2.675e8 #T^-1 s^-1

def gradient(t_w, delta_x=2): #delta_x in mm
    return 2*np.pi/(gamma*t_w*delta_x)
def delta_omega(x,t_w):
    return gamma * gradient(t_w) * x

M_in = np.array([0., 0., 1.])
M_sol = bloch_solve_prime(M_in, T1, T2, np.pi/2, delta_omega(0,1), 0, 1)

# Collecting data
data = M_sol[0::10].T #Only every 10 datapoints for faster animation
new_data = np.concatenate((np.zeros(np.shape(data)), data))

#///////////////////////// Plotting
def plot_A(): 
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    quiver = ax.quiver(*new_data[:,0])
    def update(id):
        nonlocal quiver
        quiver.remove()
        quiver = ax.quiver(*new_data[:,int(id)])
    # Setting the axes properties
    ax.set_xlim3d([-1.2, 1.2])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.2, 1.2])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.2, 1.2])
    ax.set_zlabel('Z')
    # SPHERE
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r", alpha=0.2)
    # END-SPHERE

    # Creating the Animation object
    datanums = np.linspace(0,np.shape(new_data)[1]-1,1000)
    vec_ani = animation.FuncAnimation(fig, update, frames=datanums, interval = 1)

    plt.show()
#/////////////////////////

#B
gamma = 2.675e8 #T^-1 s^-1

def gradient(t_w, delta_x=2): #delta_x in mm
    return 2*np.pi/(gamma*t_w*delta_x)
def delta_omega(x,t_w):
    return gamma * gradient(t_w) * x

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

ts = np.arange(-1, 10, dt)
#plt.plot(ts, RF_pulse(ts, 1, 1, 10))
#plt.show()

xs = np.arange(0,100,2)
def mag_prof(grad_str = 1):
    mags = []
    for x in xs:
        delta_om = grad_str*delta_omega(x,1)
        mx, my = bloch_solve_prime(M_in, T1, T2, np.pi/2, delta_om, 0, 1)[-1][0:2]
        mags.append(np.sqrt(mx**2+my**2))
    return mags
#plt.plot(xs, mag_prof())
#plt.show()
#plt.savefig('')
#plot_A()
print(RF_pulse(ts, 10e-6, 1, 1))
#M_sol = bloch_solve_prime_time(M_in, T1, T2, gamma*RF_pulse(ts, 10e-6, 1, 1), np.zeros(int((10+1)/dt)), 0, 1)
# Collecting data
#data = M_sol[0::10].T #Only every 10 datapoints for faster animation
#new_data = np.concatenate((np.zeros(np.shape(data)), data))
plot_A()