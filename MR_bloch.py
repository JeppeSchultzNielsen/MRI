import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

from phantominator import shepp_logan

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
#### Plot definitions END 

gamma = 2.675e5 #T^-1 ms^-1
t0 = -1
T = 1
dt = 1e-4 #ms
T1 = T2 = 1000 #ms
T_E = 16 #ms
T_R = 35 #ms
N = 128
L = 100 #mm
M_in = np.array([0., 0., 1.])
ts = np.arange(-t0, T, dt)
xs = np.arange(-L,L,2)

def gradient(t_w, delta_x=2): #delta_x in mm
    return 2*np.pi/(gamma*t_w*delta_x)
def delta_omega(x,t_w, delta_x = 2):
    return 2*np.pi*x/(t_w*delta_x)
    #return gamma * gradient(t_w) * x

def bloch_solve_prime(M_init, T1, T2, omega1, d_omega, phi): # Solving in primed coordinate system (hopefully?)
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

def bloch_solve_prime_time(M_init, T1, T2, omega1, d_omega, phi): # Solving in primed coordinate system (hopefully?)
    #n = int((T-t0)/dt)
    n = len(omega1)
    M = np.zeros((n,3))
    M0 = np.linalg.norm(M_init) ## Maybe?
    M[0] = M_init
    for i in range(1,n):
        B_mat = np.array([[-1/T2,                       d_omega,                     -omega1[i-1]*np.sin(phi)],
                          [-d_omega,                   -1/T2,                         omega1[i-1]*np.cos(phi)],
                          [ omega1[i-1]*np.sin(phi),   -omega1[i-1]*np.cos(phi),     -1/T1]])
        M[i] = M[i-1] + dt * (np.matmul(B_mat,M[i-1]) + np.array([0, 0, M0/T1]))
    return M

def bloch_solve_prime_pysolve(M_init, T1, T2, b1, d_omega, phi, t_w=1, n_z=1): # Solving in primed coordinate system (hopefully?)
    M0 = np.linalg.norm(M_init) ## Maybe?
    def omega1(t): 
        return gamma*RF_pulse_prime(t, b1, t_w, n_z)
    def fun(t, y):
        return np.matmul(np.array([[-1/T2,                        d_omega,                   -omega1(t)*np.sin(phi)],
                                   [-d_omega,                    -1/T2,                       omega1(t)*np.cos(phi)],
                                   [ omega1(t)*np.sin(phi),      -omega1(t)*np.cos(phi),     -1/T1]])
                        ,y) + np.array([0, 0, M0/T1])
    solve_ts = np.arange(t0, T, dt)
    sol = solve_ivp(fun, [t0,T], M_init, t_eval = solve_ts)
    return np.array(sol.t), np.array(sol.y)

def RF_pulse_prime(t, b1, t_w, n_z):
    t_rf = n_z*t_w
    if -0.5*t_rf <= t and t <= 0.5*t_rf:
        B1 = b1*np.sinc(2*t/t_w)
    else:
        B1 = 0
    return B1

#///////////////////////// Plotting
def plot_A(time, data): 
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    quiver = ax.quiver(*data[:,0])
    text = ax.text(0,0,0,f"{round(time[0],5)}")
    def update(id):
        nonlocal quiver
        nonlocal text
        quiver.remove()
        text.remove()

        text = ax.text(0,0,0,f"{round(time[int(id)],5)}")
        quiver = ax.quiver(*data[:,int(id)])
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
    datanums = np.linspace(0,np.shape(data)[1]-1,np.shape(data)[1])
    vec_ani = animation.FuncAnimation(fig, update, frames=datanums, interval = 1)

    plt.show()
#/////////////////////////

#B
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

def mag_prof(grad_str = 1):
    mags = []
    M_comps = np.zeros([len(xs),2])
    for i in range(len(xs)):
        x = xs[i]
        delta_om = grad_str*delta_omega(x,1)
        mx, my = bloch_solve_prime(M_in, T1, T2, np.pi/2, delta_om, 0)[-1][0:2]
        mags.append(np.sqrt(mx**2+my**2))
        M_comps[i,0] = mx
        M_comps[i,1] = my
    return mags, M_comps
def mag_prof_time(grad_str = 1, t_w = 1, n_z = 1): # Magnitude profile
    mags = []
    ts = np.arange(-0.5,0.5,dt)
    for x in xs:
        delta_om = grad_str*delta_omega(x,t_w)
        mx, my = bloch_solve_prime_pysolve(M_in, T1, T2, 10e-6, delta_om, 0, t_w, n_z)[1].T[-1][0:2]
        #mx, my = bloch_solve_prime_time(M_in, T1, T2, gamma*RF_pulse(ts,10e-6,1,1), delta_om, 0)[-1][0:2]
        mags.append(np.sqrt(mx**2+my**2))
    return mags

### NOTE PLOTTING PROBLEM 1 ################
# Collecting data
test_time, test_dat = bloch_solve_prime_pysolve(M_in, T1, T2, 10e-6, delta_omega(200,1), 0)
data = test_dat.T[0::10].T #Only every 10 datapoints for faster animation
new_data = np.concatenate((np.zeros(np.shape(data)), data)) # Adding points at 0 for plotting arrow
test_time = test_time[0::10]

### NOTE Getting data without carying about time-dependency of B-field.
#M_sol = bloch_solve_prime(M_in, T1, T2, np.pi/2, delta_omega(0,1), 0)

### NOTE Geting data for ex 1 with time-dependent B-field. Magnitude of 10µT.
#M_sol = bloch_solve_prime_time(M_in, T1, T2, gamma*RF_pulse(ts, 10e-6, 1, 1), delta_omega(100,1), 0)
#test_time = ts
#data = M_sol[0::10].T #Only every 10 datapoints for faster animation

#test_time = test_time[0::10]
#new_data = np.concatenate((np.zeros(np.shape(data)), data))

#plot_A(test_time, new_data) ## NOTE COMMENT THIS OUT TO PLOT
############################################# END PLOTTING OF PROBLEM 1

### PLOTTING FIGURE 1 FROM EX.
#plt.plot(ts, RF_pulse(ts, 1, 1, 10))
#plt.show()

### NOTE PLOTTING PROBLEM 2.2 AND 2.3 (ADAPTIVE)
#plt.plot(xs, mag_prof_time(1, 1, 1))
#plt.show()


### NOTE EXERCISE 3
M0 = np.abs(shepp_logan(N)[64])
### NOTE Gives figure 3.
#plt.plot(np.linspace(0,40,len(M0)), M0)
#plt.show()

### NOTE Plot slice image.
import seaborn as sb
#sb.heatmap(shepp_logan(N))
#plt.show()

### NOTE Comparison - doesn't work yet.
xs = np.arange(-0,100,2)
comps = mag_prof()[1] # Using simple Euler integration, because out solver using solve_ivp does not allow instantaneous pi/2-pulse.
# Maybe necessary to update solving functions to allow better t=0 pi/2 pulse? Should gradient be changed?
Mp_x0 = comps[:,0] + 1j*(comps[:,1]) # Making M_+(x,t=0)
print(Mp_x0)
plt.plot(xs, np.fft.ifft(Mp_x0*np.exp(-T_E/T2)))
plt.show()

def T2(x, a= 0.5): # For final problem
    return 15 + a*x