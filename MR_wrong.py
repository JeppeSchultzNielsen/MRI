import numpy as np
import scipy as sp

print('Hello')
print(2)
tester = np.zeros(shape = (2,3,5))
print(tester[0][0]) # Err
a,b,c = np.shape(tester)
print(np.shape(tester), a, b, c)

np.random.seed(2)
test_data = np.random.rand(480, 480, 496)

b = 10 # b-value
S_0 = 5
grads = np.random.rand(495, 3)

def signal_fit(Ds, grads):
    exp_sum = 0
    x_len, y_len, _ = np.shape(grads) # third input is N_slices
    for i in range(x_len):
        for j in range(y_len):
            if i!=j:
                continue
            b_ij = b*grads[i][j]
            exp_sum -= b_ij*Ds[i][j]
    return S_0*np.exp(exp_sum)

def trace_find(grads):
    Ds = sp.optimize.least_square(grads).x
    return np.trace(Ds)

def FA(lambdas, D_mean):
    return np.sqrt(1.5*((lambdas[0]-D_mean)**2 + (lambdas[1] - D_mean)**2 + (lambdas[2]-D_mean)**2)/sum(lambdas**2))
