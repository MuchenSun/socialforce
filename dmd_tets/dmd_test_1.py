import matplotlib.pyplot as plt
import numpy as np
from pydmd import DMD
from past.utils import old_div
import time


def f1(x,t):
    return 1./np.cosh(x+3)*np.exp(2.3j*t)

def f2(x,t):
    return 2./np.cosh(x)*np.tanh(x)*np.exp(2.8j*t)

x = np.linspace(-5, 5, 128)
t = np.linspace(0, 4*np.pi, 256)

xgrid, tgrid = np.meshgrid(x, t)

X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

start = time.time()
dmd = DMD(svd_rank=2)
dmd.fit(X.T)
print("fit time: ", time.time()-start)

print((X.T)[0,0:10])

diff = (X - dmd.reconstructed_data.T).real
print(diff.shape, np.sum(diff))


omega = old_div(np.log(dmd.eigs), dmd.original_time['dt'])
vander = np.exp(np.outer( omega, dmd.dmd_timesteps - dmd.original_time['t0'] ))
dynamics = vander * dmd._b[:, None]
another_reconstructed_data = np.dot(dmd.modes, dynamics)
another_diff = (X - another_reconstructed_data.T).real
print(another_diff.shape, np.sum(another_diff))
