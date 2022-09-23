# ==============================================================
# Data generation - verification exercise
# written by: Hyoung Suk Suh & Chulmin Kweon (Columbia Univ.)
# ==============================================================

# Import necessary packages and functions
import sys
import pandas as pd
import autograd.numpy as np

from autograd import elementwise_grad as egrad

sys.path.insert(0, '../offline_training')
from util.coordinate_transforms import *


# INPUT --------------------------------------------------------
# Material properties
E       = 200e3 # Young's modulus
sigma_y = 250.0 # Initial yield stress [MPa]
n_hard  = 0.2   # Hardening parameter

# Hosford parameter
N = 1.0 # Tresca (N = 1); Hosford (N = 1.5); vonMises (N = 2)

if N == 1.0:
  file_name = "Tresca"
elif N == 2.0:
  file_name = "vonMises"
else:
  file_name = "Hosford"

# Model function
def f(p, rho, theta, lamda):

  sigma1, sigma2, sigma3 = convert_prt_to_123(p, rho, theta)

  phi = (1/2)*(np.abs(sigma1-sigma2)**N + \
               np.abs(sigma2-sigma3)**N + \
               np.abs(sigma3-sigma1)**N)

  kappa = sigma_y*(1 + E*lamda/sigma_y)**n_hard

  return phi**(1/N) - kappa

# Data range
# p: (1/3)*tr(sigma) [MPa]
min_p = -2e3
max_p = 2e3
N_p   = 20

# theta: Lode's angle [rad]
min_theta = 0
max_theta = 2*np.pi
N_theta   = 50

# lamda: plastic multiplier
min_lamda = 0
max_lamda = 0.1
N_lamda   = 20
# --------------------------------------------------------------




# Generate equally-spaced stress points in cylindrical coordinates
p     = np.linspace(min_p, max_p, N_p)
theta = np.linspace(min_theta, max_theta, N_theta+1)[0:N_theta]
lamda = np.linspace(min_lamda, max_lamda, N_lamda)

p_nd, theta_nd, lamda_nd = np.meshgrid(p, theta, lamda) # n-dim. grid

p_nd     = np.reshape(p_nd,     (N_p*N_theta, N_lamda))
theta_nd = np.reshape(theta_nd, (N_p*N_theta, N_lamda))
lamda_nd = np.reshape(lamda_nd, (N_p*N_theta, N_lamda))

p_nd     = np.reshape(p_nd,     (N_p*N_theta*N_lamda,1))
theta_nd = np.reshape(theta_nd, (N_p*N_theta*N_lamda,1))
lamda_nd = np.reshape(lamda_nd, (N_p*N_theta*N_lamda,1))

p_nd     = np.reshape(p_nd, -1)
theta_nd = np.reshape(theta_nd, -1)
lamda_nd = np.reshape(lamda_nd, -1)



# Compute rho based on model function
rho_nd = np.zeros_like(p_nd)
get_dfdrho = egrad(f, 1)

maxiter = 200
tol = 1e-11

for i in range(np.shape(p_nd)[0]):

  x = sigma_y

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f(p_nd[i], x, theta_nd[i], lamda_nd[i])
    jac = get_dfdrho(p_nd[i], x, theta_nd[i], lamda_nd[i])

    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err)

    if err < tol:
      rho_nd[i] = x
      break




# Save output
p_nd     = p_nd.reshape((-1,1))
rho_nd   = rho_nd.reshape((-1,1))
theta_nd = theta_nd.reshape((-1,1))
lamda_nd = lamda_nd.reshape((-1,1))

data = np.hstack((p_nd, rho_nd, theta_nd, lamda_nd))

output = pd.DataFrame({'p':     data[:, 0], \
                       'rho':   data[:, 1], \
                       'theta': data[:, 2], \
                       'lamda': data[:, 3]})

output.to_csv("./raw_data_"+file_name+".csv", index = False)