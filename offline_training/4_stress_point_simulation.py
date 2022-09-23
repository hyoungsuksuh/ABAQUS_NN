# ==============================================================
# Step 4: run stress point simulation
# written by: Hyoung Suk Suh & Chulmin Kweon (Columbia Univ.)
# ==============================================================

# Import necessary packages and functions
import os
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import autograd.numpy as np

from sklearn import preprocessing
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from autograd import elementwise_grad as egrad

from util.tensor_operations import *
from util.coordinate_transforms import *

from NN_separate import Multiply, NeuralNetwork_f, NeuralNetwork_dfdsig

# INPUT --------------------------------------------------------
# Material properties
# >> elastic properties (for both NN and benchmark)
E  = 200.0e3     # Young's modulus [MPa]
nu = 0.3         # Poisson's ratio

# Loading steps & increment
Nstep   = 720    # loading steps
eps_inc = 1.0e-4 # strain increment

# Newton-Raphson parameters
tol     = 1e-9
maxiter = 10

# Define loading direction (strain controlled)
stress_increment = np.array([[eps_inc,  0.0,          0.0],
                             [0.0,     -0.5*eps_inc,  0.0],
                             [0.0,      0.0,         -0.5*eps_inc]])

# File name
file_name = "Tresca"
# --------------------------------------------------------------




# Initialize variables
# >> NN-prediction
sigma = np.zeros((3,3)) # stress

eps_e = np.zeros((3,3)) # elastic strain 
eps_p = np.zeros((3,3)) # plastic strain
eps   = eps_e + eps_p   # strain

lamda = 0 # plastic multiplier




# Define identity tensors
I   = np.eye(3)
II  = identity_4(3)
IxI = tensor_oMult(I,I)




# Define elasticity model
K  = E / (3*(1-2*nu))
mu = E / (2*(1+nu))

a = K + (4/3)*mu
b = K - (2/3)*mu

Ce_principal = np.array([[a, b, b],
                         [b, a, b],
                         [b, b, a]])

dsig1depse1 = a
dsig1depse2 = b
dsig1depse3 = b
dsig2depse1 = b
dsig2depse2 = a
dsig2depse3 = b
dsig3depse1 = b
dsig3depse2 = b
dsig3depse3 = a




# Define yield function & plastic flow direction
# >> yield function
f_INPUT_scaler  = joblib.load("./"+file_name+"/f_INPUT_scaler.pkl")
f_OUTPUT_scaler = joblib.load("./"+file_name+"/f_OUTPUT_scaler.pkl")

model_f = torch.load("./"+file_name+"/f_NN.pth")

def f_NN(sigma1, sigma2, sigma3, lamda):

  p, rho, theta = convert_123_to_prt(sigma1, sigma2, sigma3)

  RT = np.array([p, rho, theta, lamda]).reshape(1,4)
  RT = f_INPUT_scaler.transform(RT)
  RT = torch.tensor(RT, dtype=torch.float)
  
  f = model_f(RT)
  f = f_OUTPUT_scaler.inverse_transform(f.detach().numpy())
  f = f[0]

  return f

# >> plastic flow direction (stress gradient)
dfdsig_INPUT_scaler  = joblib.load("./"+file_name+"/dfdsig_INPUT_scaler.pkl")
dfdsig_OUTPUT_scaler = joblib.load("./"+file_name+"/dfdsig_OUTPUT_scaler.pkl")

model_df = torch.load("./"+file_name+"/dfdsig_NN.pth")

def df_NN(sigma1, sigma2, sigma3, lamda):

  p, rho, theta = convert_123_to_prt(sigma1, sigma2, sigma3)

  RT = np.array([p, rho, theta, lamda]).reshape(1,4)
  RT = dfdsig_INPUT_scaler.transform(RT)
  RT = torch.tensor(RT, dtype=torch.float)

  dfdsig = model_df(RT)
  dfdsig = dfdsig_OUTPUT_scaler.inverse_transform(dfdsig.detach().numpy())

  dfdsig1 = dfdsig[:,0]
  dfdsig2 = dfdsig[:,1]
  dfdsig3 = dfdsig[:,2]
  
  return dfdsig1, dfdsig2, dfdsig3

# >> gradient of plastic flow (jacobian)
def df2_NN(sigma1, sigma2, sigma3, lamda):

  p, rho, theta = convert_123_to_prt(sigma1, sigma2, sigma3)

  A_factor = dfdsig_INPUT_scaler.scale_
  B_factor = 1. / dfdsig_OUTPUT_scaler.scale_

  A = convert_dfdprt_to_dfd123(p, rho, theta)
  RT = np.array([p, rho, theta, lamda]).reshape(1,4)
  RT = dfdsig_INPUT_scaler.transform(RT)
  RT = torch.tensor(RT, dtype=torch.float)
  RT.requires_grad = True

  dfdsig = model_df(RT)

  sD2F_train_predicted1 = torch.autograd.grad(dfdsig[0][0], RT, create_graph=True)[0]
  sD2F_train_predicted2 = torch.autograd.grad(dfdsig[0][1], RT, create_graph=True)[0]
  sD2F_train_predicted3 = torch.autograd.grad(dfdsig[0][2], RT)[0]

  gg_d2f0 = B_factor*(sD2F_train_predicted1[0][0:3].detach().numpy())*A_factor[0:3] @ A
  gg_d2f1 = B_factor*(sD2F_train_predicted2[0][0:3].detach().numpy())*A_factor[0:3] @ A
  gg_d2f2 = B_factor*(sD2F_train_predicted3[0][0:3].detach().numpy())*A_factor[0:3] @ A

  d2fdsig1dsig1  = gg_d2f0[0]
  d2fdsig2dsig2  = gg_d2f1[1]
  d2fdsig3dsig3  = gg_d2f2[2]
  d2fdsig1dsig2  = 0.5*(gg_d2f0[1] + gg_d2f1[0])
  d2fdsig2dsig3  = 0.5*(gg_d2f1[2] + gg_d2f2[1])
  d2fdsig3dsig1  = 0.5*(gg_d2f2[0] + gg_d2f0[2])

  return d2fdsig1dsig1, d2fdsig2dsig2, d2fdsig3dsig3, d2fdsig1dsig2, d2fdsig2dsig3, d2fdsig3dsig1




# Perform material point simulation 
sigma11 = np.zeros(Nstep+1)
eps11   = np.zeros(Nstep+1)

deps = stress_increment

for i in range(Nstep):

  print("Loading step [",i+1,"] ---------------------------------------")

  if i == 50:
    deps = -deps
  elif i == 150:
    deps = -deps
  elif i == 300:
    deps = -deps
  elif i == 500:
    deps = -deps

  # [1] Compute trial strain
  eps_e_tr = eps_e + deps

  eps_e_tr_principal_mag, eps_e_tr_principal_vec = np.linalg.eig(eps_e_tr)

  eps_e_tr1 = eps_e_tr_principal_mag[0]
  eps_e_tr2 = eps_e_tr_principal_mag[1]
  eps_e_tr3 = eps_e_tr_principal_mag[2]

  n1 = eps_e_tr_principal_vec[:,0]
  n2 = eps_e_tr_principal_vec[:,1]
  n3 = eps_e_tr_principal_vec[:,2]

  # [2] Compute trial stress
  sigma_tr_principal_mag = np.inner(Ce_principal, eps_e_tr_principal_mag)

  sigma_tr1 = sigma_tr_principal_mag[0]
  sigma_tr2 = sigma_tr_principal_mag[1]
  sigma_tr3 = sigma_tr_principal_mag[2]

  sigma_tr = sigma_tr1*np.tensordot(n1,n1,axes=0) + sigma_tr2*np.tensordot(n2,n2,axes=0) + sigma_tr3*np.tensordot(n3,n3,axes=0)

  # [3] Check yielding
  f = f_NN(sigma_tr1, sigma_tr2, sigma_tr3, lamda)

  # [3.1] If f <= 0, elastic.
  if f <= 0:
    print(">> Elastic!")

    # Update stress & strain
    sigma = sigma_tr

    eps_e = eps_e_tr
    eps   = eps_e + eps_p

  # [3.2] If f > 0, plastic.
  else:
    print(">> Plastic!")

    # Initialize variables
    eps_e_principal_mag, eps_e_principal_vec = np.linalg.eig(eps_e)

    eps_e1 = eps_e_principal_mag[0]
    eps_e2 = eps_e_principal_mag[1]
    eps_e3 = eps_e_principal_mag[2]
    dlamda  = 0

    x = np.zeros(4) # target variables
    x[0] = eps_e1
    x[1] = eps_e2
    x[2] = eps_e3
    x[3] = dlamda

    # Newton-Raphson iteration (return mapping)
    for ii in range(maxiter):

      # Initialize residual and jacobian
      res = np.zeros(4)
      jac = np.zeros((4,4))

      # Current strain
      eps_e1_current = x[0]
      eps_e2_current = x[1]
      eps_e3_current = x[2]

      # Current stress
      sigma1_current = a*eps_e1_current + b*eps_e2_current + b*eps_e3_current
      sigma2_current = b*eps_e1_current + a*eps_e2_current + b*eps_e3_current
      sigma3_current = b*eps_e1_current + b*eps_e2_current + a*eps_e3_current

      sigma1_current = sigma1_current 
      sigma2_current = sigma2_current 
      sigma3_current = sigma3_current 

      # Current lamda
      lamda_current = lamda + x[3]

      # Update derivatives
      # >> First order derivatives
      dfdsig1, dfdsig2, dfdsig3 = df_NN(sigma1_current, sigma2_current, sigma3_current, lamda_current)

      # >> Second order derivatives
      d2fdsig1dsig1, d2fdsig2dsig2, d2fdsig3dsig3, d2fdsig1dsig2, d2fdsig2dsig3, d2fdsig3dsig1 \
        = df2_NN(sigma1_current, sigma2_current, sigma3_current, lamda_current)

      # Update residual
      res[0] = x[0] - eps_e_tr1 + x[3]*dfdsig1
      res[1] = x[1] - eps_e_tr2 + x[3]*dfdsig2
      res[2] = x[2] - eps_e_tr3 + x[3]*dfdsig3
      res[3] = f_NN(sigma1_current, sigma2_current, sigma3_current, lamda_current)

      # Update Jacobian ***
      jac[0,0] = 1 + x[3]*(d2fdsig1dsig1*dsig1depse1 + d2fdsig1dsig2*dsig2depse1 + d2fdsig3dsig1*dsig3depse1)
      jac[0,1] =     x[3]*(d2fdsig1dsig1*dsig1depse2 + d2fdsig1dsig2*dsig2depse2 + d2fdsig3dsig1*dsig3depse2)
      jac[0,2] =     x[3]*(d2fdsig1dsig1*dsig1depse3 + d2fdsig1dsig2*dsig2depse3 + d2fdsig3dsig1*dsig3depse3)
      jac[0,3] = dfdsig1

      jac[1,0] =     x[3]*(d2fdsig1dsig2*dsig1depse1 + d2fdsig2dsig2*dsig2depse1 + d2fdsig2dsig3*dsig3depse1)
      jac[1,1] = 1 + x[3]*(d2fdsig1dsig2*dsig1depse2 + d2fdsig2dsig2*dsig2depse2 + d2fdsig2dsig3*dsig3depse2)
      jac[1,2] =     x[3]*(d2fdsig1dsig2*dsig1depse3 + d2fdsig2dsig2*dsig2depse3 + d2fdsig2dsig3*dsig3depse3)
      jac[1,3] = dfdsig2

      jac[2,0] =     x[3]*(d2fdsig3dsig1*dsig1depse1 + d2fdsig2dsig3*dsig2depse1 + d2fdsig3dsig3*dsig3depse1)
      jac[2,1] =     x[3]*(d2fdsig3dsig1*dsig1depse2 + d2fdsig2dsig3*dsig2depse2 + d2fdsig3dsig3*dsig3depse2)
      jac[2,2] = 1 + x[3]*(d2fdsig3dsig1*dsig1depse3 + d2fdsig2dsig3*dsig2depse3 + d2fdsig3dsig3*dsig3depse3)
      jac[2,3] = dfdsig3

      jac[3,0] = dfdsig1*dsig1depse1 + dfdsig2*dsig2depse1 + dfdsig3*dsig3depse1
      jac[3,1] = dfdsig1*dsig1depse2 + dfdsig2*dsig2depse2 + dfdsig3*dsig3depse2
      jac[3,2] = dfdsig1*dsig1depse3 + dfdsig2*dsig2depse3 + dfdsig3*dsig3depse3
      jac[3,3] = 0

      # Solve system of equations
      dx = np.linalg.solve(jac, -res) # increment of target variables

      # Update x
      x = x + dx

      # Compute error
      err = np.linalg.norm(dx)

      print(" Newton iter.",ii, ": err =", err)

      if err < tol:
        break
      
    # Update strain
    eps   = eps + deps
    eps_e = x[0]*np.tensordot(n1,n1,axes=0) + x[1]*np.tensordot(n2,n2,axes=0) + x[2]*np.tensordot(n3,n3,axes=0)
    eps_p = eps - eps_e
    lamda = lamda + x[3]

    # Update stress
    sigma1 = a*x[0] + b*x[1] + b*x[2] 
    sigma2 = b*x[0] + a*x[1] + b*x[2] 
    sigma3 = b*x[0] + b*x[1] + a*x[2] 
    sigma  = sigma1*np.tensordot(n1,n1,axes=0) + sigma2*np.tensordot(n2,n2,axes=0) + sigma3*np.tensordot(n3,n3,axes=0)

  # [4] Record stress and strain
  sigma11[i+1] = sigma[0,0]
  eps11[i+1]   = eps[0,0]  




# Plot stress-strain curve
plt.figure(0,figsize=(7,7))
plt.plot(eps11, sigma11, 'r-', linewidth=1.0, label="NN prediction")
plt.xlabel(r'$\epsilon_{11}$', fontsize=15)
plt.ylabel(r'$\sigma_{11}$ [MPa]', fontsize=15)
plt.axhline(0, color = 'k',alpha = 0.5)
plt.axvline(0, color = 'k',alpha = 0.5)
plt.xlim(-0.015, 0.015)
plt.ylim(-600, 600)
plt.show()