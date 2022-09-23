# ==============================================================
# raw data generation - for level set plasticity
# written by: Hyoung Suk Suh & Chulmin Kweon (Columbia Univ.)
# ==============================================================

# Import necessary packages and functions
import os
import numpy as np
import pandas as pd

# INPUT --------------------------------------------------------
# Denoised?
denoised = 0

# File name
file_name = "Tresca"
# --------------------------------------------------------------




# Read original dataset
if denoised:
  data_sample = pd.read_csv("./"+file_name+"_train_data_denoise.csv", header=None)
else:
  data_sample = pd.read_csv("./"+file_name+"_train_data_noise.csv", header=None)
    
data_sample = data_sample.to_numpy()
sigma1pp = data_sample[:,0]
sigma2pp = data_sample[:,1]
sigma3pp = data_sample[:,2]




# Convert (sigma1pp, sigma2pp, sigma3pp) into (p, rho, theta) 
p     = np.zeros_like(sigma1pp)
rho   = np.zeros_like(sigma1pp)
theta = np.zeros_like(sigma1pp)

for i in range(np.shape(p)[0]):
  p[i]     = (1./np.sqrt(3))*sigma3pp[i]
  rho[i]   = np.sqrt(sigma1pp[i]**2 + sigma2pp[i]**2)
  theta[i] = np.arctan2(sigma2pp[i], sigma1pp[i])

  if theta[i] < 0.0:
    theta[i] = theta[i] + 2.*np.pi




# Save data
p     = p.reshape((-1,1))
rho   = rho.reshape((-1,1))
theta = theta.reshape((-1,1))

data = np.hstack((p, rho, theta))

output = pd.DataFrame({'p':       data[:, 0], \
                       'rho':     data[:, 1], \
                       'theta':   data[:, 2]})

if denoised:
  output.to_csv("../raw_data_"+file_name+"_without_noise.csv", index = False)
else:
  output.to_csv("../raw_data_"+file_name+"_with_noise.csv", index = False)