# ==============================================================
# Step 1: data augmentation
# written by: Hyoung Suk Suh & Chulmin Kweon (Columbia Univ.)
# ==============================================================

# Import necessary packages and functions
import os
import matplotlib.pyplot as plt
import pandas as pd
import autograd.numpy as np

from util.coordinate_transforms import *


# INPUT --------------------------------------------------------
# Number of discretized level-set-augmented points
N_level = 15

# File name
file_name = "Tresca"
# --------------------------------------------------------------




# Specify output directory
isExist = os.path.exists(file_name)
if not isExist:
  os.makedirs(file_name)




# Load raw data
data = pd.read_csv("../raw_data/raw_data_"+file_name+".csv")

# >> stress point cloud at f = 0
p     = data["p"].values
rho   = data["rho"].values
theta = data["theta"].values
lamda = data["lamda"].values

p     = p.reshape((-1,1))
rho   = rho.reshape((-1,1))
theta = theta.reshape((-1,1))
lamda = lamda.reshape((-1,1))




# Level set data augmentation along the rho-axis s.t. dfdrho = 1
delta = 1e-5

levels = np.linspace(0, 2, N_level)
levels[0] = delta # avoid center of pi-plane

tmp = 0
for lv in levels:

  # >> data augmentation
  p_aug     = np.copy(p)
  rho_aug   = lv*np.copy(rho)
  theta_aug = np.copy(theta)
  lamda_aug = np.copy(lamda)
  f_aug     = (lv-1)*np.copy(rho)

  # >> compute stress gradients based on augmented points
  dfdsig1_aug = np.zeros_like(p_aug)
  dfdsig2_aug = np.zeros_like(p_aug)
  dfdsig3_aug = np.zeros_like(p_aug)

  for i in range(np.shape(p_aug)[0]):
    conv_mat = convert_dfdprt_to_dfd123(p_aug[i,0], rho_aug[i,0], theta_aug[i,0])

    dfdprt = np.zeros(3)
    dfdprt[0] = 0. # dfdp
    dfdprt[1] = 1. # dfdrho
    dfdprt[2] = 0. # dfdtheta

    dfdsig_tmp = np.dot(dfdprt, conv_mat)
    dfdsig_tmp = dfdsig_tmp / np.linalg.norm(dfdsig_tmp)

    dfdsig1_aug[i,0] = dfdsig_tmp[0]
    dfdsig2_aug[i,0] = dfdsig_tmp[1]
    dfdsig3_aug[i,0] = dfdsig_tmp[2]

  if tmp == 0:
    p_level_set     = p_aug
    rho_level_set   = rho_aug
    theta_level_set = theta_aug
    lamda_level_set = lamda_aug

    dfdsig1_level_set = dfdsig1_aug
    dfdsig2_level_set = dfdsig2_aug
    dfdsig3_level_set = dfdsig3_aug

    f_level_set = f_aug

  else:
    p_level_set     = np.concatenate((p_level_set, np.copy(p_aug)),         axis = 0)
    rho_level_set   = np.concatenate((rho_level_set, np.copy(rho_aug)),     axis = 0)
    theta_level_set = np.concatenate((theta_level_set, np.copy(theta_aug)), axis = 0)
    lamda_level_set = np.concatenate((lamda_level_set, np.copy(lamda_aug)), axis = 0)

    dfdsig1_level_set = np.concatenate((dfdsig1_level_set, np.copy(dfdsig1_aug)), axis = 0)
    dfdsig2_level_set = np.concatenate((dfdsig2_level_set, np.copy(dfdsig2_aug)), axis = 0)
    dfdsig3_level_set = np.concatenate((dfdsig3_level_set, np.copy(dfdsig3_aug)), axis = 0)

    f_level_set = np.concatenate((f_level_set, np.copy(f_aug)), axis = 0)

  tmp = tmp+1




# Save output
print(">> Number of stress data points generated (augmented):", np.shape(p_level_set)[0])

data = np.hstack((p_level_set, rho_level_set, theta_level_set, lamda_level_set, \
                  dfdsig1_level_set, dfdsig2_level_set, dfdsig3_level_set,      \
                  f_level_set))

output = pd.DataFrame({'p':       data[:, 0], \
                       'rho':     data[:, 1], \
                       'theta':   data[:, 2], \
                       'lamda':   data[:, 3], \
                       'dfdsig1': data[:, 4], \
                       'dfdsig2': data[:, 5], \
                       'dfdsig3': data[:, 6], \
                       'f':       data[:, 7]})

output.to_csv("./"+file_name+"/training_data.csv", index = False)