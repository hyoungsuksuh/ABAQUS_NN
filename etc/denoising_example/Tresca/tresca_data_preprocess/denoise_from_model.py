# Provided by Mian Xiao (Columbia University)
# >> project noisy data onto the smooth yield manifold

import numpy as np
import scipy.optimize as opt
from scipy.spatial import cKDTree
from scipy.linalg import null_space
import torch
import torch.nn as nn

# neural network architecture
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return x


"""
    define a class to compute projection
"""
class Compute_project:
    def __init__(self, patch_models, patch_tx):
        self.phi = patch_models
        self.transform = patch_tx
    def config(self, target, patch_id):
        self.target = np.copy(target)
        self.idx_i = np.copy(patch_id)
        self.phi_i = self.phi[patch_id]
        translate_i, scale_i, rotate_i = self.transform[patch_id]
        self.trans_i = translate_i
        self.scale_i = scale_i
        self.rotat_i = rotate_i
    def compute(self):
        def phi(v):
            y = self.phi_i( torch.tensor(v.reshape(1,-1)).float() )
            yt = (y.squeeze() @ self.rotat_i.transpose(0, 1)) / self.scale_i - self.trans_i
            yt = yt.detach().numpy().reshape(-1)
            return yt
        def phi1(v):
            return phi(v)[0]
        def phi2(v):
            return phi(v)[1]
        def phi3(v):
            return phi(v)[2]
        def phi_grad(v):
            phi1_p = opt.approx_fprime(v, phi1, epsilon=1e-6)
            phi2_p = opt.approx_fprime(v, phi2, epsilon=1e-6)
            phi3_p = opt.approx_fprime(v, phi3, epsilon=1e-6)
            return np.stack([phi1_p, phi2_p, phi3_p])
        # projection objective function (Eucledian distance)
        def dist(v):
            diff = phi(v) - self.target
            return diff
        # optimization constraint
        x0 = np.array([0.5,0.5])
        res = opt.least_squares(dist, x0, jac=phi_grad, bounds=(0.,1.))
        soln = res.x
        proj = phi(soln)
        return res.cost, proj


# load the trained model
geom_recon_file = 'Tresca_noise1.pt'
saved_dict = torch.load(geom_recon_file)
patch_ctr = saved_dict["patch_ctr"]
patch_txs = saved_dict["patch_txs"]
num_patches = len(patch_txs)
geom_models = nn.ModuleList([MLP(2, 3) for i in range(num_patches)])
geom_models.load_state_dict(saved_dict["final_model"])
# configure a kdTrees to query neighborhood points
kdtree = cKDTree(patch_ctr)
find_proj = Compute_project(geom_models, patch_txs)

# projection routine
def proj_to_surf(x):
    _, neighbors = kdtree.query(x, k=16)
    find_proj.config(x, neighbors[0])
    _, proj = find_proj.compute()
    return proj

# load noisy data
data = np.genfromtxt('Tresca_train_data_noise.csv', delimiter=',')
data_denoise = np.zeros_like(data)
for i in range(data.shape[0]):
    data_denoise[i,:] = proj_to_surf(data[i,:])
    if (i+1)%100 == 0: print('finish projection to {} points'.format(i+1))
np.savetxt('Tresca_train_data_denoise.csv', data_denoise, fmt='%.6g,%.6g,%.6g')










