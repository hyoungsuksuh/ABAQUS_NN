# Provided by Mian Xiao (Columbia University)
# >> generate resampled datapoints for closure testing

import numpy as np
import torch
import torch.nn as nn
import point_cloud_utils as pcu

# neural network model
class MLP(nn.Module):
    """
    A simple fully connected network mapping vectors in dimension in_dim to vectors in dimension out_dim
    """
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
    routines for upsampling a grid
"""

def meshgrid_vertices(w, urange=[0, 1], vrange=[0, 1]):
    g = np.mgrid[urange[0]:urange[1]:complex(w), vrange[0]:vrange[1]:complex(w)]
    v = np.vstack(map(np.ravel, g)).T
    return np.ascontiguousarray(v)

def meshgrid_from_lloyd_ts(model_ts, n, scale=1.0):
    model_ts_min = np.min(model_ts, axis=0)
    model_ts_max = np.max(model_ts, axis=0)
    urange = np.array([model_ts_min[0], model_ts_max[0]])
    vrange = np.array([model_ts_min[1], model_ts_max[1]])
    ctr_u = np.mean(urange)
    ctr_v = np.mean(vrange)
    urange = (urange - ctr_u) * scale + ctr_u
    vrange = (vrange - ctr_v) * scale + ctr_v
    return meshgrid_vertices(n, urange=urange, vrange=vrange)


def upsample_surface(patch_uvs, patch_tx, patch_models, devices, scale=1.0, num_samples=12, normal_samples=32,
                     num_batches=1, compute_normals=True):
    vertices = []
    normals = []
    indices = []
    num_patches = len(patch_models)
    batch_size = int(np.ceil(num_patches / num_batches))
    with torch.no_grad():
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, num_patches)
            for i in range(start_idx, end_idx):
                dev_i = devices[i % len(devices)]
                patch_models[i] = patch_models[i].to(dev_i)
                patch_uvs[i] = patch_uvs[i].to(dev_i)
                patch_tx[i] = tuple(txj.to(dev_i) for txj in patch_tx[i])

            for i in range(start_idx, end_idx):
                if (i + 1) % 10 == 0:
                    print("Upsamling %d/%d" % (i + 1, len(patch_models)))

                device = devices[i % len(devices)]

                n = num_samples
                translate_i, scale_i, rotate_i = (patch_tx[i][j].to(device) for j in range(len(patch_tx[i])))
                uv_i = meshgrid_from_lloyd_ts(patch_uvs[i].cpu().numpy(), n, scale=scale).astype(np.float32)
                uv_i = torch.from_numpy(uv_i).to(patch_uvs[i])
                y_i = patch_models[i](uv_i)

                mesh_v = ((y_i.squeeze() @ rotate_i.transpose(0, 1)) / scale_i - translate_i).cpu().numpy()

                if compute_normals:
                    mesh_f = utils.meshgrid_face_indices(n)
                    mesh_n = pcu.per_vertex_normals(mesh_v, mesh_f)
                    normals.append(mesh_n)

                vertices.append(mesh_v)
                indices .append(i * np.ones((mesh_v.shape[0], 1)))
            for i in range(start_idx, end_idx):
                dev_i = 'cpu'
                patch_models[i] = patch_models[i].to(dev_i)
                patch_uvs[i] = patch_uvs[i].to(dev_i)
                patch_tx[i] = tuple(txj.to(dev_i) for txj in patch_tx[i])

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    indices = np.concatenate(indices, axis=0)
    if compute_normals:
        normals = np.concatenate(normals, axis=0).astype(np.float32)
    else:
        print("Fixing normals...")
        normals = np.zeros_like(vertices)

    return vertices, normals, indices


# load saved model dictionary
data_dict = torch.load('Tresca_noise1.pt')
# parametric grids (u,v) for each patch
patch_uvs = data_dict["patch_uvs"]
# transformation policy for each patch
patch_tx  = data_dict["patch_txs"]
# define the neural network model (for patch embedding functions)
num_patches = len(patch_uvs); print("num of patches:", num_patches)
phi = nn.ModuleList([MLP(2,3) for i in range(num_patches)])
# and load the trained weights
phi.load_state_dict(data_dict["final_model"])


# use CPU
devices = ["cpu"]
# density of resampled grid for each patch
resample_resoln = 30
# call the upsampling routine
v, _, _ = upsample_surface(patch_uvs, patch_tx, phi, devices,
    scale=1.0, num_samples=resample_resoln,
    normal_samples=64, num_batches=1, compute_normals=False)

# take the cross section at some mean pressure level
# aproximated by -0.4 < p < 0.4
p_ = 39.44 * np.sqrt(3)
v_sec = v[ (v[:,2] > p_-0.5) & (v[:,2] < p_+0.5) , :]
print(v_sec.shape)
np.savetxt('data_resampled.csv', v_sec, fmt='%.6g,%.6g,%.6g')