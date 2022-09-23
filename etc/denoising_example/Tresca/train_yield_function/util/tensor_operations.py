import autograd.numpy as np

# tensor outer product: C_ijkl = A_ij B_kl
def tensor_oMult(A, B):
  # A, B: 2nd tensors
  # return: 4th tensor
  assert(A.shape == B.shape)
  nDim = A.shape[0]
  res = np.zeros((nDim,nDim,nDim,nDim))
  for i in range(nDim):
    for j in range(nDim):
      for k in range(nDim):
        for l in range(nDim):
          res[i,j,k,l] = A[i,j] * B[k,l]
  return res

# tensor oPlus operation: C_ijkl = A_jl B_ik
def tensor_oPlus(A, B):
  # A, B: 2nd tensors
  # return: 4th tensor
  assert(A.shape == B.shape)
  nDim = A.shape[0]
  res = np.zeros((nDim,nDim,nDim,nDim))
  for i in range(nDim):
    for j in range(nDim):
      for k in range(nDim):
        for l in range(nDim):
          res[i,j,k,l] = A[j,l] * B[i,k]
  return res

# tensor oMinus operation: C_ijkl = A_il B_jk
def tensor_oMinus(A, B):
  # A, B: 2nd tensors
  # return: 4th tensor
  assert(A.shape == B.shape)
  nDim = A.shape[0]
  res = np.zeros((nDim,nDim,nDim,nDim))
  for i in range(nDim):
    for j in range(nDim):
      for k in range(nDim):
        for l in range(nDim):
          res[i,j,k,l] = A[i,l] * B[j,k]
  return res

# compute the 4th order identity tensor II
# such that for any symmetric 2nd order tensor A_ij, II_ijkl A_kl = A_ij
def identity_4(nDim):
  I = np.eye(nDim)
  res = np.zeros((nDim,nDim,nDim,nDim))
  for i in range(nDim):
    for j in range(nDim):
      for k in range(nDim):
        for l in range(nDim):
          res[i,j,k,l] = (I[i,l] * I[j,k] + I[i,k] * I[j,l]) / 2.
  return res