import autograd.numpy as np

def convert_prt_to_123(p, rho, theta):

  sigma1pp = rho*np.cos(theta)
  sigma2pp = rho*np.sin(theta)
  sigma3pp = np.sqrt(3)*p

  sigma_pp = np.array([sigma1pp, sigma2pp, sigma3pp])

  R = np.array([[ np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3],
                [            0,  np.sqrt(6)/3, np.sqrt(3)/3],
                [-np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3]])

  sigma = np.dot(R, sigma_pp)

  sigma1 = sigma[0]
  sigma2 = sigma[1]
  sigma3 = sigma[2]

  return sigma1, sigma2, sigma3

def convert_123_to_prt(sigma1, sigma2, sigma3):

  sigma = np.array([sigma1, sigma2, sigma3])

  Rinv = np.array([[ np.sqrt(2)/2,             0, -np.sqrt(2)/2],
                   [-np.sqrt(6)/6,  np.sqrt(6)/3, -np.sqrt(6)/6],
                   [ np.sqrt(3)/3,  np.sqrt(3)/3,  np.sqrt(3)/3]])

  sigma_pp = np.dot(Rinv, sigma)

  sigma1_pp = sigma_pp[0]
  sigma2_pp = sigma_pp[1]
  sigma3_pp = sigma_pp[2]

  rho   = np.sqrt(sigma1_pp**2 + sigma2_pp**2)
  theta = np.arctan2(sigma2_pp, sigma1_pp)
  p     = (1/np.sqrt(3))*sigma3_pp

  if theta < 0:
    theta = theta + 2*np.pi

  return p, rho, theta

def convert_dfdprt_to_dfd123(p, rho, theta):

  sigma1_pp = rho*np.cos(theta)
  sigma2_pp = rho*np.sin(theta)
  sigma3_pp = np.sqrt(3)*p

  sigma_pp = np.array([sigma1_pp, sigma2_pp, sigma3_pp])

  R = np.array([[ np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3],
                [            0,  np.sqrt(6)/3, np.sqrt(3)/3],
                [-np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3]])

  sigma = np.dot(R, sigma_pp)

  sigma1 = sigma[0]
  sigma2 = sigma[1]
  sigma3 = sigma[2]

  dpdsig1 = 1./3.
  dpdsig2 = 1./3.
  dpdsig3 = 1./3.

  if np.abs(sigma1**2 + sigma2**2 + sigma3**2 - sigma1*sigma2 - sigma2*sigma3 - sigma3*sigma1) < 1e-12:
    denom = 1e-12
  else:
    denom = 6*np.sqrt(sigma1**2 + sigma2**2 + sigma3**2 - sigma1*sigma2 - sigma2*sigma3 - sigma3*sigma1)

  drhodsig1 = (np.sqrt(6)*(2*sigma1 - sigma2 - sigma3)) / denom
  drhodsig2 = (np.sqrt(6)*(2*sigma2 - sigma3 - sigma1)) / denom
  drhodsig3 = (np.sqrt(6)*(2*sigma3 - sigma1 - sigma2)) / denom

  tmp = sigma2_pp / sigma1_pp
  dthetadsig1 = 1./(1+ tmp**2)/sigma1_pp**2 * (-np.sqrt(6.)/6. *sigma1_pp - np.sqrt(2.)/2.*sigma2_pp)
  dthetadsig2 = 1./(1+ tmp**2)/sigma1_pp**2 * ( np.sqrt(6.)/3. *sigma1_pp )
  dthetadsig3 = 1./(1+ tmp**2)/sigma1_pp**2 * (-np.sqrt(6.)/6. *sigma1_pp + np.sqrt(2.)/2.*sigma2_pp)

  A = np.array([[dpdsig1, dpdsig2, dpdsig3],
                [drhodsig1, drhodsig2, drhodsig3],
                [dthetadsig1, dthetadsig2, dthetadsig3]])

  return A