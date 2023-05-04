from utils.freq_imports import *


# pca pose prior term
def compute_I_minus_Pi_M_PiT(Pi, Sigma, w_pca, w_pca_mean):
    # Ref: Appendix A in https://lgg.epfl.ch/publications/2015/Htrack_ICP/paper.pdf
    # w_pca is w_4, w_pca_mean is w_5
    
    M_inv = 1/(w_pca+np.finfo(float).eps) * (w_pca_mean*Sigma**2 + w_pca*np.identity(len(Sigma)))   # (20, 20)
    M = np.zeros_like(M_inv)    # (20, 20)
    np.fill_diagonal(M, 1 / (M_inv.diagonal() + +np.finfo(float).eps))

    I = np.identity(len(Sigma))
    I_minus_Pi_M_PiT = I - Pi @ M @ Pi.T        # (20, 20)

    return I_minus_Pi_M_PiT
