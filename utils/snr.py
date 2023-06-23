"""
Utilities functions for SNR
"""
import numpy as np

# -------------------------------------------------------------------------------------------------

def create_pseudo_replica(im, added_noise_sd=0.1, N=100):
    """Perform the pseudo replica data
    """
    n = np.random.randn(*im.shape, N) * added_noise_sd
    im_noised = im[..., np.newaxis] + n
    return im_noised

# -------------------------------------------------------------------------------------------------

if __name__=="__main__":
    pass
