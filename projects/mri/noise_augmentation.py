"""
Noise augmentation utilities

provides a wide range of utility function used to create training data for MRI

copied from Hui's original commit in CNNT
"""
import time
import os
import math
import numpy  as np
#from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, ifftshift, fftshift
from scipy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, ifftshift, fftshift
from colorama import Fore, Style

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *

# --------------------------------------------------------------

def centered_pad(data, new_shape):
    padding = (np.array(new_shape) - np.array(data.shape))//2

    data_padded = np.pad(data,[(padding[0],padding[0]),(padding[1],padding[1])])

    scaling = np.sqrt(np.prod(new_shape))/np.sqrt(np.prod(data.shape))
    data_padded *= scaling

    return data_padded

# --------------------------------------------------------------

def create_complex_noise(noise_sigma, size):
    nns = np.random.standard_normal(size=size)+np.random.standard_normal(size=size)*1j
    return (noise_sigma * nns).astype(np.complex64)

# --------------------------------------------------------------

def centered_fft(image, norm='ortho'):
    return fftshift(fft2(ifftshift(image),norm=norm))

def fft1c(image, norm='ortho'):
    """Perform centered 1D fft

    Args:
        image ([RO, ...]): Perform fft2c on the first dimension
        norm : 'ortho' or 'backward'
    Returns:
        res: fft1c results
    """
    return fftshift(fft(ifftshift(image, axes=(0,)), axis=0, norm=norm), axes=(0,))

def fft2c(image, norm='ortho'):
    """Perform centered 2D fft

    Args:
        image ([RO, E1, ...]): Perform fft2c on the first two dimensions
        norm : 'ortho' or 'backward'
    Returns:
        res: fft2c results
    """
    return fftshift(fft2(ifftshift(image, axes=(0,1)), axes=(0,1), norm=norm), axes=(0,1))

def fft3c(image, norm='ortho'):
    """Perform centered 3D fft

    Args:
        image ([RO, E1, E2, ...]): Perform fft3c on the first three dimensions
        norm : 'ortho' or 'backward'
    Returns:
        res: fft3c results
    """
    return fftshift(fftn(ifftshift(image, axes=(0,1,2)), axes=(0,1,2), norm=norm), axes=(0,1,2))

# --------------------------------------------------------------

def centered_ifft(kspace, norm='ortho'):
    return fftshift(ifft2(ifftshift(kspace),norm=norm))

def ifft1c(kspace, norm='ortho'):
    """Perform centered 1D ifft

    Args:
        image ([RO, ...]): Perform fft2c on the first dimension
        norm : 'ortho' or 'backward'
    Returns:
        res: fft1c results
    """
    return fftshift(ifft(ifftshift(kspace, axes=(0,)), axis=0, norm=norm), axes=(0,))

def ifft2c(kspace, norm='ortho'):
    """Perform centered 2D ifft

    Args:
        image ([RO, E1, ...]): Perform fft2c on the first two dimensions
        norm : 'ortho' or 'backward'
    Returns:
        res: fft2c results
    """
    return fftshift(ifft2(ifftshift(kspace, axes=(0,1)), axes=(0,1), norm=norm), axes=(0,1))

def ifft3c(kspace, norm='ortho'):
    """Perform centered 2D ifft

    Args:
        image ([RO, E1, E2, ...]): Perform fft3c on the first two dimensions
        norm : 'ortho' or 'backward'
    Returns:
        res: fft3c results
    """
    return fftshift(ifftn(ifftshift(kspace, axes=(0,1,2)), axes=(0,1,2), norm=norm), axes=(0,1,2))

# --------------------------------------------------------------

def generate_symmetric_filter(len, filterType, sigma=1.5, width=10, snr_scaling=True):
    """Compute the SNR unit symmetric filter

    Args:
        len (int): length of filter
        filterType (str): Gaussian or None
        sigma (float, optional): sigma for gaussian filter. Defaults to 1.5.
        snr_scaling (bool): if True, keep the noise level; if False, keep the signal level
        
    Returns:
        filter: len array
    """

    filter = np.ones(len, dtype=np.float32)

    if (filterType == "Gaussian") and sigma>0:
        r = -1.0*sigma*sigma / 2

        if (len % 2 == 0):
            # to make sure the zero points match and boundary of filters are symmetric
            stepSize = 2.0 / (len - 2)
            x = np.zeros(len - 1)

            for ii in range(len-1):
                x[ii] = -1 + ii*stepSize

            for ii in range(len-1):
                filter[ii + 1] = math.exp(r*(x[ii] * x[ii]))

            filter[0] = 0
        else:
            stepSize = 2.0 / (len - 1)
            x = np.zeros(len)

            for ii in range(len):
                x[ii] = -1 + ii*stepSize

            for ii in range(len):
                filter[ii] = math.exp(r*(x[ii] * x[ii]))

    if snr_scaling:
        sos = np.sum(filter*filter)
        filter /= math.sqrt(sos / len)
    else:
        filter /= np.max(filter)

    return filter

# --------------------------------------------------------------

def generate_asymmetric_filter(len, start, end, filterType='TapperedHanning', width=10):
    """Create the asymmetric kspace filter

    Args:
        len (int): length of the filter
        start (int): start of filter
        end (int): end of the filter
        filterType (str): None or TapperedHanning   
        width (int, optional): width of transition band. Defaults to 10.
    """
    
    if (start > len - 1):
        start = 0
        
    if (end > len - 1):
        end = len - 1

    if (start > end):
        start = 0
        end = len - 1

    filter = np.zeros(len, dtype=np.float32)

    for ii in range(start, end+1):
        filter[ii] = 1.0

    if (width == 0 or width >= len):
        width = 1

    w = np.ones(width)

    if (filterType == "TapperedHanning"):
        for ii in range(1, width+1):
            w[ii - 1] = 0.5 * (1 - math.cos(2.0*math.pi*ii / (2 * width + 1)))
    
    if (start == 0 and end == len - 1):
        for ii in range(1, width+1):
            filter[ii - 1] = w[ii - 1]
            filter[len - ii] = filter[ii - 1]

    if (start == 0 and end<len - 1):
        for ii in range(1, width+1):
            filter[end - ii + 1] = w[ii - 1]

    if (start>0 and end == len - 1):
        for ii in range(1, width+1):
            filter[start + ii - 1] = w[ii - 1]

    if (start>0 and end<len - 1):
        for ii in range(1, width+1):
            filter[start + ii - 1] = w[ii - 1]
            filter[end - ii + 1] = w[ii - 1]

    sos = np.sum(filter*filter)
    #filter /= math.sqrt(sos / (end - start + 1))
    filter /= math.sqrt(sos / (len))

    return filter

# --------------------------------------------------------------

def apply_kspace_filter_1D(kspace, fRO):
    """Apply the 1D kspace filter

    Args:
        kspace ([RO, E1, CHA, PHS]): kspace, can be 1D, 2D or 3D or 4D
        fRO ([RO]): kspace fitler along RO

    Returns:
        kspace_filtered: filtered ksapce
    """

    RO = kspace.shape[0]
    assert fRO.shape[0] == RO

    if(kspace.ndim==1):
        kspace_filtered = kspace * fRO
    if(kspace.ndim==2):
        kspace_filtered = kspace * fRO.reshape((RO, 1))
    if(kspace.ndim==3):
        kspace_filtered = kspace * fRO.reshape((RO, 1, 1))
    if(kspace.ndim==4):
        kspace_filtered = kspace * fRO.reshape((RO, 1, 1, 1))

    return kspace_filtered

# --------------------------------------------------------------

def apply_kspace_filter_2D(kspace, fRO, fE1):
    """Apply the 2D kspace filter

    Args:
        kspace ([RO, E1, CHA, PHS]): kspace, can be 2D or 3D or 4D
        fRO ([RO]): kspace fitler along RO
        fE1 ([E1]): kspace filter along E1

    Returns:
        kspace_filtered: filtered ksapce
    """

    RO = kspace.shape[0]
    E1 = kspace.shape[1]
    
    assert fRO.shape[0] == RO
    assert fE1.shape[0] == E1

    filter2D = np.outer(fRO, fE1)
    
    if(kspace.ndim==2):
        kspace_filtered = kspace * filter2D
    if(kspace.ndim==3):
        kspace_filtered = kspace * filter2D[:,:,np.newaxis]
    if(kspace.ndim==4):
        kspace_filtered = kspace * filter2D[:,:,np.newaxis,np.newaxis]

    return kspace_filtered

# --------------------------------------------------------------

def apply_resolution_reduction_2D(im, ratio_RO, ratio_E1, snr_scaling=True, norm = 'ortho'):
    """Add resolution reduction, keep the image matrix size

    Inputs:
        im: complex image [RO, E1, ...]
        ratio_RO, ratio_E1: ratio to reduce resolution, e.g. 0.75 for 75% resolution
        snr_scaling : if True, apply SNR scaling
        norm : backward or ortho
        
        snr_scaling should be False and norm should be backward to preserve signal level
    Returns:
        res: complex image with reduced phase resolution [RO, E1, ...]
        fRO, fE1 : equivalent kspace filter
    """
       
    kspace = fft2c(im, norm=norm)
    
    RO = kspace.shape[0]
    E1 = kspace.shape[1]

    assert ratio_RO <= 1.0 and ratio_RO > 0
    assert ratio_E1 <= 1.0 and ratio_E1 > 0
        
    num_masked_RO = int((RO-ratio_RO*RO) // 2)
    num_masked_E1 = int((E1-ratio_E1*E1) // 2)
    
    fRO = np.ones(RO)
    fE1 = np.ones(E1)
    
    if(kspace.ndim==2):
        if(num_masked_RO>0):
            kspace[0:num_masked_RO, :] = 0
            kspace[RO-num_masked_RO:RO, :] = 0

        if(num_masked_RO>0):
            kspace[:, 0:num_masked_E1] = 0
            kspace[:, E1-num_masked_E1:E1] = 0
            
    if(kspace.ndim==3):
        if(num_masked_RO>0):
            kspace[0:num_masked_RO, :, :] = 0
            kspace[RO-num_masked_RO:RO, :, :] = 0

        if(num_masked_RO>0):
            kspace[:, 0:num_masked_E1, :] = 0
            kspace[:, E1-num_masked_E1:E1, :] = 0

    if(kspace.ndim==4):
        if(num_masked_RO>0):
            kspace[0:num_masked_RO, :, :, :] = 0
            kspace[RO-num_masked_RO:RO, :, :, :] = 0

        if(num_masked_RO>0):
            kspace[:, 0:num_masked_E1, :, :] = 0
            kspace[:, E1-num_masked_E1:E1, :, :] = 0
            
    fRO[0:num_masked_RO] = 0
    fRO[RO-num_masked_RO:RO] = 0
    
    fE1[0:num_masked_E1] = 0
    fE1[E1-num_masked_E1:E1] = 0
    
    if(snr_scaling is True):
        ratio = math.sqrt(RO*E1)/math.sqrt( (RO-2*num_masked_RO) * (E1-2*num_masked_E1))
        im_low_res = ifft2c(kspace) * ratio
    else:
        im_low_res = ifft2c(kspace, norm=norm)

    return im_low_res, fRO, fE1

# --------------------------------------------------------------

def apply_image_filter(data, sigma_RO=1.25, sigma_E1=1.25):
    # apply image filter, keep the signal level
    # data : [RO, E1, T, ...]
    
    fRO = generate_symmetric_filter(data.shape[0], filterType="Gaussian", sigma=sigma_RO, width=10, snr_scaling=False)
    fE1 = generate_symmetric_filter(data.shape[1], filterType="Gaussian", sigma=sigma_E1, width=10, snr_scaling=False)
    data_filtered = ifft2c(apply_kspace_filter_2D(fft2c(data), fRO, fE1))
    return data_filtered, fRO, fE1

def apply_image_filter_T(data, sigma_T=1.25):
    # apply image filter along the T, keep the signal level
    # data : [RO, E1, T, ...]
    
    fT = generate_symmetric_filter(data.shape[2], filterType="Gaussian", sigma=sigma_T, width=10, snr_scaling=False)
    
    im = np.transpose(data, (2, 0, 1))
    im = ifft1c(apply_kspace_filter_1D(fft1c(im), fT))
    return np.transpose(im, (1, 2, 0)), fT
    
# --------------------------------------------------------------

def apply_matrix_size_reduction_2D(im, dst_RO, dst_E1, norm = 'ortho'):
    """Apply the matrix size reduction, keep the FOV

    Inputs:
        im: complex image [RO, E1, ...]
        dst_RO, dst_E1: target matrix size
        norm : backward or ortho
        
    Returns:
        res: complex image with reduced matrix size [dst_RO, dst_E1, ...]
    """

    RO = im.shape[0]
    E1 = im.shape[1]
    
    assert dst_RO<=RO
    assert dst_E1<=E1

    kspace = fft2c(im, norm=norm)
           
    num_ro = int((RO-dst_RO)//2)
    num_e1 = int((E1-dst_E1)//2)
       
    if(kspace.ndim==2):
        kspace_dst = kspace[num_ro:num_ro+dst_RO, num_e1:num_e1+dst_E1]
    if(kspace.ndim==3):
        kspace_dst = kspace[num_ro:num_ro+dst_RO, num_e1:num_e1+dst_E1,:]
    if(kspace.ndim==4):
        kspace_dst = kspace[num_ro:num_ro+dst_RO, num_e1:num_e1+dst_E1,:,:]
            
    res = ifft2c(kspace_dst, norm=norm)

    return res

# --------------------------------------------------------------

def zero_padding_resize_2D(im, dst_RO, dst_E1, snr_scaling=True, norm = 'ortho'):
    """zero padding resize up the image

    Args:
        im ([RO, E1, ...]): complex image
        dst_RO (int): destination size
        dst_E1 (int): destination size
        norm : backward or ortho
    """
    
    RO = im.shape[0]
    E1 = im.shape[1]
    
    assert dst_RO>=RO and dst_E1>=E1
    
    kspace = fft2c(im, norm=norm)
    
    new_shape = list(im.shape)
    new_shape[0] = dst_RO
    new_shape[1] = dst_E1
    padding = (np.array(new_shape) - np.array(im.shape))//2

    if(im.ndim==2):
        data_padded = np.pad(kspace, [(padding[0],padding[0]),(padding[1],padding[1])])
        
    if(im.ndim==3):
        data_padded = np.pad(kspace, [(padding[0],padding[0]),(padding[1],padding[1]), (0, 0)])
        
    if(im.ndim==4):
        data_padded = np.pad(kspace, [(padding[0],padding[0]),(padding[1],padding[1]), (0, 0), (0, 0)])

    if(snr_scaling is True):
        scaling = np.sqrt(dst_RO*dst_E1)/np.sqrt(RO*E1)
        data_padded *= scaling
    
    im_padded = ifft2c(data_padded, norm=norm)
    
    return im_padded

# --------------------------------------------------------------

def adjust_matrix_size(data, ratio):
    """Adjust matrix size, uniform signal transformation

    Args:
        data ([RO, E1]): complex image
        ratio (float): <1.0, reduce matrix size; >1.0, increase matrix size; 1.0, do nothing
    """
    
    if(abs(ratio-1.0)<0.0001):
        return data
    
    RO, E1 = data.shape
    dst_RO = int(round(ratio*RO))
    dst_E1 = int(round(ratio*E1))
    
    if(ratio<1.0):        
        res_im = apply_matrix_size_reduction_2D(data, dst_RO, dst_E1)
        
    if(ratio>1.0):        
        res_im = zero_padding_resize_2D(data, dst_RO, dst_E1)
           
    return res_im

# --------------------------------------------------------------

def generate_3D_MR_correlated_noise(T=30, RO=192, E1=144, REP=1, 
                                    min_noise_level=3.0, 
                                    max_noise_level=7.0, 
                                    kspace_filter_sigma=[0, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
                                    pf_filter_ratio=[1.0, 0.875, 0.75, 0.625, 0.55],
                                    kspace_filter_T_sigma=[0, 0.5, 0.65, 0.85, 1.0, 1.5, 2.0, 2.25],
                                    phase_resolution_ratio=[1.0, 0.85, 0.7, 0.65, 0.55],
                                    readout_resolution_ratio=[1.0, 0.85, 0.7, 0.65, 0.55],
                                    rng=np.random.Generator(np.random.PCG64(int(time.time()))),
                                    only_white_noise=False,
                                    verbose=False):

    # create noise
    noise_sigma = (max_noise_level - min_noise_level) * rng.random() + min_noise_level
    if(REP>1):
        nns = create_complex_noise(noise_sigma, (T, RO, E1, REP))
    else:
        nns = create_complex_noise(noise_sigma, (T, RO, E1))

    if(verbose is True):
        print(f"{Fore.GREEN}------------------------------------------{Style.RESET_ALL}")
        print(f"noise sigma is {noise_sigma}")
        std_real = np.mean(np.std(np.real(nns), axis=3))
        std_imag = np.mean(np.std(np.imag(nns), axis=3))
        print("noise, real, std is ", std_real)
        print("noise, imag, std is ", std_imag)

    if only_white_noise:
        return nns, noise_sigma

    # apply resolution reduction
    ratio_RO = readout_resolution_ratio[rng.integers(0, len(readout_resolution_ratio))]
    ratio_E1 = phase_resolution_ratio[rng.integers(0, len(phase_resolution_ratio))]

    # no need to apply snr scaling here, but scale equally over time
    nn_reduced = [apply_resolution_reduction_2D(nn, ratio_RO, ratio_E1, snr_scaling=False) for nn in nns]

    nns = np.array([x for x,_,_ in nn_reduced])
    fdROs = np.array([y for _,y,_ in nn_reduced])
    fdE1s = np.array([z for _,_,z in nn_reduced])

    if(verbose is True):
        print("--" * 20)
        print(f"ratio_RO is {ratio_RO}, ratio_E1 is {ratio_RO}")

    # apply pf filter
    pf_lottery = rng.integers(0, 3) # 0, only 1st dim; 1, only 2nd dim; 2, both dim
    pf_ratio_RO = pf_filter_ratio[rng.integers(0, len(pf_filter_ratio))]
    pf_ratio_E1 = pf_filter_ratio[rng.integers(0, len(pf_filter_ratio))]

    if(rng.random()<0.5):
        start = 0
        end = int(pf_ratio_RO*RO)
    else:
        start = RO-int(pf_ratio_RO*RO)
        end = RO-1
    pf_fRO = generate_asymmetric_filter(RO, start, end, filterType='TapperedHanning', width=10)

    if(rng.random()<0.5):
        start = 0
        end = int(pf_ratio_E1*E1)
    else:
        start = E1-int(pf_ratio_E1*E1)
        end = E1-1
    pf_fE1 = generate_asymmetric_filter(E1, start, end, filterType='TapperedHanning', width=10)

    if(pf_lottery==0):
        pf_ratio_E1 = 1.0
        pf_fE1 = np.ones(E1)

    if(pf_lottery==1):
        pf_ratio_RO = 1.0
        pf_fRO = np.ones(RO)

    if(verbose is True):
        print("--" * 20)
        print(f"pf_lottery is {pf_lottery}, pf_ratio_RO is {pf_ratio_RO}, pf_ratio_E1 is {pf_ratio_E1}")

    # apply kspace filter
    ro_filter_sigma = kspace_filter_sigma[rng.integers(0, len(kspace_filter_sigma))]
    fRO = generate_symmetric_filter(RO, filterType="Gaussian", sigma=ro_filter_sigma, width=10)
    e1_filter_sigma = kspace_filter_sigma[rng.integers(0, len(kspace_filter_sigma))]
    fE1 = generate_symmetric_filter(E1, filterType="Gaussian", sigma=e1_filter_sigma, width=10)
    T_filter_sigma = kspace_filter_T_sigma[rng.integers(0, len(kspace_filter_T_sigma))]
    if np.random.uniform() < 0.5:
        # not always apply T filter
        T_filter_sigma = 0
    fT = generate_symmetric_filter(T, filterType="Gaussian", sigma=T_filter_sigma, width=10)

    # repeat the filters across the time dimension
    fROs = np.repeat(fRO[None,:], T, axis=0)
    fE1s = np.repeat(fE1[None,:], T, axis=0)

    pf_fROs = np.repeat(pf_fRO[None,:], T, axis=0)
    pf_fE1s = np.repeat(pf_fE1[None,:], T, axis=0)

    # compute final filter
    fROs_used = fROs * pf_fROs * fdROs
    fE1s_used = fE1s * pf_fE1s * fdE1s

    ratio_RO = 1/np.sqrt(1/RO * np.sum(fROs_used * fROs_used, axis=1))
    ratio_E1 = 1/np.sqrt(1/E1 * np.sum(fE1s_used * fE1s_used, axis=1))

    fROs_used *= ratio_RO[:, np.newaxis]
    fE1s_used *= ratio_E1[:, np.newaxis]

    # apply fft over time
    for i in range(T):
        nns[i] = ifft2c(apply_kspace_filter_2D(fft2c(nns[i]), fROs_used[i], fE1s_used[i]))

    if T_filter_sigma > 0:
        # apply extra T filter
        nns = ifft1c(apply_kspace_filter_1D(fft1c(nns), fT))

    if(verbose is True):
        print("--" * 20)
        print(f"kspace_filter_sigma is {ro_filter_sigma}, {e1_filter_sigma}, {T_filter_sigma}")

    if(verbose is True):
        print("--" * 20)
        std_r = np.mean(np.std(np.real(nns), axis=3))
        std_i = np.mean(np.std(np.imag(nns), axis=3))
        print("final noise, real, std is ", std_r, std_r - std_real)
        print("final noise, imag, std is ", std_i, std_i - std_imag)
        print(f"{Fore.GREEN}==============================================={Style.RESET_ALL}")
        
        assert abs(std_real-std_r) < 0.1
        assert abs(std_imag-std_i) < 0.1

    return nns, noise_sigma

# --------------------------------------------------------------

if __name__ == "__main__":
    
    import h5py 
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt 
    #matplotlib.use('TkAgg')

    # --------------------------------------------------------------------
    
    DATA_HOME = os.path.dirname(os.path.abspath(__file__)) + "/../"
    print("DATA_HOME is", DATA_HOME)


    # --------------------------------------------------------------------

    sigmas = np.linspace(1.0, 31.0, 30)
    for sigma in sigmas:
        t0 = start_timer(enable=True)
        nns, noise_sigma = generate_3D_MR_correlated_noise(T=8, RO=32, E1=32, REP=256, 
                                        min_noise_level=sigma, 
                                        max_noise_level=sigma, 
                                        kspace_filter_sigma=[0, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
                                        pf_filter_ratio=[1.0, 0.875, 0.75, 0.625, 0.55],
                                        kspace_filter_T_sigma=[0, 0.5, 0.65, 0.85, 1.0, 1.5, 2.0, 2.25],
                                        phase_resolution_ratio=[1.0, 0.85, 0.7, 0.65, 0.55],
                                        readout_resolution_ratio=[1.0, 0.85, 0.7, 0.65, 0.55],
                                        rng=np.random.default_rng(),
                                        verbose=True)
        end_timer(enable=True, t=t0, msg=f"generate_3D_MR_correlated_noise - sigma {sigma}, ")

    # --------------------------------------------------------------------

    # R1
    data_dir = os.path.join(DATA_HOME, 'data', 'snr_unit_data', 'meas_MID03480_FID51667_FLASH_PAT1_n=256')

    unwrappedIm_real = np.load(os.path.join(data_dir, 'unwrappedIm_real.npy'))
    unwrappedIm_imag = np.load(os.path.join(data_dir, 'unwrappedIm_imag.npy'))

    unwrappedIm = unwrappedIm_real + 1j * unwrappedIm_imag
    print("unwrappedIm is ", unwrappedIm.shape)
    
    unwrappedIm.astype(np.complex256)
    
    gmap = np.load(os.path.join(data_dir, 'gmap.npy'))
    print("gmap is ", gmap.shape)
    
    gmap.astype(np.float64)
    
    mask = np.load(os.path.join(data_dir, 'mask.npy'))
    print("mask is ", mask.shape)
    
    snr_im = unwrappedIm / gmap[:,:,np.newaxis]   
    std_map = np.std(np.abs(snr_im), axis=2)
    
    noise_level = np.mean(std_map[mask>0])
    print("noise level is ", noise_level)
    
    assert abs(noise_level-1) < 0.03
    
    plt.figure()
    plt.imshow(std_map,cmap='gray')    
    plt.show()

    RO, E1, PHS = unwrappedIm.shape

    kspace = fft2c(unwrappedIm)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(abs(unwrappedIm[:,:,0]),cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(abs(kspace[:,:,0]),cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(abs(ifft2c(kspace[:,:,0])),cmap='gray')
    plt.show()
            
    # --------------------------------------------------------------------
    # test reduce matrix size and reduce resolution
    t0 = start_timer(enable=True)
    im_low_matrix = apply_matrix_size_reduction_2D(unwrappedIm, int(0.8*RO), int(0.8*E1), norm='backward')
    end_timer(enable=True, t=t0, msg="apply_matrix_size_reduction_2D")
    print("im_low_matrix is ", im_low_matrix.shape)

    t0 = start_timer(enable=True)
    mask_low_matrix = cv2.resize(mask, dsize=im_low_matrix.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    end_timer(enable=True, t=t0, msg="cv2.resize")

    signal_level = np.abs(np.mean(unwrappedIm[mask>0.1]))
    print("test reduce matrix size and resolution, signal level is ", signal_level)
    signal_level_low_matrix = np.abs(np.mean(im_low_matrix[mask_low_matrix>0.1]))
    print("test reduce matrix size and resolution, signal level of im_low_matrix is ", signal_level_low_matrix)
    
    assert abs(signal_level - signal_level_low_matrix) < 3
    
    ratio_RO = 0.57
    ratio_E1 = 0.65
    im_low_matrix_low_res, fRO, fE1 = apply_resolution_reduction_2D(im_low_matrix, ratio_RO, ratio_E1, snr_scaling=False, norm='backward')
    
    signal_level_low_matrix_low_res = np.abs(np.mean(im_low_matrix_low_res[mask_low_matrix>0.1]))
    print("test reduce matrix size and resolution, signal level of im_low_matrix_low_res is ", signal_level_low_matrix_low_res)
    
    assert abs(signal_level_low_matrix - signal_level_low_matrix_low_res) < 0.1
    
    im_low_matrix_low_res_filtered, _, _ = apply_image_filter(im_low_matrix_low_res, sigma_RO=2.2, sigma_E1=1.53)
    signal_level_low_matrix_low_res_filtered = np.abs(np.mean(im_low_matrix_low_res_filtered[mask_low_matrix>0.1]))
    print("test reduce matrix size and resolution, signal level of im_low_matrix_low_res_filtered is ", signal_level_low_matrix_low_res_filtered)    
    assert abs(signal_level_low_matrix - signal_level_low_matrix_low_res_filtered) < 0.1
    
    im_low_matrix_low_res_filtered_T, _ = apply_image_filter_T(im_low_matrix_low_res_filtered, sigma_T=2.2)
    signal_level_low_matrix_low_res_filtered_T = np.abs(np.mean(im_low_matrix_low_res_filtered_T[mask_low_matrix>0.1]))
    print("test reduce matrix size and resolution, signal level of signal_level_low_matrix_low_res_filtered_T is ", signal_level_low_matrix_low_res_filtered_T)    
    assert abs(signal_level_low_matrix - signal_level_low_matrix_low_res_filtered_T) < 0.1
        
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(np.abs(unwrappedIm[:,:,12]),cmap='gray')
    plt.subplot(1, 4, 2)
    plt.imshow(np.abs(im_low_matrix[:,:,12]),cmap='gray')
    plt.subplot(1, 4, 3)
    plt.imshow(mask_low_matrix,cmap='gray')
    plt.subplot(1, 4, 4)
    plt.imshow(np.abs(im_low_matrix_low_res[:,:,12]),cmap='gray')
    plt.show()
    
    std_map = np.std(np.abs(im_low_matrix_low_res), axis=2)    
    noise_level = np.mean(std_map[mask_low_matrix>0.1])
    print("test reduce matrix size and resolution, noise level is ", noise_level)
            
    # --------------------------------------------------------------------
    # test partial fourier kspace filter for noise
    
    RO = 192
    E1 = 144
    
    pf_fRO = generate_asymmetric_filter(RO, 0, int(0.8*RO), filterType="TapperedHanning", width=10)
    pf_fE1 = generate_asymmetric_filter(E1, 0, E1-1, filterType="TapperedHanning", width=20)
    fRO = generate_symmetric_filter(RO, filterType="Gaussian", sigma=1.23, width=10)
    fE1 = generate_symmetric_filter(E1, filterType="Gaussian", sigma=1.45, width=10)
       
    noise_sigma = 3.7
    nn = create_complex_noise(noise_sigma, size=(RO, E1, 256))
    
    std_r = np.mean(np.std(np.real(nn), axis=2))
    print("test noise adding, real std is ", std_r)    
    assert abs(std_r - noise_sigma) < 0.1
    
    std_i = np.mean(np.std(np.imag(nn), axis=2))
    print("test noise adding, imag std is ", std_i)
    assert abs(std_i - noise_sigma) < 0.1
            
    ratio_RO = 0.85
    ratio_E1 = 0.65
    
    nn, fdRO, fdE1 = apply_resolution_reduction_2D(nn, ratio_RO, ratio_E1, snr_scaling=False)
    
    std_r = np.mean(np.std(np.real(nn), axis=2))
    print("test noise pf filter, real std is ", std_r)
    
    std_i = np.mean(np.std(np.imag(nn), axis=2))
    print("test noise pf filter, imag std is ", std_i)
    
    fRO_used = fRO * pf_fRO * fdRO
    fE1_used = fE1 * pf_fE1 * fdE1
    
    ratio_RO = 1/math.sqrt(1/RO * np.sum(fRO_used*fRO_used))
    ratio_E1 = 1/math.sqrt(1/E1 * np.sum(fE1_used*fE1_used))
    
    fRO_used *= ratio_RO
    fE1_used *= ratio_E1
    
    plt.figure()
    plt.plot(fRO_used, 'r')
    plt.plot(fE1_used, 'b')
    plt.show()
        
    kspace_filtered = apply_kspace_filter_2D(fft2c(nn), fRO_used, fE1_used)    
    nn_filtered = ifft2c(kspace_filtered)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(abs(nn[:,:,12]),cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(abs(nn_filtered[:,:,12]),cmap='gray')
    plt.show()
        
    std_r_filtered = np.mean(np.std(np.real(nn_filtered), axis=2))
    print("test noise pf filter, real std is ", std_r_filtered)
    assert abs(std_r_filtered - noise_sigma) < 0.1
    
    std_i_filtered = np.mean(np.std(np.imag(nn_filtered), axis=2))
    print("test noise pf filter, imag std is ", std_i)
    assert abs(std_i_filtered - noise_sigma) < 0.1
       
    # --------------------------------------------------------------------
    # test kspace filter    
    
    fRO = generate_symmetric_filter(RO, filterType="Gaussian", sigma=1.5, width=10)
    fE1 = generate_symmetric_filter(E1, filterType="None", sigma=1.5, width=10)
    
    plt.figure()
    plt.plot(fRO, 'r')
    plt.plot(fE1, 'b')
    plt.show()
    
    kspace_filtered = apply_kspace_filter_2D(kspace, fRO, fE1)    
    im_filtered = ifft2c(kspace_filtered)    
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(abs(unwrappedIm[:,:,12]),cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(abs(im_filtered[:,:,12]),cmap='gray')
    plt.show()
    
    snr_im = im_filtered / gmap[:,:,np.newaxis]   
    std_map = np.std(np.abs(snr_im), axis=2)    
    noise_level = np.mean(std_map[mask>0])
    print("test kspace filter, noise level is ", noise_level)
    assert abs(noise_level - 1.0) < 0.1
    
    plt.figure()
    plt.imshow(std_map,cmap='gray')    
    plt.show()
    
    # ----------------------------
    
    fRO = generate_symmetric_filter(RO, filterType="Gaussian", sigma=1.23, width=10)
    fE1 = generate_symmetric_filter(E1, filterType="Gaussian", sigma=1.45, width=10)
    
    plt.figure()
    plt.plot(fRO, 'r')
    plt.plot(fE1, 'b')
    plt.show()
    
    kspace_filtered = apply_kspace_filter_2D(kspace, fRO, fE1)    
    im_filtered = ifft2c(kspace_filtered)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(abs(unwrappedIm[:,:,12]),cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(abs(im_filtered[:,:,12]),cmap='gray')
    plt.show()
    
    snr_im = im_filtered / gmap[:,:,np.newaxis]   
    std_map = np.std(np.abs(snr_im), axis=2)    
    noise_level = np.mean(std_map[mask>0])
    print("test kspace filter, noise level is ", noise_level)
    assert abs(noise_level - 1.0) < 0.1
    
    plt.figure()
    plt.imshow(std_map,cmap='gray')    
    plt.show()
    
    # --------------------------------------------------------------------
    # test partial fourier kspace filter
    
    fRO = generate_asymmetric_filter(RO, 0, int(0.8*RO), filterType="TapperedHanning", width=10)
    fE1 = generate_asymmetric_filter(E1, int(0.2*E1), E1, filterType="None", width=20)
    
    plt.figure()
    plt.plot(fRO, 'r')
    plt.plot(fE1, 'b')
    plt.show()
    
    kspace_filtered = apply_kspace_filter_2D(kspace, fRO, fE1)    
    im_filtered = ifft2c(kspace_filtered)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(abs(unwrappedIm[:,:,12]),cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(abs(im_filtered[:,:,12]),cmap='gray')
    plt.show()
    
    snr_im = im_filtered / gmap[:,:,np.newaxis]   
    std_map = np.std(np.abs(snr_im), axis=2)    
    noise_level = np.mean(std_map[mask>0])
    print("test partial fourier kspace filter, noise level is ", noise_level)
    assert abs(noise_level - 1.0) < 0.1
    
    plt.figure()
    plt.imshow(std_map,cmap='gray')    
    plt.show()
    
    # --------------------------------------------------------------------
    # test reduce resolution
    
    ratio_RO = 0.87
    ratio_E1 = 0.75
    im_low_res, fRO, fE1 = apply_resolution_reduction_2D(unwrappedIm, ratio_RO, ratio_E1)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(abs(unwrappedIm[:,:,12]),cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(abs(im_low_res[:,:,12]),cmap='gray')
    plt.show()
    
    std_map = np.std(np.abs(im_low_res), axis=2)    
    noise_level = np.mean(std_map[mask>0.1])
    print("test reduce resolution, noise level is ", noise_level)
    assert abs(noise_level - 1.0) < 0.1
    
    # --------------------------------------------------------------------
    # test change matrix size
    
    im_low_matrix = apply_matrix_size_reduction_2D(unwrappedIm, int(0.8*RO), int(0.8*E1))
    print("im_low_matrix is ", im_low_matrix.shape)
    
    mask_low_matrix = cv2.resize(mask, dsize=im_low_matrix.shape[1::-1], interpolation=cv2.INTER_NEAREST)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(abs(unwrappedIm[:,:,12]),cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(abs(im_low_matrix[:,:,12]),cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(mask_low_matrix,cmap='gray')
    plt.show()
    
    std_map = np.std(np.abs(im_low_matrix), axis=2)    
    noise_level = np.mean(std_map[mask_low_matrix>0.1])
    print("test reduce matrix size, noise level is ", noise_level)
    assert abs(noise_level - 1.0) < 0.1
    
    # --------------------------------------------------------------------
    # test reduce matrix size and reduce resolution

    im_low_matrix = apply_matrix_size_reduction_2D(unwrappedIm, int(0.8*RO), int(0.8*E1), norm='backward')
    print("im_low_matrix is ", im_low_matrix.shape)
    
    mask_low_matrix = cv2.resize(mask, dsize=im_low_matrix.shape[1::-1], interpolation=cv2.INTER_NEAREST)

    ratio_RO = 0.57
    ratio_E1 = 0.65
    im_low_matrix_low_res, fRO, fE1 = apply_resolution_reduction_2D(im_low_matrix, ratio_RO, ratio_E1, snr_scaling=False)

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(np.abs(unwrappedIm[:,:,12]),cmap='gray')
    plt.subplot(1, 4, 2)
    plt.imshow(np.abs(im_low_matrix[:,:,12]),cmap='gray')
    plt.subplot(1, 4, 3)
    plt.imshow(mask_low_matrix,cmap='gray')
    plt.subplot(1, 4, 4)
    plt.imshow(np.abs(im_low_matrix_low_res[:,:,12]),cmap='gray')
    plt.show()

    std_map = np.std(np.abs(im_low_matrix_low_res), axis=2)    
    noise_level = np.mean(std_map[mask_low_matrix>0.1])
    print("test reduce matrix size and resolution, noise level is ", noise_level)

    # --------------------------------------------------------------------
    # test zero-padding resizing

    mask_resized = zero_padding_resize_2D(mask, 2*RO, 2*E1)
    im_resized = zero_padding_resize_2D(unwrappedIm, 2*RO, 2*E1)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(abs(unwrappedIm[:,:,12]),cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(abs(im_resized[:,:,12]),cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(abs(mask_resized),cmap='gray')
    plt.show()

    std_map = np.std(np.abs(im_resized), axis=2)    
    noise_level = np.mean(std_map[abs(mask_resized)>0.1])
    print("test zero-padding resizing, noise level is ", noise_level)
    assert abs(noise_level - 1.0) < 0.1

    # --------------------------------------------------------------------
    # test noise adding

    noise_sigma = 2.3
    nn = create_complex_noise(noise_sigma, size=unwrappedIm.shape)

    std_r = np.mean(np.std(np.real(nn), axis=2))
    print("test noise adding, real std is ", std_r)

    std_i = np.mean(np.std(np.imag(nn), axis=2))
    print("test noise adding, imag std is ", std_i)

    fRO = generate_symmetric_filter(RO, filterType="Gaussian", sigma=1.23, width=10)
    fE1 = generate_symmetric_filter(E1, filterType="Gaussian", sigma=1.45, width=10)

    kspace_filtered = apply_kspace_filter_2D(fft2c(nn), fRO, fE1)    
    nn_filtered = ifft2c(kspace_filtered)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(abs(nn[:,:,12]),cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(abs(nn_filtered[:,:,12]),cmap='gray')
    plt.show()

    std_r = np.mean(np.std(np.real(nn_filtered), axis=2))
    print("test kspace filter for noise, real, std is ", std_r)
    assert abs(std_r - noise_sigma) < 0.1

    std_i = np.mean(np.std(np.imag(nn_filtered), axis=2))
    print("test kspace filter for noise, imag, std is ", std_i)    
    assert abs(std_i - noise_sigma) < 0.1

    # --------------------------------------------------------------------
