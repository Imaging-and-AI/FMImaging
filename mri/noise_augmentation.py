"""
Noise augmentation utilities

provides a wide range of utility function used to create training data for MRI

copied from Hui's original commit in CNNT
"""
import math
import numpy  as np
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, ifftshift, fftshift

# --------------------------------------------------------------

def centered_pad(data, new_shape):
    padding = (np.array(new_shape) - np.array(data.shape))//2

    data_padded = np.pad(data,[(padding[0],padding[0]),(padding[1],padding[1])])

    scaling = np.sqrt(np.prod(new_shape))/np.sqrt(np.prod(data.shape))
    data_padded *= scaling

    return data_padded

# --------------------------------------------------------------

def create_complex_noise(noise_sigma, size):
    return np.random.normal(0,noise_sigma,size=size)+np.random.normal(0,noise_sigma,size=size)*1j

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

def generate_symmetric_filter(len, filterType, sigma=1.5, width=10):
    """Compute the SNR unit symmetric filter

    Args:
        len (int): length of filter
        filterType (str): Gaussian or None
        sigma (float, optional): sigma for gaussian filter. Defaults to 1.5.

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

    sos = np.sum(filter*filter)
    filter /= math.sqrt(sos / len)

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
                                    rng=np.random.Generator(np.random.PCG64(8754132)),
                                    verbose=False):

    # create noise
    noise_sigma = (max_noise_level - min_noise_level) * rng.random() + min_noise_level
    if(REP>1):
        nns = create_complex_noise(noise_sigma, (T, RO, E1, REP))
    else:
        nns = create_complex_noise(noise_sigma, (T, RO, E1))

    if(verbose is True):
        print("--" * 40)
        print(f"noise sigma is {noise_sigma}")
        print("noise, real, std is ", np.mean(np.std(np.real(nns), axis=2)))
        print("noise, imag, std is ", np.mean(np.std(np.imag(nns), axis=2)))

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
        std_r = np.mean(np.std(np.real(nns), axis=0))
        std_i = np.mean(np.std(np.imag(nns), axis=0))
        print("noise, real, std is ", std_r)
        print("noise, imag, std is ", std_i)

        assert abs(noise_sigma-std_r) < 0.4
        assert abs(noise_sigma-std_i) < 0.4

    return nns, noise_sigma

# --------------------------------------------------------------

if __name__ == "__main__":
    
    sigmas = np.linspace(1.0, 30.0, 60)
    for sigma in sigmas:
        nns, noise_sigma = generate_3D_MR_correlated_noise(T=30, RO=192, E1=144, REP=1, 
                                        min_noise_level=sigma, 
                                        max_noise_level=sigma, 
                                        kspace_filter_sigma=[0, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
                                        pf_filter_ratio=[1.0, 0.875, 0.75, 0.625, 0.55],
                                        kspace_filter_T_sigma=[0, 0.5, 0.65, 0.85, 1.0, 1.5, 2.0, 2.25],
                                        phase_resolution_ratio=[1.0, 0.85, 0.7, 0.65, 0.55],
                                        readout_resolution_ratio=[1.0, 0.85, 0.7, 0.65, 0.55],
                                        rng=np.random.Generator(np.random.PCG64(8754132)),
                                        verbose=True)
    