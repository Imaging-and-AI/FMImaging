import time
import numpy
import scipy
import pyfftw
import multiprocessing
import os
import numpy as np
import h5py
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *

F = h5py.File('/export/Lab-Xue/projects/mri/data/VIDA_test_0430.h5', 'r')

res_dir = '/export/Lab-Xue/projects/mri/non_cardiac/'
os.makedirs(res_dir, exist_ok=True)

for key in F:
    
    im = F[key + '/image']
    gmap = F[key + '/gmap']
    
    print(im.shape, gmap.shape)
    
    im = np.transpose(im, [1, 2, 0])
    gmap = np.transpose(gmap, [1, 2, 0])
    
    case_dir = os.path.join(res_dir, key)
    os.makedirs(case_dir, exist_ok=True)
    
    save_inference_results(im, None, gmap, case_dir)

nthread = multiprocessing.cpu_count()
a = numpy.random.rand(2364,2756).astype('complex128')
b = a.astype(numpy.complex64)
""" 
Uncomment below to use 32 bit floats, 
increasing the speed by a factor of 4
and remove the difference between the "builders" and "FFTW" methods
"""
#a = numpy.random.rand(2364,2756).astype('complex64')

for i in range(20):
    start = time.time()
    b1 = numpy.fft.fft2(a)
    end1 = time.time() - start
    print('numpy.fft.fft2:                        %.3f secs.' % end1)

    start = time.time()
    b1 = scipy.fft.fft2(a)
    end1 = time.time() - start
    print('scipy.fft.fft2:                        %.3f secs.' % end1)

    start = time.time()
    b1 = numpy.fft.fft2(b)
    end1 = time.time() - start
    print('complex64, numpy.fft.fft2:                        %.3f secs.' % end1)

    start = time.time()
    b1 = scipy.fft.fft2(b)
    end1 = time.time() - start
    print('complex64, scipy.fft.fft2:                        %.3f secs.' % end1)


start = time.time()
b2 = pyfftw.interfaces.scipy_fftpack.fft2(a, threads=nthread)
end2 = time.time() - start

pyfftw.forget_wisdom()
start = time.time()
b3 = pyfftw.interfaces.numpy_fft.fft2(a, threads=nthread)
end3 = time.time() - start

""" By far the most efficient method """
pyfftw.forget_wisdom()
start = time.time()
b4 = numpy.zeros_like(a)
fft = pyfftw.FFTW( a, b4, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), threads=nthread, planning_timelimit=None )
fft()
end4 = time.time() - start

""" 
For large arrays avoiding the copy is very important, 
doing this I get a speedup of 2x compared to not using it 
"""
pyfftw.forget_wisdom()
start = time.time()
b5 = numpy.zeros_like(a)
fft = pyfftw.builders.fft2(a, s=None, axes=(-2, -1), overwrite_input=False, planner_effort='FFTW_MEASURE', threads=nthread, auto_align_input=False, auto_contiguous=False, avoid_copy=True)
b5 = fft()
end5 = time.time() - start



print('numpy.fft.fft2:                        %.3f secs.' % end1)
print('pyfftw.interfaces.scipy_fftpack.fft2:  %.3f secs.' % end2)
print('pyfftw.interfaces.numpy_fft.fft2:      %.3f secs.' % end3)
print('pyfftw.FFTW:                           %.3f secs.' % end4)
print('pyfftw.builders:                       %.3f secs.' % end5)