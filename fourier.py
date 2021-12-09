import numpy as np
import numba

def slow_fourier(arr: np.ndarray) -> np.ndarray:
    w = len(arr)
    h = len(arr[0])
    N = w*h

    c = -1j*2*np.pi/N
    n = np.arange(N)
    c = c * n
    x = arr.flatten()
    fourier = np.empty(N, dtype=np.complex_)
    for k in range(N):
        fourier[k] = np.sum(x * np.exp(c*k))
        
    return fourier.reshape(w,h)

def slow_inverse_fourier(freq: np.ndarray) -> np.ndarray:
    w = len(freq)
    h = len(freq[0])
    N = w*h

    c = 1j*2*np.pi/N
    n = np.arange(N)
    c = c * n
    x = freq.flatten()
    
    fourier = np.empty(N, dtype=np.complex_)
    for k in range(N):
        fourier[k] = np.sum(x * np.exp(c*k)) / N
    
    return fourier.reshape(w,h)

def frequency_shift(freq: np.ndarray) -> np.ndarray:
    pass