import numpy as np
from numpy.fft import fft, fftfreq

class TimeSeriesAnalysis:
    
    def __init__(self, data: np.ndarray):
        self.data             = data
        self.mean             = None
        self.std              = None
        self.spectral_density = None
        self.frequencies      = None
        self.memory           = None
        self.red_noise        = None
    
    def compute_statistics(self):
        self.mean = np.mean(self.data)
        self.std  = np.std(self.data)
        
    def compute_spectral_density(self):
        n     = len(self.data)
        freqs = fftfreq(n)
        sp    = fft(self.data)
        spectral_density = 2 * sp * sp.conj() / n**2
        
        self.frequencies      = freqs[:n//2]
        self.spectral_density = spectral_density[:n//2].real
    
    def generate_red_noise(self):
        self.compute_statistics()
        data        = (self.data - self.mean) / self.std
        self.memory = np.dot(data[1:], data[:-1]) / np.dot(data[:-1], data[:-1])
        _sigma      = np.sqrt(1 - self.memory**2); f = np.random.normal(0, _sigma, size=10000)
        
        self.red_noise = np.zeros_like(data)
        for i in range(1, len(data)):
            self.red_noise[i] = self.memory * self.red_noise[i-1] + np.random.choice(f)