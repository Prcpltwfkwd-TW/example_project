import pytest
import numpy as np
from example_package_bm99.time_series_analysis import TimeSeriesAnalysis
from .fixture import rmm_data

def test_compute_statistics():
    data = np.array([1, 2, 3, 4, 5])
    tsa  = TimeSeriesAnalysis(data)
    tsa.compute_statistics()
    
    assert tsa.mean == pytest.approx(3.0)
    assert tsa.std  == pytest.approx(np.sqrt(2.0))


def test_compute_spectral_density():
    data = 2 * np.sin(np.linspace(0, 2 * np.pi, 1000))
    tsa  = TimeSeriesAnalysis(data)
    tsa.compute_spectral_density()
    
    assert tsa.frequencies.size == 500
    assert tsa.spectral_density.size == 500
    
    peak_freq_index = np.argmax(tsa.spectral_density)
    assert tsa.spectral_density[peak_freq_index] == pytest.approx(2**2/2, rel=1e-2)


def test_generate_red_noise(rmm_data):
    rmm1 = rmm_data[0]
    tsa  = TimeSeriesAnalysis(rmm1)
    tsa.generate_red_noise()
    
    red_noise_memory = np.dot(tsa.red_noise[1:], tsa.red_noise[:-1]) / np.dot(tsa.red_noise[:-1], tsa.red_noise[:-1])
    assert red_noise_memory == pytest.approx(tsa.memory, rel=1e-2)