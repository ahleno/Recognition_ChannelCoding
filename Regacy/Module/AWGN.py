"""
@author: Hyunwoo Cho
@date: 2024/02/15
"""
import numpy as np

def AWGN(dB, code_rate, data):
    dB = 10**(dB/10)
    sigma = np.sqrt(1/(2*code_rate*dB))
    noise_i = sigma * np.random.randn(len(data))
    noise_q = sigma * np.random.randn(len(data))

    noise_data = data + noise_i + 1j * noise_q

    return noise_data