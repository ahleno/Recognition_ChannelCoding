"""
@author: Hyunwoo Cho
@date: 2024/02/15
"""
import numpy as np

class Modulation():
    def __init__(self, case):
        if case == 1:
            self.M = 1
            self.padding_len = 0

        elif case == 2:
            self.M = 2
            self.padding_len = 0
        
        elif case == 3:
            self.M = 3
            self.padding_len = 0

    def mod(self, data):
        if len(data)%self.M != 0:
            data = self.padding(data)

        self.data_len = int(len(data)/self.M)

        if self.M == 1:
            return self.BPSK_modulation(data)
        
        elif self.M == 2:
            return self.QPSK_modulation(data)
        
        elif self.M == 3:
            return self.PSK8_modulation(data)
        
    def demod(self, data, M):
        self.M = M
        if self.M == 1:
            return self.BPSK_demodulation(data)
        
        elif self.M == 2:
            return self.QPSK_demodulation(data)
        
        elif self.M == 3:
            return self.PSK8_demodulation(data)


    def padding(self, data):
        self.padding_len = int(self.M-len(data)%self.M)
        paddinng_data = np.concatenate((data, np.zeros(self.padding_len)))
        return paddinng_data
    
    def toInt(self, data):
        power_list = [2**i for i in range(self.M)][::-1]
        int_data = np.zeros((self.data_len), dtype=int)
        for i in range(self.data_len):
            int_data[i] += int(np.dot(power_list,data[self.M*i:self.M*i+self.M]))

        return int_data
    
    def BPSK_modulation(self, data):
        bpsk_data = np.zeros((self.data_len), dtype=complex)
        for i in range(self.data_len):
            bpsk_data[i] = (-1)**data[i]

        return bpsk_data


    def BPSK_demodulation(self, data):
        demod_data = np.zeros((self.data_len))
        for i in range(self.data_len):
            if data[i] > 0:
                demod_data[i] = 0
            else:
                demod_data[i] = 1

        return demod_data
