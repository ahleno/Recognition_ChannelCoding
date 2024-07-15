import numpy as np
from conv import Convolutional_code
from mod import Modulation
from AWGN import AWGN
import os
import os.path
from tqdm import tqdm

from pathlib import Path

# (23, 12) Golay code
def golay(data_size):
    m_len = 12

    P = np.array([
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    ], dtype='int')

    G = np.concatenate((np.eye(m_len, dtype="int"), P), axis=1)

    msg = np.random.randint(0,2, (data_size, m_len))
    codeword = np.dot(msg, G) %2

    return codeword

# Hamming Code (8, 4)
def Hamming(data_size):
    G = np.array([
        [1, 1, 1, 0, 0 ,0, 0, 1],
        [1, 0 ,0, 1, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, 1, 0]
    ], dtype='int')

    msg = np.random.randint(0,2, (data_size, 4))
    codeword = np.dot(msg, G) %2

    return codeword

# BCH Code (15, 7)
def BCH(data_size):
    c_length = 15
    m_length = 7
    G_X_15_7 = [1,1,1,0,1,0,0,0,1]

    BCH_15_7_G = np.zeros((7,15))

    for i in range(7):
        BCH_15_7_G[i,i:i+9] = G_X_15_7

    for i in range(7):
        for j in range(i+1,7):
            if BCH_15_7_G[i, j] == 1:
                BCH_15_7_G[i] = (BCH_15_7_G[i] + BCH_15_7_G[j])%2

    G = BCH_15_7_G # G =BCH_7_4_G
    msg = np.random.randint(0,2,(data_size,m_length))
    codeword = np.dot(msg, G)%2

    return codeword

# Product Code (8, 4)
def product(data_size):
    G = np.array([
        [1, 0, 1, 0, 0 ,0, 1, 0],
        [0, 1 ,1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 1]
    ], dtype='int')

    msg = np.random.randint(0,2, (data_size, 4))
    codeword = np.dot(msg, G) %2

    return codeword

# RM Code (16, 11)
def G_matrix(length, m, r):
    G = np.ones(length)
    for i in range(m):
        v = np.zeros((int(length/(2**(i+1)))))
        v = np.hstack((v, np.ones((int(length/(2**(i+1)))))))
        while v.shape[0] < length :
            v = np.hstack((v, np.zeros((int(length/(2**(i+1)))))))
            v = np.hstack((v, np.ones((int(length/(2**(i+1)))))))
        G = np.vstack((G,v))
    if r == 1:
        return G
    elif r > 1 :
        for i in range(1,m):
            for j in range(i+1,m+1):
                G = np.vstack((G,(G[i]*G[j])))
        if r == 3:
            G = np.vstack((G,(G[1]*G[2]*G[3])))
            G = np.vstack((G,(G[1]*G[3]*G[4])))
            G = np.vstack((G,(G[1]*G[2]*G[4])))
            G = np.vstack((G,(G[2]*G[3]*G[4])))
        return G
    return G

def rm(data_size):
    m = 4
    r = 2
    length = 2**m

    if r == 1:
        masking_length=0
        msg_length = m+r
    elif r == 2:
        masking_length=6
        msg_length = 11
    elif r == 3:
        masking_length=10
        msg_length = 15

    G = G_matrix(length, m, r)
    msg = np.random.randint(0,2,(data_size,msg_length))
    codeword = np.dot(msg, G) %2

    return codeword

# (n, k) Polar code
def make_F(power):
    F = np.array([[1,0],[1,1]])
    for i in range(1,power):
        first = np.concatenate((F,np.zeros((2**i, 2**i))), axis=1)
        second = np.concatenate((F,F), axis=1)
        F = np.concatenate((first, second), axis =0)

    return F
def Compute_z(z, k, i = 1):
    for j in range(i):
        z[(2*i,2*j)] = 2*z[(i,j)] - (z[(i,j)])**2
        z[(2*i,2*j+1)] = (z[(i,j)])**2
    if 2*i < 2**k:
        z = Compute_z(z, k, 2*i)

    return z
def Frozen_bits(n, z, slice_index):
    bit_index = np.zeros(n)
    for i in range(n):
        bit_index[i] = z[(n, i)]

    bit_index = np.argsort(bit_index)[::-1]

    frozen_bit_index = bit_index[:slice_index]
    message_bit_index = bit_index[slice_index:]
    
    return np.sort(frozen_bit_index), np.sort(message_bit_index)

def polar(data_size):
    # (n, k) Polar code
    k = 4
    n = 2*k
    power = int(np.log2(n))

    F = make_F(power)

    z = {}
    z[(1,0)] = 0.5
    z = Compute_z(z, power)

    frozen_bit_index, message_bit_index = Frozen_bits(n, z, int(n-k))

    msg = np.random.randint(0,2,(data_size, k))
    u = np.zeros((data_size, n))
    u[:,message_bit_index] = msg

    codeword = np.dot(u,F)%2

    return codeword


# C(2,1,5) [27,31] (10111, 11001)
def conv(datas, m = None):
    
    m_num = 4
    g_func = np.array([[False,True,True,True],[True,False, False,True]])
    code_rate = 1/2
    
    if m is None:
        m = np.zeros(m_num)
        datas.extend([0 for i in range(m_num)])

    result = []
    tr_m = np.array(m)

    for data in datas:
        for j in range(tr_m[g_func[0]].shape[0]):
            if j == 0:
                result_1 = (tr_m[g_func[0]][j]+data) % 2
            else:
                result_1 = (tr_m[g_func[0]][j]+result_1) % 2

        for j in range(tr_m[g_func[1]].shape[0]):
            if j == 0:
                result_2 = (tr_m[g_func[1]][j]+data) % 2
            else:
                result_2 = (tr_m[g_func[1]][j]+result_2) % 2

        result.extend([result_1, result_2])

        tr_m = np.roll(tr_m, 1)
        tr_m[0] = data
    
    return np.array(result)