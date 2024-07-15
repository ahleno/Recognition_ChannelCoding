"""
@author: Hyunwoo Cho
@date: 2024/03/13
"""
import numpy as np

class Convolutional_code():
    def __init__(self, case):
        if case == 1:  # C(2,1,3) [5,7] (101, 111)
            self.m_num = 2
            self.g_func = np.array([[False,True],[True,True]])
            self.code_rate = 1/2

        elif case == 2:  # C(2,1,4) [13,17] (1011, 1111)
            self.m_num = 3
            self.g_func = np.array([[False,True,True],[True,True,True]])
            self.code_rate = 1/2

        elif case == 3:  # C(2,1,5) [27,31] (10111, 11001)
            self.m_num = 4
            self.g_func = np.array([[False,True,True,True],[True,False, False,True]])
            self.code_rate = 1/2

    def encoder(self,datas, m = None):
        if m is None:
            m = np.zeros(self.m_num)
            datas.extend([0 for i in range(self.m_num)])

        result = []
        tr_m = np.array(m)

        for data in datas:
            for j in range(tr_m[self.g_func[0]].shape[0]):
                if j == 0:
                    result_1 = (tr_m[self.g_func[0]][j]+data) % 2
                else:
                    result_1 = (tr_m[self.g_func[0]][j]+result_1) % 2

            for j in range(tr_m[self.g_func[1]].shape[0]):
                if j == 0:
                    result_2 = (tr_m[self.g_func[1]][j]+data) % 2
                else:
                    result_2 = (tr_m[self.g_func[1]][j]+result_2) % 2

            result.extend([result_1, result_2])

            tr_m = np.roll(tr_m, 1)
            tr_m[0] = data
        
        return np.array(result), tr_m
    
    def decoder(self, demod_data, case, mode = 'hard'):
        if case == 1:  # C(2,1,3) [5,7] (101, 111)
            self.m_num = 2
            self.g_func = np.array([[False,True],[True,True]])

        elif case == 2:  # C(2,1,4) [13,17] (1011, 1111)
            self.m_num = 3
            self.g_func = np.array([[False,True,True],[True,True,True]])

        elif case == 3:  # C(2,1,5) [27,31] (10111, 11001)
            self.m_num = 4
            self.g_func = np.array([[False,True,True,True],[True,False, False,True]])
            
        self.state_diagram = self.create_state_diagram()
        self.traceback_depth = (self.m_num+1)*10

        path = np.zeros([2**self.m_num, 2 + self.traceback_depth])  # path[0]: state, path[1]: cost, path[2:]: imformation bit
        traceback_m = []

        self.first_state_func(0, path, 0, 2**self.m_num, demod_data, 0, mode)

        for i in range(self.m_num, (demod_data.shape[0]//2)-self.m_num):
            self.state_func(path, demod_data, i, traceback_m, mode)

        for i in range((demod_data.shape[0]//2)-self.m_num, demod_data.shape[0]//2):
            path = self.final_state_func(path, demod_data, i, traceback_m, mode)

        traceback_m.extend(path[0,2:])
        
        return traceback_m

    def create_state_diagram(self):
        state_diagram = np.zeros((2**self.m_num, 2, 3))
        g_m = np.zeros(self.m_num)
        
        for i in range(state_diagram.shape[0]):
            num = i
            for k in range(g_m.shape[0]):
                g_m[k] = num % 2
                num = num // 2

            for j in range(state_diagram.shape[1]):
                data = np.array([j])
                result, tr_g_m = self.encoder(data, g_m)
                state = 0
                for k in range(tr_g_m.shape[0]):
                    state += int((2**k) * tr_g_m[k])
                state_diagram[i,j] = np.hstack((result, state))
                
        return state_diagram
    
    def first_state_func(self, state, path, start, end, data, step, mode): 
        for j in range((end-start)//2):
            if mode == 'hard':
                path[start + j, 1] += ((data[2*step:2*step+2] + self.state_diagram[state, 0,:2]) % 2).sum()
            elif mode == 'soft':
                path[start +j, 1] += np.dot(data[2*step:2*step+2],np.where(self.state_diagram[state, 0,:2] > 0, -1, 1))
            zero_state = int(self.state_diagram[state, 0, 2])
            path[start + j, 0] = zero_state
            
            if mode == 'hard':
                path[start + (end-start)//2 + j, 1] += ((data[2*step:2*step+2] + self.state_diagram[state, 1,:2]) % 2).sum()
            elif mode == 'soft':
                path[start + (end-start)//2 + j, 1] += np.dot(data[2*step:2*step+2],np.where(self.state_diagram[state, 1,:2] > 0, -1, 1))
            one_state = int(self.state_diagram[state, 1, 2])
            path[start + (end-start)//2 + j, 0] = one_state
                    
        if step < self.m_num-1:   
            self.first_state_func(zero_state, path, start, start + (end-start)//2, data, step+1, mode)
            self.first_state_func(one_state, path, start + (end-start)//2, end, data, step+1, mode)
        else:
            return
        
    def state_func(self, path, data, step, traceback_m, mode):
        new_path = np.zeros([path.shape[0]*2, 2 + self.traceback_depth])
        path_len = step - self.m_num
        
        if path_len >= self.traceback_depth:
            if mode == 'hard':
                min_index = np.argmin(path[:,1])
                traceback_m.extend([path[min_index,2]])
            elif mode == 'soft':
                max_index = np.argmax(path[:,1])
                traceback_m.extend([path[max_index,2]])
            path[:, 2:] = np.roll(path[:,2:], -1, axis = 1)
            path_len = self.traceback_depth-1
            
        for i in range(path.shape[0]):
            if mode == 'hard':
                new_path[2*i, 1] = path[i,1] + ((data[2*step:2*step+2] + self.state_diagram[int(path[i,0]), 0, :2]) % 2).sum()
            elif mode == 'soft':
                new_path[2*i, 1] = path[i,1] + np.dot(data[2*step:2*step+2], np.where(self.state_diagram[int(path[i,0]), 0,:2] > 0, -1, 1))
            new_path[2*i, 2:] = path[i, 2:]
            new_path[2*i, 2+path_len] = format(int(path[i,0]), 'b').zfill(self.m_num)[0]
            new_path[2*i, 0] = int(self.state_diagram[int(path[i,0]), 0, 2])
            
            if mode == 'hard':
                new_path[2*i+1, 1] = path[i,1] + ((data[2*step:2*step+2] + self.state_diagram[int(path[i,0]), 1, :2]) % 2).sum()
            elif mode == 'soft':
                new_path[2*i+1, 1] = path[i,1] + np.dot(data[2*step:2*step+2], np.where(self.state_diagram[int(path[i,0]), 1,:2] > 0, -1, 1))
            new_path[2*i+1, 2:] = path[i, 2:]
            new_path[2*i+1, 2+path_len] = format(int(path[i,0]), 'b').zfill(self.m_num)[0]
            new_path[2*i+1, 0] = int(self.state_diagram[int(path[i,0]), 1, 2])

        for i in range(2**self.m_num):
            index = np.where(new_path[:, 0] == i)[0]
            if index.shape[0] > 1:
                if mode == 'hard':
                    min_index = np.argmin(new_path[index, 1])
                    path[i] = new_path[index[min_index]]
                elif mode == 'soft':
                    max_index = np.argmax(new_path[index, 1])
                    path[i] = new_path[index[max_index]]

    def final_state_func(self, path, data, step, traceback_m, mode):
        path_len = step - self.m_num
        
        if path_len >= self.traceback_depth:
            if mode == 'hard':
                min_index = np.argmin(path[:,1])
                traceback_m.extend([path[min_index,2]])
            elif mode == 'soft':
                max_index = np.argmax(path[:,1])
                traceback_m.extend([path[max_index,2]])
            path[:, 2:] = np.roll(path[:,2:], -1, axis = 1)
            path_len = self.traceback_depth-1
        
        for i in range(path.shape[0]):
            if mode == 'hard':
                path[i,1] += ((data[2*step:2*step+2] + self.state_diagram[int(path[i,0]), 0, :2]) % 2).sum()
            elif mode == 'soft':
                path[i,1] += np.dot(data[2*step:2*step+2], np.where(self.state_diagram[int(path[i,0]), 0,:2] > 0, -1, 1))
            path[i, 2+path_len] = format(int(path[i,0]), 'b').zfill(self.m_num)[0]
            path[i, 0] = int(self.state_diagram[int(path[i,0]), 0, 2])
            
        del_index=[]
        
        for i in range(2**self.m_num):
            index = np.where(path[:, 0] == i)[0]
            if index.shape[0] > 1:
                if mode == 'hard':
                    min_index= np.argmin(path[index, 1])
                    index = np.delete(index, min_index)
                elif mode == 'soft':
                    max_index= np.argmax(path[index, 1])
                    index = np.delete(index, max_index)
                del_index.extend(index)
                
        if del_index != []:
            return np.delete(path, del_index, 0)
        
        else:
            return path
                    
def error_check(bitstring, esti_m):
    error = [(m+m_hat)%2 for m, m_hat in zip(bitstring,esti_m)]

    return sum(error)/len(bitstring)