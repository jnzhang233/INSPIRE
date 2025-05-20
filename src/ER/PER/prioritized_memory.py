import numpy as np
np.random.seed(1)
import torch

class PER_Memory(object):  
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    # epsilon = 0.01  # small amount to avoid zero priority
    # alpha = 0.6  # [0~1] convert the importance of TD error to priority
    # beta = 0.4  # importance-sampling, from initial value increasing to 1
    # beta_increment_per_sampling = 0.001
    # abs_err_upper = 1.  # clipped abs error

    def __init__(self, args, td_error, mask):
        self.alpha = args.selected_alpha
        self.res = torch.zeros_like(td_error)
        self.B, self.T, self.N =  td_error.shape

        #根据掩膜来结构化复制epsilon参数，方便后面计算
        self.mask = mask
        epsilon = mask * args.selected_epsilon
        #根据PER算式计算每个td-error的采样概率
        td_error_epi = torch.abs(td_error) + epsilon
        td_error_epi_alpha = td_error_epi ** self.alpha
        self.prob = (td_error_epi_alpha / td_error_epi_alpha.sum())
        
        # beta
        self.max_step = args.t_max
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end

    def sample(self, n):
        # 函数：从PER中采样n个数据。输出的是采样掩膜权重，乘上td-error矩阵就得到采样结果了
        # 注：在实际执行中，这个PER是每次训练生成一个，采样后就丢弃的

        #使用torch的按概率分布采样函数计算采样结果。这里直接对[0,n-1]的列表进行采样，后面再实装值
        sampled_pos = torch.multinomial(self.prob.reshape(-1), n, replacement=True)

        #计算每个采样结果的坐标
        index = sampled_pos
        pos_2 = index % self.N
        index = index // self.N
        pos_1 = index % self.T
        index = index // self.T
        pos_0 = index % self.B
        for i in range(n):#根据坐标制造掩膜，乘上掩膜就是完成采样了
            self.res[pos_0[i],pos_1[i],pos_2[i]] += 1

        return self.res
    
    def sample_weight(self, n, step):
        # 函数：从PER中采样n个数据。输出的是采样掩膜权重，乘上td-error矩阵就得到采样结果了
        #这个版本加了一个优先级权重
        # 注：在实际执行中，这个PER是每次训练生成一个，采样后就丢弃的
        sampled_pos = torch.multinomial(self.prob.reshape(-1), n, replacement=True) 
        N = self.B * self.T * self.N #计算td-error总数
        #计算beta值
        beta = (self.beta_end - self.beta_start)*step/self.max_step + self.beta_start
        #计算优先级权重以修正采样引起的分布偏差
        weight = torch.pow(1 / (self.prob * N + 1e-8), beta) * self.mask
        norm_weight = weight/ weight.max()#归一化权重

        #计算每个采样结果的坐标
        index = sampled_pos
        pos_2 = index % self.N
        index = index // self.N
        pos_1 = index % self.T
        index = index // self.T
        pos_0 = index % self.B
        for i in range(n):#和sample函数的唯一区别就是这边的掩膜值是归一化的权重
            self.res[pos_0[i],pos_1[i],pos_2[i]] += norm_weight[pos_0[i],pos_1[i],pos_2[i]]
        return self.res
            

