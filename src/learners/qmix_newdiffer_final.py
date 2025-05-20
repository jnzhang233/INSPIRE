import copy
import time

import numpy
import torch
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from ER.PER.prioritized_memory import PER_Memory
from modules.inspire_modules.transformer_scorer import Transformer_Scorer


class INSPIRE_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.target_mixer = copy.deepcopy(self.mixer)

        self.mixer_params = list(self.mixer.parameters())
        self.q_params = list(mac.parameters())
        self.mixer_optimiser = RMSprop(params=self.mixer_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.q_optimiser = RMSprop(params=self.q_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        indi_terminated = batch["indi_terminated"][:, :-1].float()
        visibility_matrix = batch["visibility_matrix"][:, :-1].float()
        # Calculate estimated Q-Values
        mac_out = []
        embedding_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):#将transformer的forward拆为两步，从而获取embedding
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # ——————————————————————————————————————————————————————————————————————————————————————————————————————————
        batch_size,seq,n_agents= visibility_matrix.shape
        matrix = torch.zeros((batch_size,seq,n_agents,n_agents),dtype=torch.int)
        visibility_matrix = visibility_matrix.to(torch.int)
        for bit_position in range(n_agents):
            matrix[:, :, :, bit_position] = (visibility_matrix >> (n_agents - 1  - bit_position)) & 1
        visibility_matrix = matrix
        visibility_matrix = visibility_matrix.to(torch.bool)
        visibility_matrix = visibility_matrix == False #反转一下做掩膜，让在视野范围的为1，不在视野范围内的为0
        visibility_matrix = visibility_matrix.to(self.args.device)
        # ——————————————————————————————————————————————————————————————————————————————————————————————————————————


        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals_clone = chosen_action_qvals.clone().detach()
            chosen_action_qvals_clone.requires_grad = True 
            target_max_qvals_clone = target_max_qvals.clone().detach()
            chosen_action_q_tot_vals = self.mixer(chosen_action_qvals_clone, batch["state"][:, :-1])
            target_max_q_tot_vals = self.target_mixer(target_max_qvals_clone, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_q_tot_vals #(B,T,1)

        # Td-error
        td_error = (chosen_action_q_tot_vals - targets.detach()) #(B,T,1)

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data。这个是mix网络的loss。
        mixer_loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise。优化部分
        self.mixer_optimiser.zero_grad()
        chosen_action_qvals_clone.retain_grad() #the grad of qi。智能体i的梯度
        chosen_action_q_tot_vals.retain_grad() #the grad of qtot。智能体群的全局Q梯度
        mixer_loss.backward()#优化mixer

        grad_l_qtot = chosen_action_q_tot_vals.grad.repeat(1, 1, self.args.n_agents) + 1e-8 #把全局Q梯度复制为列数为智能体数目的矩阵
        grad_l_qi = chosen_action_qvals_clone.grad #复制智能体i的梯度
        #将个体Q除以全局Q，并限制数据范围在[-10,10].注意：原文这个算式用的比值是q_tot / q_i。但是代码实现时候是q_i / q_tot
        grad_qtot_qi = th.clamp(grad_l_qi/ grad_l_qtot, min=-10, max=10)#(B,T,n_agents)

        #裁剪mixer梯度，令其不大于args.grad_norm_clip
        mixer_grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params, self.args.grad_norm_clip)
        self.mixer_optimiser.step()#执行mixer更新

        #下面进行agent网络的更新

        #计算个体奖励值，并克隆个体奖励值
        q_rewards = self.cal_indi_reward(grad_qtot_qi, td_error, chosen_action_qvals, target_max_qvals, indi_terminated) #(B,T,n_agents)
        q_rewards_clone = q_rewards.clone().detach()#复制一下用于后续计算，防止影响梯度传播

        # Calculate 1-step Q-Learning targets。计算智能体的td-error值。
        q_targets = q_rewards_clone + self.args.gamma * (1 - indi_terminated) * target_max_qvals #(B,T,n_agents)
        q_td_error = (chosen_action_qvals - q_targets.detach()) #(B,T,n_agents)

        #计算td-error的掩膜，去除智能体已死时、轨迹结束时的td-error
        q_mask = batch["filled"][:, :-1].float().repeat(1, 1, self.args.n_agents) #(B,T,n_agents)
        q_mask[:, 1:] = q_mask[:, 1:] * (1 - indi_terminated[:, :-1]) * (1 - terminated[:, :-1]).repeat(1, 1, self.args.n_agents)
        # q_mask[:, 1:] = q_mask[:, 1:] * (1 - indi_terminated[:, :-1])
        q_mask = q_mask.expand_as(q_td_error)
        q2_mask = q_mask #存档以备信息记录
        #上掩膜去除垃圾数据
        masked_q_td_error = q_td_error * q_mask

        masked_q_td_error,q_mask = self.ESR_with_priority(masked_q_td_error,visibility_matrix = visibility_matrix, t=t)
                                                               #注意，会覆盖原有的masked_q_td_error。不过后续运算过程加不加这个函数都一样
        #计算td-error的采样权重和采样比例
        q_selected_weight, selected_ratio = self.select_trajectory(masked_q_td_error.abs(), q_mask, t_env)
        q_selected_weight = q_selected_weight.clone().detach()
        # 0-out the targets that came from padded data

        # Normal L2 loss, take mean over actual data。根据采样结果计算智能体的loss
        q_loss = (masked_q_td_error ** 2 * q_selected_weight).sum() / q_mask.sum()

        # Optimise。优化智能体网络
        self.q_optimiser.zero_grad()
        q_loss.backward()
        q_grad_norm = th.nn.utils.clip_grad_norm_(self.q_params, self.args.grad_norm_clip)
        self.q_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("selected_ratio", selected_ratio, t_env)
            self.logger.log_stat("mixer_loss", mixer_loss.item(), t_env)
            self.logger.log_stat("mixer_grad_norm", mixer_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("mixer_td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("mixer_target_mean", (targets * mask).sum().item()/mask_elems, t_env)

            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("q_grad_norm", q_grad_norm, t_env)
            q_mask_elems = q_mask.sum().item()
            q2_mask_elems = q2_mask.sum().item()
            self.logger.log_stat("q_td_error_abs", (masked_q_td_error.abs().sum().item()/q_mask_elems), t_env)
            self.logger.log_stat("q_q_taken_mean", (chosen_action_qvals * q2_mask).sum().item()/(q2_mask_elems), t_env)
            self.logger.log_stat("mixer_target_mean", (q_targets * q2_mask).sum().item()/(q2_mask_elems), t_env)
            self.logger.log_stat("reward_i_mean", (q_rewards * q2_mask).sum().item()/(q2_mask_elems), t_env)
            self.logger.log_stat("q_selected_weight_mean", (q_selected_weight * q_mask).sum().item()/(q_mask_elems), t_env)

            self.log_stats_t = t_env
    
    # def cal_indi_reward(grad_qtot_qi, td_error, chosen_action_qvals, target_max_qvals):
    def cal_indi_reward(self, grad_qtot_qi, mixer_td_error, qi, target_qi, indi_terminated):
        #DIFFER独有，计算个体奖励的函数。
        # ___________________
        # 输入：
        # grad_qtot_qi：个体Q梯度和全局Q梯度的比值
        # mixer_td_error：mix网络给出的td—error
        # qi:智能体i的Q值
        # target_qi：target网络中智能体i的Q值
        # indi_terminated：智能体终止标志，表示智能体是否存活
        # ___________________
        # 输出：
        # reward_i:个体奖励值

        # input: grad_qtot_qi (B,T,n_agents)  mixer_td_error (B,T,1)  qi (B,T,n_agents)  indi_terminated (B,T,n_agents)
        #获取个体Q*mix网络TD-error/全局Q的值
        grad_td = th.mul(grad_qtot_qi, mixer_td_error.repeat(1, 1, self.args.n_agents)) #(B,T,n_agents)
        # 计算个体奖励值。注意：原文这个算式用的比值是q_tot / q_i。但是代码实现时候是q_i / q_tot
        reward_i = - grad_td + qi - self.args.gamma * (1 - indi_terminated) * target_qi
        return reward_i

    def select_trajectory(self, td_error, mask, t_env):
        # 函数：DIFFER独有，计算采样权重和采样比例
        # 输入：
        # td_error:待采样的智能体个体td_error
        # mask:td_error的掩膜，用于去除智能体已死或者轨迹结束后的td_error
        # t_env：时间步
        # 输出：
        # 权重和采样比例

        # td_error (B, T, n_agents)
        if self.args.warm_up:#执行warm-up，随着训练推移线性更新采样比例selected_radio
            if t_env/self.args.t_max<=self.args.warm_up_ratio:
                selected_ratio = t_env * (self.args.selected_ratio_end - self.args.selected_ratio_start)/(self.args.t_max * self.args.warm_up_ratio) + self.args.selected_ratio_start
            else:
                selected_ratio = self.args.selected_ratio_end
        else:
            selected_ratio = self.args.selected_ratio

        if self.args.selected == 'all':#采样策略：全部返回。消融时候用的
            return th.ones_like(td_error).cuda(), selected_ratio
        elif self.args.selected == 'greedy':#采样策略：贪婪策略
            valid_num = mask.sum().item() #借助掩膜获取有效数据总数
            selected_num = int(valid_num * selected_ratio) #计算采样个数
            td_reshape = td_error.reshape(-1) #改为线性列表
            sorted_td, _ = th.topk(td_reshape, selected_num) #从中选取最大的selected_num个值
            pivot = sorted_td[-1] #获取其中第selected_num个最大值作为阈值，方便最后筛选TD-error
            # 采样，保留最大的selected_num个值，同时保留矩阵结构
            weight = th.where(td_error>=pivot, th.ones_like(td_error), th.zeros_like(td_error))
            return weight, selected_ratio
        elif self.args.selected == 'greedy_weight':#采样策略：贪婪权重
            valid_num = mask.sum().item()
            selected_num = int(valid_num * selected_ratio)
            td_reshape = td_error.reshape(-1)
            sorted_td, _ = th.topk(td_reshape, selected_num)
            pivot = sorted_td[-1]
            weight = th.where(td_error>=pivot, td_error-pivot, th.zeros_like(td_error))
            #前面的和greedy完全一样，只是做了个归一化，防止极端值干扰
            norm_weight = weight/weight.max()
            return norm_weight, selected_ratio
        elif self.args.selected == 'PER_hard':#采样策略：PER——hard。直接返回sample结果
            memory_size = int(mask.sum().item())
            selected_num = int(memory_size * selected_ratio)
            return  PER_Memory(self.args, td_error, mask).sample(selected_num), selected_ratio
        elif self.args.selected == 'PER_weight':#采样策略：PER——weight，直接返回使用sample_weight的采样结果
            memory_size = int(mask.sum().item())
            selected_num = int(memory_size * selected_ratio)
            return  PER_Memory(self.args, td_error, mask).sample_weight(selected_num, t_env), selected_ratio

    def sum_and_sig(self, masked_q_td_error, down_value, up_value):
        # 函数：计算智能体tderror的均值，标准差，均值+-标准差的结果
        # 输入：掩膜后的智能体tderror列表masked_q_td_error。待填写的上下限清单down_value，up_value
        # 输出：每个batch的均值each_batch_mean，标准差each_batch_std，下限down_value，上限up_value
        each_batch_mean = th.mean(masked_q_td_error, dim=1) #计算均值
        each_batch_std = th.std(masked_q_td_error, dim=1) #计算标准差

        down_value = each_batch_mean - each_batch_std
        up_value = each_batch_mean + each_batch_std
        return each_batch_mean, each_batch_std, down_value, up_value

    def ESR_with_priority(self,masked_q_td_error,visibility_matrix,t):
        #函数：基于正态分布实现选择性的经验分享和接收算法，这版本针对Transformer改过用法了
        # 输入：
        # masked_q_td_error，掩膜处理后的智能体个体TD-erorr矩阵.
        # visibility_matrix:可见性矩阵.size为[batch_size,seq_len,n_agents.n_agents]
        # t:时间步
        # 输出：
        # received_q_td_error，masked_q_td_error和接收经验列表receive_list的组合
        # q_mask:掩膜，记录有效位置

        # 获取batch_size,seq_len,n_agents的值
        self.batch_size = masked_q_td_error.shape[0]
        self.seq_len = masked_q_td_error.shape[1]
        self.n_agents = masked_q_td_error.shape[2]

        #计算共享td-error的清单
        # 计算每个agent的每个经验批次的方差和均值
        masked_q_td_error_clone = masked_q_td_error.clone().detach()  # 复制一下操作以防更改原值
        mean_td_error = masked_q_td_error_clone.mean(dim=[1])
        var_td_error = masked_q_td_error_clone.var(dim=[1])
        var_td_error = torch.clamp(var_td_error , min=float(self.args.min_eps)) #防止为0

        #对每个agent的高斯概率分布，计算一遍概率密度函数并求和归一化
        probabilities_sum = torch.zeros_like(masked_q_td_error_clone)
        for agent_index in range(self.n_agents):#对每个agent的概率分布计算一次
            #扩充agent_index的概率分布到所有agent
            mean_td_error_agent = mean_td_error[:, agent_index].unsqueeze(1).unsqueeze(2).expand(self.batch_size, self.seq_len, self.n_agents)
            var_td_error_agent = var_td_error[:, agent_index].unsqueeze(1).unsqueeze(2).expand(self.batch_size, self.seq_len, self.n_agents)
            #完成一次概率密度函数计算
            var_td_error_power = (var_td_error_agent * float(self.args.probability_temperature)) ** 2
            exponent = -((masked_q_td_error_clone - mean_td_error_agent) ** 2) / (2 * var_td_error_power)
            numerator = torch.exp(exponent)
            denominator = torch.sqrt(2 * numpy.pi * (var_td_error_power))
            probabilities = numerator / denominator
            # 直到这里，计算的是高斯概率密度函数，下面需要进行变换以满足需求
            probabilities = -probabilities  # 沿x轴翻转，实现越靠近均值的td-error概率越小
            probabilities = probabilities - torch.min(probabilities)  # 减去最小值以保证所有值≥0
            #计算接收值的视野矩阵掩膜
            visible_mask = visibility_matrix[:,:,agent_index]
            probabilities_sum += probabilities * visible_mask #加上相对于这个概率分布的概率密度函数
        # 对每个batch的每个agent的td-error，做归一化以得到概率分布
        probabilities = probabilities_sum / (torch.sum(probabilities_sum,dim = 1,keepdim=True).expand(self.batch_size, self.seq_len, self.n_agents) + float(self.args.min_eps))

        #计算采样比例
        if self.args.ESR_warm_up:#执行warm-up，随着训练推移线性更新采样比例selected_radio
            if t/self.args.t_max<=self.args.ESR_warm_up_ratio:
                selected_ratio = t * (self.args.ESR_selected_ratio_end - self.args.ESR_selected_ratio_start)/(self.args.t_max * self.args.ESR_warm_up_ratio) + self.args.ESR_selected_ratio_start
            else:
                selected_ratio = self.args.ESR_selected_ratio_end
        else:
            selected_ratio = self.args.ESR_selected_ratio
        num_sample =max( int((self.seq_len - 1) * (self.n_agents - 1) * selected_ratio),1) #每个批次的采样数目，最小不小于1
        share_gate_mask = torch.zeros_like(masked_q_td_error_clone)
        for i in range(self.batch_size):#对每个批次执行一次采样
            batch_pro = probabilities[i].view(-1)
            sampled_index = torch.multinomial(batch_pro,num_samples=num_sample,replacement=False)
            #修正坐标
            sampled_index_index_seq = sampled_index % (self.seq_len - 1)
            sampled_index = sampled_index // (self.seq_len - 1)
            sampled_index_index_agent = sampled_index % (self.n_agents - 1)
            share_gate_mask[i,sampled_index_index_seq,sampled_index_index_agent] = 1
        #使用掩膜生成分享清单
        share_list = masked_q_td_error_clone * share_gate_mask
        # ——————————————————————————————————————————————————————————————————————————————————————————————————————————

        # 计算接收td-error的清单
        #直接接收
        receive_list = torch.zeros(
            [self.batch_size, self.seq_len, self.args.n_agents,self.args.n_agents]).to(self.args.device)  # 创建一个空矩阵存received_experience
        # 创建agent索引张量
        re_agent_indices = th.arange(self.args.n_agents)
        # 创建禁止向自己分享的掩膜
        re_mask_notself = (re_agent_indices.unsqueeze(0) != re_agent_indices.unsqueeze(1))  # (n_agents, n_agents)
        re_mask_notself = re_mask_notself.unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.seq_len,
                                                                           self.args.n_agents,
                                                                           self.args.n_agents)  # (batch_size, seq_len, n_agents, n_agents)
        re_mask_notself = re_mask_notself.to(self.args.device)

        #创建禁止向不可见对象分享经验的掩膜
        re_mask_visible = visibility_matrix.to(torch.bool).to(self.args.device)

        # 进行条件赋值
        if self.args.use_visible_matrix == True:
            receive_list = th.where(re_mask_notself & re_mask_visible, share_list.unsqueeze(-1).expand(-1, -1, -1, self.n_agents),
                                    receive_list)
        else:
            receive_list = th.where(re_mask_notself, share_list.unsqueeze(-1).expand(-1, -1, -1, self.n_agents),
                                receive_list)
        # 每个agent取其接收值的最大值
        abs_receive = torch.abs(receive_list)
        re_max_indices = th.argmax(abs_receive, dim=3, keepdim=True)  # 找到绝对值最大值的索引
        receive_list = th.gather(receive_list, dim=3, index=re_max_indices).squeeze(-1)
        receive_list = receive_list.to(self.args.device)
        # ——————————————————————————————————————————————————————————————————————————————————————————————————————————

        # 链接receive_list和个体TD-error
        received_q_td_error = torch.cat((masked_q_td_error, receive_list), dim=0)

        #计算掩膜q_mask，标识非0值位置
        q_mask = received_q_td_error != 0
        return received_q_td_error,q_mask


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.q_optimiser.state_dict(), "{}/q_opt.th".format(path))
        th.save(self.mixer_optimiser.state_dict(), "{}/mixer_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.q_optimiser.load_state_dict(th.load("{}/q_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer_optimiser.load_state_dict(th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))