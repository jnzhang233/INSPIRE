import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
from ER.PER.prioritized_memory import PER_Memory

class VDN_Differ_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')


        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        
        
    
        self.mixer_params = list(self.mixer.parameters())#vdn的可学习参数为空
        self.q_params = list(mac.parameters())
        # self.mixer_optimiser = Adam(params=self.mixer_params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        self.q_optimiser = Adam(params=self.q_params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))


        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)

        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        indi_terminated = batch["indi_terminated"][:, :-1].float()
        
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            # target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
            target_mac_out = th.stack(target_mac_out[1:], dim=1)

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals_clone = target_max_qvals.clone().detach()
            target_max_q_tot_vals = self.target_mixer(target_max_qvals_clone, batch["state"][:, 1:])     

            if getattr(self.args, 'q_lambda', False):#use double-Q
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_q_tot_vals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_q_tot_vals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals_clone = chosen_action_qvals.clone().detach()
        chosen_action_qvals_clone.requires_grad = True 
        chosen_action_q_tot_vals = self.mixer(chosen_action_qvals_clone, batch["state"][:, :-1])

        td_error = (chosen_action_q_tot_vals - targets.detach())#(B,T,1)
        td_error2 = 0.5 * td_error.pow(2)
        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask



        mixer_loss  = L_td = masked_td_error.sum() / mask.sum()

        # Optimise
        # self.mixer_optimiser.zero_grad()
        
        chosen_action_qvals_clone.retain_grad() #the grad of qi
        chosen_action_q_tot_vals.retain_grad() #the grad of qtot
        
        mixer_loss.backward()#由于vdn优化器取消，mixer_loss可能没优化
        
        
        grad_l_qtot = chosen_action_q_tot_vals.grad.repeat(1, 1, self.args.n_agents) + 1e-8
        grad_l_qi = chosen_action_qvals_clone.grad
        grad_qtot_qi = th.clamp(grad_l_qi/ grad_l_qtot, min=-10, max=10)#(B,T,n_agents)

        mixer_grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params, self.args.grad_norm_clip)
        # self.mixer_optimiser.step()

        q_rewards = self.cal_indi_reward(grad_qtot_qi, td_error, chosen_action_qvals, target_max_qvals, indi_terminated) #(B,T,n_agents)
        
        
        q_rewards_clone = q_rewards.clone().detach()

        # Calculate 1-step Q-Learning targets
        q_targets = q_rewards_clone + self.args.gamma * (1 - indi_terminated) * target_max_qvals #(B,T,n_agents)

        # Td-error
        q_td_error = (chosen_action_qvals - q_targets.detach()) #(B,T,n_agents)

        q_mask = batch["filled"][:, :-1].float().repeat(1, 1, self.args.n_agents) #(B,T,n_agents)
        q_mask[:, 1:] = q_mask[:, 1:] * (1 - indi_terminated[:, :-1]) * (1 - terminated[:, :-1]).repeat(1, 1, self.args.n_agents)
        # q_mask[:, 1:] = q_mask[:, 1:] * (1 - indi_terminated[:, :-1])
        
        q_mask = q_mask.expand_as(q_td_error)
        

        masked_q_td_error = q_td_error * q_mask 
        q_selected_weight, selected_ratio = self.select_trajectory(masked_q_td_error.abs(), q_mask, t_env)
        q_selected_weight = q_selected_weight.clone().detach()
        # 0-out the targets that came from padded data

        # Normal L2 loss, take mean over actual data
        q_loss = (masked_q_td_error ** 2 * q_selected_weight).sum() / q_mask.sum()
        # Optimise
        self.q_optimiser.zero_grad()
        q_loss.backward()
        
        q_grad_norm = th.nn.utils.clip_grad_norm_(self.q_params, self.args.grad_norm_clip)
        self.q_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("mixer_loss", mixer_loss.item(), t_env)
            self.logger.log_stat("grad_norm", q_grad_norm, t_env)
            q_mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_q_td_error.abs().sum().item()/q_mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_q_tot_vals * q_mask).sum().item()/(q_mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (q_targets * q_mask).sum().item()/(q_mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

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
        th.save(self.q_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

        
    # def cal_indi_reward(grad_qtot_qi, td_error, chosen_action_qvals, target_max_qvals):
    def cal_indi_reward(self, grad_qtot_qi, mixer_td_error, qi, target_qi, indi_terminated):
        # print(grad_qtot_qi.shape,mixer_td_error.shape, qi.shape, target_qi.shape, indi_terminated.shape)
        #input: grad_qtot_qi (B,T,n_agents)  mixer_td_error (B,T,1)  qi (B,T,n_agents) target_qi(B,T,n_agents) indi_terminated (B,T,n_agents)
        grad_td = th.mul(grad_qtot_qi, mixer_td_error.repeat(1, 1, self.args.n_agents)) #(B,T,n_agents)
        reward_i = - grad_td + qi - self.args.gamma * (1 - indi_terminated) * target_qi
        return reward_i

    def select_trajectory(self, td_error, mask, t_env):
        # td_error (B, T, n_agents)
        #self.args.warm_up:
        if t_env/self.args.t_max<=self.args.warm_up_ratio:
            selected_ratio = t_env * (self.args.selected_ratio_end - self.args.selected_ratio_start)/(self.args.t_max * self.args.warm_up_ratio) + self.args.selected_ratio_start
        else:
            selected_ratio = self.args.selected_ratio_end

        #args.selected == 'PER_weight':
        memory_size = int(mask.sum().item())
        selected_num = int(memory_size * selected_ratio/4)#128的batch size过大
        return  PER_Memory(self.args, td_error, mask).sample_weight(selected_num, t_env), selected_ratio