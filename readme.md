## PYARML3对differ做的适应：

### 模块添加部分

根据github工程https://github.com/cathyhxh/DIFFER/tree/master

1. 在config/algs添加differ_qmix.yml如下：

   ```yaml
   # --- DIFFER specific parameters ---
   #基于qmix的differ在smac运行情况良好，但是在GRF上不佳
   
   # use epsilon greedy action selector
   action_selector: "epsilon_greedy"
   epsilon_start: 1.0
   epsilon_finish: 0.05
   epsilon_anneal_time: 50000
   
   runner: "episode"
   
   buffer_size: 5000
   
   # update the target network every {} episodes
   target_update_interval: 200
   
   # use the Q_Learner to train
   agent_output_type: "q"
   learner: "q_divide_learner" #Differ的QMIX版本
   double_q: True
   mixer: "qmix"
   mixing_embed_dim: 32
   hypernet_layers: 2
   hypernet_embed: 64
   
   #独有参数
   selected: "PER_weight"
   warm_up: True
   selected_alpha: 0.8  #计算优先级分数用的参数
   selected_ratio: 0.6  #采样比例
   selected_ratio_start: 0.8  #采样比例的起始值
   selected_ratio_end: 1.0  #采样比例的最终值
   warm_up_ratio: 0.6  #热身区间占总训练轮次的比例。在热身区间，selected_radio会从start线性增长到end
   selected_epsilon: 0.01  #采样可能性的参数
   beta_start: 0.6  #PER优先级采样中，beta的起始值
   beta_end: 1  #PER优先级采样中，beta的最终值
   
   name: "differ_qmix"
   ```

2. 在learners添加q_learner_divide.py(github仓库中),在init.py写入：

   ```python
   from .q_learner_divide import QDivedeLearner
   REGISTRY["q_divide_learner"] = QDivedeLearner
   ```

3. 添加github仓库的ER文件夹。

### 实验工程改造部分

#### 更新实验环境

1. src/envs/smac_v1/official/starcraft2.py和src/envs/smac_v2/official/starcraft2.py写入：

   ```python
       （最后面） 
       def get_indi_terminated(self):
           """Returns the terminated of all agents in a list."""
           terminate = []
           for agent_id in range(self.n_agents):
               unit = self.get_unit_by_id(agent_id)
               if unit.health > 0:
                   terminate.append(0)
               else:
                   terminate.append(1)
           return terminate
       
       #后面的部分疑似没有用到
           def get_obs_agent_kaitu(self, agent_id):
           """Returns observation for agent_id. The observation is composed of:
   
              - agent movement features (where it can move to, height information and pathing grid)
              - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
              - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
              - agent unit features (health, shield, unit_type)
   
              All of this information is flattened and concatenated into a list,
              in the aforementioned order. To know the sizes of each of the
              features inside the final list of features, take a look at the
              functions ``get_obs_move_feats_size()``,
              ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
              ``get_obs_own_feats_size()``.
   
              The size of the observation vector may vary, depending on the
              environment configuration and type of units present in the map.
              For instance, non-Protoss units will not have shields, movement
              features may or may not include terrain height and pathing grid,
              unit_type is not included if there is only one type of unit in the
              map etc.).
   
              NOTE: Agents should have access only to their local observations
              during decentralised execution.
           """
           unit = self.get_unit_by_id(agent_id)
   
           move_feats_dim = self.get_obs_move_feats_size()
           enemy_feats_dim = self.get_obs_enemy_feats_size()
           ally_feats_dim = self.get_obs_ally_feats_size()
           own_feats_dim = self.get_obs_own_feats_size()
   
           move_feats = np.zeros(move_feats_dim, dtype=np.float32)
           enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
           ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
           own_feats = np.zeros(own_feats_dim, dtype=np.float32)
   
           if unit.health > 0:  # otherwise dead, return all zeros
               x = unit.pos.x
               y = unit.pos.y
               sight_range = self.unit_sight_range(agent_id)
   
               # Movement features
               avail_actions = self.get_avail_agent_actions(agent_id)
               for m in range(self.n_actions_move):
                   move_feats[m] = avail_actions[m + 2]
   
               ind = self.n_actions_move
   
               if self.obs_pathing_grid:
                   move_feats[
                       ind : ind + self.n_obs_pathing
                   ] = self.get_surrounding_pathing(unit)
                   ind += self.n_obs_pathing
   
               if self.obs_terrain_height:
                   move_feats[ind:] = self.get_surrounding_height(unit)
   
               # Enemy features
               for e_id, e_unit in self.enemies.items():
                   e_x = e_unit.pos.x
                   e_y = e_unit.pos.y
                   dist = self.distance(x, y, e_x, e_y)
   
                   if (
                       e_unit.health > 0
                   ):  # visible and alive
                       # Sight range > shoot range
                       enemy_feats[e_id, 0] = avail_actions[
                           self.n_actions_no_attack + e_id
                       ]  # available
                       enemy_feats[e_id, 1] = dist / sight_range  # distance
                       enemy_feats[e_id, 2] = (
                           e_x - x
                       ) / sight_range  # relative X
                       enemy_feats[e_id, 3] = (
                           e_y - y
                       ) / sight_range  # relative Y
   
                       ind = 4
                       if self.obs_all_health:
                           enemy_feats[e_id, ind] = (
                               e_unit.health / e_unit.health_max
                           )  # health
                           ind += 1
                           if self.shield_bits_enemy > 0:
                               max_shield = self.unit_max_shield(e_unit)
                               enemy_feats[e_id, ind] = (
                                   e_unit.shield / max_shield
                               )  # shield
                               ind += 1
   
                       if self.unit_type_bits > 0:
                           type_id = self.get_unit_type_id(e_unit, False)
                           enemy_feats[e_id, ind + type_id] = 1  # unit type
   
               # Ally features
               al_ids = [
                   al_id for al_id in range(self.n_agents) if al_id != agent_id
               ]
               for i, al_id in enumerate(al_ids):
   
                   al_unit = self.get_unit_by_id(al_id)
                   al_x = al_unit.pos.x
                   al_y = al_unit.pos.y
                   dist = self.distance(x, y, al_x, al_y)
   
                   if (
                       al_unit.health > 0
                   ):  # visible and alive
                       ally_feats[i, 0] = 1  # visible
                       ally_feats[i, 1] = dist / sight_range  # distance
                       ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                       ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y
   
                       ind = 4
                       if self.obs_all_health:
                           ally_feats[i, ind] = (
                               al_unit.health / al_unit.health_max
                           )  # health
                           ind += 1
                           if self.shield_bits_ally > 0:
                               max_shield = self.unit_max_shield(al_unit)
                               ally_feats[i, ind] = (
                                   al_unit.shield / max_shield
                               )  # shield
                               ind += 1
   
                       if self.unit_type_bits > 0:
                           type_id = self.get_unit_type_id(al_unit, True)
                           ally_feats[i, ind + type_id] = 1
                           ind += self.unit_type_bits
   
                       if self.obs_last_action:
                           ally_feats[i, ind:] = self.last_action[al_id]
   
               # Own features
               ind = 0
               if self.obs_own_health:
                   own_feats[ind] = unit.health / unit.health_max
                   ind += 1
                   if self.shield_bits_ally > 0:
                       max_shield = self.unit_max_shield(unit)
                       own_feats[ind] = unit.shield / max_shield
                       ind += 1
   
               if self.unit_type_bits > 0:
                   type_id = self.get_unit_type_id(unit, True)
                   own_feats[ind + type_id] = 1
   
           agent_obs = np.concatenate(
               (
                   move_feats.flatten(),
                   enemy_feats.flatten(),
                   ally_feats.flatten(),
                   own_feats.flatten(),
               )
           )
   
           if self.obs_timestep_number:
               agent_obs = np.append(agent_obs,
                                     self._episode_steps / self.episode_limit)
   
           if self.debug:
               logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
               logging.debug("Avail. actions {}".format(
                   self.get_avail_agent_actions(agent_id)))
               logging.debug("Move feats {}".format(move_feats))
               logging.debug("Enemy feats {}".format(enemy_feats))
               logging.debug("Ally feats {}".format(ally_feats))
               logging.debug("Own feats {}".format(own_feats))
   
           return agent_obs
   
       def get_obs_kaitu(self):
           """Returns all agent observations in a list.
           NOTE: Agents should have access only to their local observations
           during decentralised execution.
           """
           agents_obs_kaitu = [self.get_obs_agent_kaitu(i) for i in range(self.n_agents)]
           return agents_obs_kaitu
   ```

2. 在run/run.py中写入：

   ```python
   第124行的scheme，加入indi_terminated
   scheme = {
           "state": {"vshape": env_info["state_shape"]},
           "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
           "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
           "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
           "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
           "reward": {"vshape": (1,)},
           "terminated": {"vshape": (1,), "dtype": th.uint8},
           "indi_terminated": {"vshape": (env_info["n_agents"],), "dtype": th.uint8},#DIFFER特有的
       }
   ```

3. 在episode_runner中新增了一个专用的differrunner，写入：

   ```python
   第87行加入indi_terminated
   post_transition_data = {
                   "actions": cpu_actions,
                   "reward": [(reward,)],
                   "terminated": [(terminated != env_info.get("episode_limit", False),)],
                   "indi_terminated": [self.env.get_indi_terminated()]
               }
   ```

4. 在在src/envs/gfootball/FootballEnv.py：

   ```python
       def get_indi_terminated(self):
           #differ用的，需要个体存活标签。
           terminate = []
           for agent in range(self.n_agents):
               if self.full_obs["left_team_yellow_card"][agent] == False:
                   terminate.append(0)
               else:
                   terminate.append(1)
           return terminate
   ```

   这个官方没做，我们自己做的

5. 在config/.yaml中添加运行参数以使用专用部件：

   scheme:"differ"

   runner:"episode_differ"

#### 我们方法要求的更新

1. 在src/envs/gfootball/FootballEnv.py：

   ```python
   在init函数中加入了可选输入sight_field=0.2，并添加下面这行：
   self.sight_field = sight_field #视野范围，自定义
   
   ```

2. 

3. 在src/runners，添加episode_runner_inspire.py：

   ```python
   其他地方一样，在87行，改成：
   post_transition_data = {
                   "actions": cpu_actions,
                   "reward": [(reward,)],
                   "terminated": [(terminated != env_info.get("episode_limit", False),)],
                   "indi_terminated": [self.env.get_indi_terminated()],
                   "visibility_matrix":[self.env.get_ally_visibility_matrix()]
               }
   ```

4. 在run.py:

   ```python
       if args.scheme == "inspire":#inspire专用的scheme，硬性规定了单位存活标签信息和可视标签矩阵
           scheme = {
               "state": {"vshape": env_info["state_shape"]},
               "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
               "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
               "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
               "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
               "reward": {"vshape": (1,)},
               "terminated": {"vshape": (1,), "dtype": th.uint8},
               "indi_terminated": {"vshape": (env_info["n_agents"],), "dtype": th.uint8},  # 单位存活标签
               "visibility_matrix": {"vshape": (env_info["n_agents"],), "dtype": th.uint8}, # 可视标签
           }
   ```



在.yaml中加入参数：

scheme = inspire

runner = episode_inspire

#### Kalei要求的更新

1. 更新了controller/Kalei_type_n_controller.py
2. 更新了modules/agent/Kalei_type_NRNNAgent.py
3. 更新了learners/Kalei_nq_learner.py
4. 更新了config/algs/Kalei_qmix_rnn.yaml
5. 在src/utils/rl_utils.py增加了原工程的build_td_lambda_targets函数。

#### ICES要求的更新

1. pyro:pip3 install pyro-ppl

2. 更新了src/config/algs/ices.yaml,ices_QPLEX.yaml（GRF的），内容：

   ```yaml
   # --- QMIX specific parameters ---
   
   # use epsilon greedy action selector
   action_selector: "epsilon_expl"
   epsilon_start: 1.0
   epsilon_finish: 0.0
   epsilon_anneal_time: 50000
   
   runner: "episode_ices"
   run: "ices_run_grf"
   
   buffer_size: 5000
   
   # update the target network every {} episodes
   target_update_interval: 200
   
   # use the Q_Learner to train
   agent_output_type: "q"
   learner: "ices_QPLEX"
   double_q: True
   mixer: "dmaq_qatten"
   mixing_embed_dim: 32
   hypernet_embed: 64
   adv_hypernet_layers: 1
   adv_hypernet_embed: 64
   
   num_kernel: 4
   is_minus_one: True
   is_adv_attention: True
   is_stop_gradient: True
   
   # Qatten coefficient
   n_head: 4
   attend_reg_coef: 0.001
   state_bias: True
   mask_dead: False
   weighted_head: False
   nonlinear: False
   
   num_circle: 2
   
   embedding_dim: 4
   hidden_dim: 64
   z_dim: 16
   pred_s_len: 1
   world_bl_lr: 0.0001
   world_lr: 0.0001
   world_clip_param: 0.1
   world_gamma: 0.01
   weight_decay: 0
   int_sign: False
   
   agent: "ices_grf"
   mac: "ices_mac"
   name: "ices_QPlex"
   
   #修正默认参数
   obs_agent_id: False
   ```

3. 在smacv1和smacv2的starcraft.py中添加了专用函数

4. 更新了src/controllers/ices_n_controller.py，src/controllers/ices_controller.py

5. 更新了src/learners/ices_nq_learner.py，src/learners/ices_dmaq_qatten_learner.py

6. 更新了src/runners/parallel_runner_ices.py

7. 在modules，更新了src/modules/agents/ices_n_rnn_agent.py，更新了src/modules/exp，src/modules/agents/ices_agent.py，src/modules/ices。在ices_agent.py

   ```python
          forward函数开头加入 
       if input_shape[0] == 1:
               input_shape = input_shape[1:]
               inputs = inputs.squeeze(0)
   ```

   

8. 更新了src/utils/rl_utils.py

9. 在src/components/action_selectors.py，更新了专用函数，添加import torch.nn.functional as F

10. 更新了src/components/ices_episode_buffer.py，src/run/ices_run.py。在ices.yaml加入run:"ices_run"。在ices_run.py把from components.episode_buffer import ReplayBuffer改成from components.ices_episode_buffer import ReplayBuffer。在ices_run.py：

    ```python
    在run_sequential函数开头加入以下代码以配置必要参数
    # 根据grf地图来确定必要参数
        if args.env == "ices_gfootball":
            if args.env_args["map_name"] == "academy_3_vs_1_with_keeper":
                args.int_ratio = 0.2
                args.int_finish = 0.05
                args.int_ent_coef = 0.001
                args.unit_dim = 26
            elif args.env_args["map_name"] == "academy_corner":
                args.int_ratio = 0.1
                args.int_finish = 0.05
                args.int_ent_coef = 0.002
                args.unit_dim = 34
            elif args.env_args["map_name"] == "academy_counterattack_hard":
                args.int_ratio = 0.05
                args.int_finish = 0.05
                args.int_ent_coef = 0.005
                args.unit_dim = 34
            else:
                args.int_ratio = 0.05
                args.int_finish = 0.05
                args.int_ent_coef = 0.005
                args.unit_dim = 34
        args.env_args["obs_dim"] = args.unit_dim
    ```

    在ICES_episode_buffer注释掉writereward函数（62行，144行）

11. 在utils/rl_utils.py，加入以下函数

    ```python
    def build_td_lambda_targets_ices(
        rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda
    ):
        # EXP
        if td_lambda is False:
            return rewards + gamma * (1 - terminated) * target_qs[:, 1:, :]
        # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
        # Initialise  last  lambda -return  for  not  terminated  episodes
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
        # Backwards  recursive  update  of the "forward  view"
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
                rewards[:, t]
                + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t])
            )
        # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
        return ret[:, 0:-1]
    ```

    并把learner的build_td_lambda_targets改成build_td_lambda_targets_ices

12. 加入ICES_FootballEnv，是根据ICES_GRF的GRF环境文件加了我们GRF环境init函数的输入，然后加入一个obs_dim参数即可。

13. 在config/default.yaml，加入burn_in_period: 32

14. 加入ices_run_grf,ices_grf_episode_buffer,ices_prioritize,ices_sum_tree。注释掉ices_run_grf的模型保存部分(282-296行)。

15. 在ices_run_grf.py：

    ```python
    在run_sequential函数开头加入以下代码以配置必要参数
    # 根据grf地图来确定必要参数
        if args.env == "ices_gfootball":
            if args.env_args["map_name"] == "academy_3_vs_1_with_keeper":
                args.int_ratio = 0.2
                args.int_finish = 0.05
                args.int_ent_coef = 0.001
                args.unit_dim = 26
            elif args.env_args["map_name"] == "academy_corner":
                args.int_ratio = 0.1
                args.int_finish = 0.05
                args.int_ent_coef = 0.002
                args.unit_dim = 34
            elif args.env_args["map_name"] == "academy_counterattack_hard":
                args.int_ratio = 0.05
                args.int_finish = 0.05
                args.int_ent_coef = 0.005
                args.unit_dim = 34
            else:
                args.int_ratio = 0.05
                args.int_finish = 0.05
                args.int_ent_coef = 0.005
                args.unit_dim = 34
        args.env_args["obs_dim"] = args.unit_dim
    ```

16. 添加了src/controllers/ices_based_n_controller.py，src/controllers/ices_basic_controller.py。对src/controllers/ices_n_controller.py的importNMAC，改为

    ```
    from .ices_based_n_controller import NMAC
    ```

    对ices_based_n_controller.py的importMAC，改为：

    ```
    from .ices_basic_controller import BasicMAC
    ```



## 运行指令：

目前在12服务器运行。

### 对照算法

注意：跑GRF的时候每个地图都要对一下num_agents和time_limit两个参数，不然结果会有问题

super：python src/main.py --config=super_qmix --env-config=sc2 with env_args.map_name=2s3z t_max=2005000

per：python src/main.py --config=per_qmix --env-config=sc2 with env_args.map_name=2s3z t_max=2005000

differ(qmix版本，用在smac)：python src/main.py --config=differ/differ_qmix --env-config=sc2 with env_args.map_name=2s3z t_max=2005000

differ(vdn版本，用在grf)：python src/main.py --config=differ/differ_vdn --env-config=gfootball with env_args.map_name=academy_3_vs_1_with_keeper env_args.num_agents=3 env_args.time_limit=400 t_max=2005000。跑起来比qmix版本好一些，但是也不怎么样

qmix: python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z t_max=2005000

kalei(QMIX版本，用在smac):python src/main.py --config=kalei/Kalei_qmix_rnn --env-config=sc2 with env_args.map_name=2s3z t_max=2005000 跑起来特别特别特别慢，要尽快跑

kalei(VDN版本，用在grf)：python src/main.py --config=kalei/Kalei_vdn_rnn --env-config=gfootball with env_args.map_name=academy_3_vs_1_with_keeper env_args.num_agents=3 env_args.time_limit=400 t_max=2005000 跑起来特别特别特别慢，要尽快跑

ices：python src/main.py --config=ices --env-config=sc2 with env_args.map_name=2s3z t_max=7005000

ICES_GRF（我的版本，表现不太好）:python src/main.py  --config=ices_QPLEX --env-config=ices_gfootball with env_args.map_name=academy_counterattack_hard env_args.num_agents=11 env_args.time_limit=400 t_max=2005000   失败，胜率11%且不稳定。目前改用原论文的工程跑，之后看看什么原因。原论文工程连接：https://github.com/LXXXXR/ICES/。

ICES_GRF(原工程版本)：





根据运行环境做修改：

1. SMACV1：--env-config=sc2 with env_args.map_name=地图名

2. SMACV2：--env-config=sc2_v2_zerg(地图名) with 。SMACV2是在args/envs里面的三个sc2_v2_文件里面调参数的，也可以每个地图单独调好然后保存，到时候改一下--env-config参数就可以调用到对应的

3. GRF:--env-config=gfootball with env_args.map_name=academy_3_vs_1_with_keeper env_args.num_agents=3 env_args.time_limit=400 。换地图的时候要自己查一下map_name对应的num_agents和time_limit两个参数并填上。

   --env-config=gfootball with env_args.map_name=academy_corner env_args.num_agents=4 env_args.time_limit=400

   --env-config=gfootball with env_args.map_name=academy_counterattack_hard env_args.num_agents=4 env_args.time_limit=400

   

### 方法测试

**differ_qmix版本，用的episode_runner:**

（SMACV1）python src/main.py --config=differ_qmix --env-config=sc2 with env_args.map_name=2s3z t_max=20000

（SMACV2）python src/main.py --config=qmix --env-config=sc2_v2_zerg with t_max=20000

(GRF)python3 src/main.py --config=qmix --env-config=gfootball with env_args.map_name=academy_3_vs_1_with_keeper env_args.num_agents=3 env_args.time_limit=400  t_max=20000



**运行inspire_qmix_v0:** DIFFER+正态分布门版本的ESR

（SMACV1）python src/main.py --config=inspire_qmix_v0 --env-config=sc2 with env_args.map_name=2s3z t_max=2005000

**运行inspire_qmix_V2:**

（SMACV1）python src/main.py --config=inspire_qmix_v2 --env-config=sc2 with env_args.map_name=2s3z t_max=2005000

**运行inspire_qmix:**

（SMACV1）python src/main.py --config=inspire_qmix --env-config=sc2 with env_args.map_name=2s3z t_max=3005000



测试项目：

1. differ_qmix_parallel版本，用的parallel_runner，可能更适合SMAC环境:python src/main.py --config=differ_qmix_parallel --env-config=sc2 with env_args.map_name=2s3z t_max=20000

   目前已被放弃使用，效果不好，时间倒是挺快。大概得10005000轮次才能稳定训练完成感觉

2. transformer做的agent网络：python src/main.py --config=differ_qmix_transformertest --env-config=sc2 with env_args.map_name=2s3z t_max=20000

   跑1505000轮就行，一般100万轮就收敛了。如和其他算法比较可以跑200万轮，因为其他算法可能150万轮左右收敛，再慢的就不管了。

   transformer层消融：python src/main.py --config=differ_qmix_transformertest --env-config=sc2 with env_args.map_name=2s3z t_max=2005000 transformer_n_layers=1

   transformer头消融：python src/main.py --config=differ_qmix_transformertest --env-config=sc2 with env_args.map_name=2s3z t_max=2005000 transformer_n_head=1

   embedding_dim消融：python src/main.py --config=differ_qmix_transformertest --env-config=sc2 with env_args.map_name=2s3z t_max=2005000 transformer_embedding_dim=128

   正在测试

3. ESR算法的分支改进：

   经验分享部分：python src/main.py --config=inspire_ESRtest --env-config=sc2 with env_args.map_name=2s3z t_max=2005000 probabilities_version=1

   python src/main.py --config=inspire_ESRtest --env-config=sc2 with env_args.map_name=2s3z t_max=2005000 probabilities_version=2 receive_version=0

   版本1是只对自己的分布做概率密度函数，版本2是对所有人的分布各算一次概率密度函数

   经验接收部分：python src/main.py --config=inspire_ESRtest --env-config=sc2 with env_args.map_name=2s3z t_max=2005000 probabilities_version=1 receive_version=1

   版本0是直接接收，版本1是正态分布门，版本2是sigmoid门
   
   参数消融部分：
   
   ESR_selected_ratio_end：python src/main.py --config=inspire_ESRtest --env-config=sc2 with env_args.map_name=2s3z t_max=2005000 ESR_selected_ratio_end=0.4
   
   probability_temperature：python src/main.py --config=inspire_ESRtest --env-config=sc2 with env_args.map_name=2s3z t_max=2005000 probability_temperature=1 
   
   ESR_warm_up_ratio：python src/main.py --config=inspire_ESRtest --env-config=sc2 with env_args.map_name=2s3z t_max=2005000 ESR_warm_up_ratio=0.6
   
   python src/main.py --config=inspire_ESRtest --env-config=sc2 with env_args.map_name=2s3z t_max=2005000 probabilities_version=2 receive_version=2 agent=n_rnn

## 目前在跑

**目前在跑（12服务器）：**



**目前在跑（10服务器）：**





## 改进思路

1一定要做。2可以简化，但是还用视野范围随便选的话可能被怼。4相对简单，高优先级做，可以找沈学姐。3相对简单，可以考虑做，可以找乔学长。

做到哪里算哪里

1（transformer做agent网络，生成embedding并分享）>3（per采样）>2（做视野范围（考虑引入CommomNet，Tarmmac））>4（做基于互信息的公式）.





1. 每个智能体对经验批次过transformer生成固定批次的embedding，然后分享embedding。接收端接收embedding后进行处理。可以削减通信频次。

   **存在的问题：**

   1. 在工程实践中，智能体实际分享的是TD-ERROR而不是整个经验，而且一次训练用的是batch_size个轨迹的batch，并不是实时交互和分享的。——已忽略
   3. 目前agent网络的输入是整个显式数据构成的episode_batch的每一步。而且input_size是锁死的，同时接收embedding和state很可能造成训练的混乱。直接使用embedding进行训练，也会导致训练的结果不能直接用于与环境的交互中，这是本末倒置的。——已解决，将agent网络换成了transformer。
   4. 接收embedding后解压缩为新的经验，可行性很低。而且训练工程中，所有agent的数据都在一个批次内，真有这个需要直接把完整的数据送过去就好了。——已忽略，不进行解压缩。
   
   可能的实现方式：
   
   1. 构建一个Transformer做agent网络，自己的经验走全过程，别人的embedding从attention层走。——因为agent是同构的，所以其实别人的embedding和经验进来也是那个TD-error，这一点不用太担心。
   
2. 只和视野范围内的共享。采样一些批次计算是不是在视野范围内。视野范围需要参数消融。可以考虑mean_shift。可以考虑改成用ROMA的角色聚类来实现共享对象的选择。可以看看谢老师发的局部观测通信方面的文章找找灵感。

   存在的问题:

   1. 目前经验的batch中不包括真正意义的智能体[x,y]坐标，所以需要另外更改实验环境才可以实现。——done
   2. 一次训练用的是batch_size个轨迹的batch，并不是实时交互和分享的。所以实现这个可能需要对要分享的经验逐一计算位置是否在范围内。可以试试L1距离和L2距离。——done
   3. 不在视野范围但是所处OBS相似的，也应该进行分享，但是本文并没有体现。

   如果用视野范围：

   1. 如果视野范围没有人怎么办。是略过还是扩大范围继续搜索。——忽略
   2. 视野范围的消融怎么做。——忽略

   如果用ROMA：

   1. 虽然代码好用，但是论述可能比较复杂

   如果用CommomNet，Tarmmac之类的：

   1. 调研一下,能不能直接用。也是视野范围基础上做的东西。可以问乔学长。

   如果用GCN做邻居选择：

   1. 目前不考虑？

   新思路：增加一个transformer的scorer，对agent之间的影响进行打分，根据分数概率采样，选择前K%概的分享-被分享方进行分享。

   1. 实现scorer——done

   2. 实现采样和掩膜——done。是【batch_size,n_agent,n_agent】的掩膜，[i,j,k]决定i-th batch的agent[j]是否接受来自k-th agent的经验。

   3. 为了让scorer得到训练，需要将分数乘到agent.forward的输入中。将scorer的打分矩阵score，每个agent得到的评分作为权重，然后乘到agent_input上作为输入。

      ```
      在inspire_controller:
      agent_weight = score.permute(0, 2, 1)
      agent_weight = th.sum(agent_weight,dim=2)
      在inspire_agent:
      inputs = inputs * score
      ```

      

3. 选取分享经验的机制。

   学长版本的机制是计算TD-ERROR是不是自己这个批次的极端值（正态分布的边缘值）并只分享极端值，这个是SUPER给出的一种策略。

   存在的问题：

   1. 我们可以考虑训练一个网络来为经验打分，判断经验是否值得分享。但是具体网络怎么设置还有待商榷。——目前不做

   2. 或许可以仿照PER的采样机制，让极端值有高概率被分享，但是一般的值也有一定概率被分享。被分享的概率取决于值的概率。
      1. 设置一个正态分布反函数的概率密度函数，根据TD-error进行分享。距离均值越远，分享概率越大。——已实现，version1

         设 TD-error 为 $x$，均值为 $\mu$，方差为 $\sigma^2$，温度系数为 $T$，最小稳定项为 $\epsilon$，则概率密度函数计算过程为：
         $$
         P(x) = \frac{1}{\sqrt{2\pi (T^2 \cdot \sigma^2)}} \exp\left( -\frac{(x - \mu)^2}{2T^2 \cdot \sigma^2} \right)
         $$
         对概率密度函数进行变换：
         $$
         P'(x) = -P(x)
         $$

         $$
         P''(x) = P'(x) - \min(P'(x))
         $$

         $$
         \tilde{P}(x) = \frac{P''(x)}{\sum_{t=1}^{L} P''(x_t) + \epsilon}
         $$

         最终得到满足“远离均值误差越大，概率越高”的归一化概率分布。

      2. 新增一种可行性，计算概率密度函数的时候对所有智能体的正态分布都计算一遍求和，这样可以说明联合分布下的分享价值。——已实现，version2

         设 $x_{b,t,i}$ 为第 $b$ 个 batch，第 $t$ 个时间步，第 $i$ 个 agent 的 TD-error 值，
         $\mu_j$ 和 $\sigma_j^2$ 分别为第 $j$ 个 agent 的 TD-error 均值与方差，
         $T$ 为温度参数，$\epsilon$ 为最小稳定项。

         每个 agent 的概率密度函数定义为：
         $$
         P_j(x_{b,t,i}) = \frac{1}{\sqrt{2\pi (T^2 \cdot \sigma_j^2)}} \exp\left( -\frac{(x_{b,t,i} - \mu_j)^2}{2 T^2 \cdot \sigma_j^2} \right)
         $$
         经过负变换与平移：
         $$
         \tilde{P}_j(x_{b,t,i}) = -P_j(x_{b,t,i}) - \min_{b,t,i}( -P_j(x_{b,t,i}) )
         $$
         将所有 agent 的变换后概率加和：
         $$
         S(x_{b,t,i}) = \sum_{j=1}^{n} \tilde{P}_j(x_{b,t,i})
         $$
         最终的归一化概率分布为：
         $$
         \text{FinalProb}(x_{b,t,i}) = \frac{S(x_{b,t,i})}{\sum_{t'=1}^{L} S(x_{b,t',i}) + \epsilon}
         $$

   3. 这样做了之后经验接收的部分也一定要进行配套改进，比如说高于正态分布阈值的一定接收，低于阈值的有概率接收。

      增加了sigmoid函数门：Preceive=σ(α⋅(∣e−μ∣−β⋅σ))。其中前面的σ()是sigmoid函数。aphla是敏感度控制因子，aphla越大，接收概率随td-error距离的变化越大。∣e−μ∣是对均值的距离，σ是方差。β是缩放因子，控制接收的严苛程度。

4. 共享的频次可以通过经验采样的变化熵来判断，熵越大频次越高。

   感觉有可行性，需要进一步学习变化熵的相关概念，常见用法和计算方法。可以设置一个算法来卡共享比例实现这种逻辑。

5. 实验论述方面的改进：讲一个其他算法+我们方法的效果比其他算法要好。



1. 最迟4月中/下旬应当确定实验地图，开始跑新增对照算法的实验数据。
2. 最迟5月上/中旬应当开始跑我们方法的实验数据，并完成消融实验。
3. 进入5月下旬/6月上旬，应当已经完成实验并专注于论文写作的改进。



## 周期计划

添加对于选择性经验回放或经验回放算法促进奖励稀疏下的收敛性的理论证明或理论推导。

1. 5.1开始写论文
2. 代码，最迟5.15结束。
3. 实验要一直讨论要不要加新东西。需要做一个实验计划。





## 开发日志记录

### stage1：尝试添加本科毕设优化过的代码：done

1. 新增了config/algs/inspire_qmix.yaml。新增了src/learners/qmix_newdiffer_test.py。并在src/learners/_init_.py加入对应导入代码。——done

2. 开始尝试把本科毕设优化过的经验分享、经验接收代码调整一下，加入newdiffer——done

   a. 完成了上下限计算和经验分享的code——done
   
   b. 完成了经验接收的code——done

### stage2:尝试设计一个基于transformer的agent网络：done，需要调优参数

1. 添加了src/modules/agents/trnasformer_agent.py。并在init.py添加了导入语句——done

### stage3:尝试实现idea1：基本可以跑了，只是还需要调参

1. 创建一个专用controller——done
2. 在QMIX上实现了SUPER——done
3. 设置v0版本的inspire：使用RNN_agent，用的是基于正态分布的ESR算法——done
4. 调优参数以达到理想效果——done

### stage4：尝试实现Idea3

TODOlist:

1. 修改概率密度函数为倒过来的正态分布，保证极端的有更高概率分享——done
2. 对应修改经验接收部分，确保不会把不极端的全拦截下来——done，但是sigmoid门效果奇差
3. 配置公用服务器的环境以备用。环境配置好了，代码要看看在gpu下不在一个device的问题。——done
4. 实现对所agent的正态分布各计算一次概率密度函数并求和。——done
5. 在2+0的基础上进行参数消融，尝试改进胜率

### stage5：尝试实现idea2

1. 在starcraft和grf实现visiblity_matrix的实现 ——done
2. 实现专用的runner和scheme ——done
3. learner将接受的visiblity_matrix还原成01的bool矩阵 ——done
4. 将visiblity_matrix处理成掩膜，加入经验接收部分 ——done
5. 参数消融

### stage6:尝试实现idea4

1. 阅读信息熵相关代码，思考可行方案

2. 将grf特化过的VDN-differ加入工程，QMIX-DIFFER表现可能很差

3. 尝试添加24年算法，目前有可能实现的有
   1. Kaleidoscope: Learnable Masks for Heterogeneous Multi-agent Reinforcement Learning（方学长用过了）
   2. Individual Contributions as Intrinsic Exploration Scaffolds for Multi-agent Reinforcement Learning

4. 对于kale：

   

## idea测试

在/python/画图/new_differ。数据放到data_json/exp_test(smacv1)，按格式存放即可。格式如下：

实验名/算法名/地图名。加入实验名是为了保证不同实验之间不会记录到一起。实际使用的时候记得给生成的excel另外起名字以防止被覆盖。

### 测试parallel_runner和episode_runner哪个更好使

在3005000轮次下，测试结果如图：

![](D:\study_work\python\New_Differ\picture\ablition-runner_test-2s3z.jpg)

episode_runner的话3005000轮次即可，如果用pareall_runner的话可能更推荐10005000轮次，而且明显更不稳定。目前来说推荐用episode_runner。

### 模块coding测试

#### 阶段1：

![](D:\study_work\python\New_Differ\picture\ablition-module_test-2s3z.jpg)

{'DIFFER(baseline)': 0.9855769230769232,原版differ

 'ESR-Transformer': 0.8878048780487805, 原版ESR算法+transformer_agent

'INSPIRE_ESR_using_normal_distribution': 0.9815668202764976,原版ESR

'rnn-share2-receive0': 0.9953488372093025, 我们的经验分享2+经验接收0

'originalESR': 0.976851851851852



### transformer参数消融：

用differ实现的，因为主算法还在构建。

![](D:\study_work\python\New_Differ\picture\ablition-transformer_test-2s3z.jpg)

**head消融（多头注意力头数，transformer_n_head）成绩（2005000轮次）：**

| 类型           | 最优值             | 运行时间                         | 收敛轮次（百万轮） |
| -------------- | ------------------ | -------------------------------- | ------------------ |
| head=1,layer=2 | 0.9418604651162791 | 16 hours, 57 minutes, 15 seconds | 0.65168            |
| head=2,layer=2 | 0.9532710280373832 | 16 hours, 55 minutes, 39 seconds | 1.082992           |
| head=4,layer=2 | 0.9108910891089109 | 19 hours, 24 minutes, 27 seconds | 1.092767           |
| head=8,layer=2 | 0.9086538461538461 | 19 hours, 15 minutes, 07 seconds | 1.092928           |

head的值必须可以整除embedding_dim，所以一般设置为2、4、8的倍数。在不想用多头注意力时，可以令head为1。

根据消融结果，选择head=2就可以了。

**layer消融（2005000轮次）**

| 类型           | 最优值             | 运行时间                                | 收敛轮次（百万轮） |
| -------------- | ------------------ | --------------------------------------- | ------------------ |
| head=2,layer=1 | 0.9128205128205128 | 13 hours, 7 minutes, 5 seconds          | 1.16314            |
| head=2,layer=2 | 0.9532710280373832 | 16 hours, 55 minutes, 39 seconds        | 1.082992           |
| head=3,layer=3 | 0.9015544041450777 | 22 hours, 28 minutes, 34 seconds        | 0.882727           |
| head=4,layer=4 | 0.9033816425120773 | 1 days, 1 hours, 27 minutes, 51 seconds | 1.052848           |

layer表示transformer_encoder由几个transformer_layer构成。以目前试验结果，layer=2是效果最好的。

**embedding_dim消融（2005000轮次）**

| 类型              | 最优值             | 运行时间                         | 收敛轮次（百万轮） |
| ----------------- | ------------------ | -------------------------------- | ------------------ |
| embedding_dim=64  | 0.9029126213592233 | 17 hours, 21 minutes, 13 seconds | 1.644216           |
| embedding_dim=96  | 0.9234449760765551 | 17 hours, 54 minutes, 31 seconds | 1.624229           |
| embedding_dim=112 | 0.937799043062201  | 17 hours, 49 minutes, 9 seconds  | 1.042666           |
| embedding_dim=128 | 0.9532710280373832 | 16 hours, 55 minutes, 39 seconds | 1.08299            |
| embedding_dim=144 | 0.8916256157635468 | 18 hours, 11 minutes, 11 seconds | 1.253238           |
| embedding_dim=160 | 0.8995215311004785 | 19 hours, 58 minutes, 7 seconds  | 1.51366            |

embedding_dim表示由输入数据经过线性层生成的embedding向量维度数。为便于多头注意力机制运行，一般设置为2,4,8的倍数。目前是以32为间隔进行消融实验。还测试了一下112,144的效果，但是都不太好。

### ESR测试

原版ESR：正态分布门版本的经验分享+经验接收

新版经验分享：版本1是只对自己的分布做概率密度函数，版本2是对所有人的分布各算一次概率密度函数

新版经验接收：版本0是直接接收，版本1是正态分布门，版本2是sigmoid门

| 类型                | 最优值             | 运行时间                         | 收敛轮次（百万轮） |
| ------------------- | ------------------ | -------------------------------- | ------------------ |
| 原版ESR             | 0.9768518518518519 | 13 hours, 43 minutes, 13 seconds | 1.794756           |
| 原版ESR+Transformer | 0.8967136150234741 | 19 hours, 24 minutes, 43 seconds | 0.892317           |
| 经验分享1+经验接收0 | 0.9095238095238095 | 17 hours, 53 minutes, 46 seconds | 1.112817           |
| 经验分享1+经验接收1 | 0.9014084507042254 | 18 hours, 5 minutes, 10 seconds  | 1.433636           |
| 经验分享1+经验接收2 | 0.8796296296296297 | 18 hours, 35 minutes, 49 seconds | 1.423853           |
| 经验分享2+经验接收2 | 0.8711340206185567 | 15 hours, 29 minutes, 56 seconds | 0.822215           |
| 经验分享2+经验接收1 | 0.909952606635071  | 19 hours, 31 minutes, 25 seconds | 1.935178           |
| 经验分享2+经验接收0 | 0.908675799086758  | 16 hours, 49 minutes, 21 seconds | 1.674556           |

分享1+接收2可以弃用了。分享2可以用，但是感觉参数要仔细调整一下，起码不能比原版ESR差太多。目前打算在分享2+接收0的基础上进行调优。先看看实验结果怎么样。目前确认在经验分享2+经验接收0的基础上进行调优。

#### 参数消融测试（带Transformer）

ESR_selected_ratio_end：采样比例的终止值，0-1

| 类型                                   | 最优值             | 运行时间                         | 收敛轮次（百万轮） |
| -------------------------------------- | ------------------ | -------------------------------- | ------------------ |
| ESR_selected_ratio_end=0.2             | 0.8761904761904762 | 19 hours, 20 minutes, 23 seconds | 0.992248           |
| ESR_selected_ratio_end=0.3             | 0.8780487804878049 | 19 hours, 31 minutes, 49 seconds | 0.702259           |
| ESR_selected_ratio_end=0.4             | 0.8986175115207373 | 19 hours, 27 minutes, 8 seconds  | 1.925303           |
| ESR_selected_ratio_end=0.5（目前的值） | 0.9047619047619048 | 19 hours, 29 minutes, 50 seconds | 1.725106           |
| ESR_selected_ratio_end=0.6             | 0.8669950738916257 | 19 hours, 38 minutes, 48 seconds | 0.922693           |
| ESR_selected_ratio_end=0.7             | 0.7910447761194029 | 19 hours, 9 minutes, 28 seconds  | 0.601545           |
| ESR_selected_ratio_end=0.8             | 0.8737373737373737 | 19 hours, 23 minutes, 50 seconds | 0.761839           |

目前认为采样比例在0.5比较好，也就是原值。

probability_temperature: 1。方差的放缩值.为1使用原值，>1增大，选择更加随机，<1减小，便于倾向于偏差大的经验

| 类型                                  | 最优值             | 运行时间                         | 收敛轮次（百万轮） |
| ------------------------------------- | ------------------ | -------------------------------- | ------------------ |
| probability_temperature=0.6           | 0.8732394366197183 |                                  | 1.583765           |
| probability_temperature=0.8           | 0.8732394366197183 |                                  | 0.802421           |
| probability_temperature=1（目前的值） | 0.9047619047619048 | 19 hours, 29 minutes, 50 seconds | 1.725106           |
| probability_temperature=1.2           | 0.8817733990147784 |                                  | 0.792014           |
| probability_temperature=1.4           | 0.8844             | 19 hours, 44 minutes, 14 seconds | 1.112915           |

#### 参数消融测试（不带Transformer）

ESR_selected_ratio_end：采样比例的终止值，0-1。目前来看end=0.5最佳

| 类型                                   | 最优值             | 运行时间                        | 收敛轮次（百万轮） |
| -------------------------------------- | ------------------ | ------------------------------- | ------------------ |
| ESR_selected_ratio_end=0.2             |                    |                                 |                    |
| ESR_selected_ratio_end=0.3             |                    |                                 |                    |
| ESR_selected_ratio_end=0.4             | 0.9906976744186047 | 13 hours, 20 seconds            | 1.864917           |
| ESR_selected_ratio_end=0.5（目前的值） | 0.9953488372093023 | 9hours, 2 minutes, 8 seconds    | 1.664335           |
| ESR_selected_ratio_end=0.6             | 0.9851485148514851 | 13 hours, 2 minutes, 28 seconds | 1.834819           |
| ESR_selected_ratio_end=0.7             | 0.9724770642201835 | 9 hours, 40 minutes, 1 seconds  | 1.87523            |
| ESR_selected_ratio_end=0.8             |                    |                                 |                    |

probability_temperature: 1。方差的放缩值.为1使用原值，>1增大，选择更加随机，<1减小，便于倾向于偏差大的经验。选取1.0最佳

| 类型                                  | 最优值             | 运行时间                         | 收敛轮次（百万轮） |
| ------------------------------------- | ------------------ | -------------------------------- | ------------------ |
| probability_temperature=0.6           | 0.9849246231155779 | 12 hours, 57 minutes, 26 seconds | 1.985558           |
| probability_temperature=0.8           | 0.9814814814814815 |                                  | 1.924852           |
| probability_temperature=1（目前的值） | 0.9953488372093023 | 9hours, 2 minutes, 8 seconds     | 1.664335           |
| probability_temperature=1.2           | 0.9502487562189055 |                                  | 1.865391           |
| probability_temperature=1.4           |                    |                                  |                    |

ESR_warm_up_ratio: 0.6  #热身区间占总训练轮次的比例。在热身区间，selected_radio会从start线性增长到end

| 类型                  | 最优值             | 运行时间                     | 收敛轮次（百万轮） |
| --------------------- | ------------------ | ---------------------------- | ------------------ |
| ESR_warm_up_ratio=0.5 | 0.9817351598173516 |                              | 1.684735           |
| ESR_warm_up_ratio=0.6 | 0.9953488372093023 | 9hours, 2 minutes, 8 seconds | 1.664335           |
| ESR_warm_up_ratio=0.7 | 0.9715639810426541 |                              | 1.885191           |
| ESR_warm_up_ratio=0.8 | 0.9805825242718447 |                              | 1.945252           |
|                       |                    |                              |                    |

## 对照实验设计

运行设备：2080ti或3090.

实验环境：SMACV1,SMACV2,GRF

实验条件：奖励正常+奖励稀疏

### 实验地图选择

| 实验环境 | 地图名                     | 地图难度/简介                    | 运行时间估计 |
| -------- | -------------------------- | -------------------------------- | ------------ |
| SMAC     | 2s3z                       | hard                             | 1-2day       |
| SMAC     | MMM2                       | super hard                       | 3day         |
| SMAC     | 10m_vs_11m                 | super hard                       | 3day         |
| SMAC     | 8m                         | easy                             | 1-2day       |
| GRF      | academy_corner             | 足球比赛的角球罚球场景，11_vs_11 | 3day         |
| GRF      | academy_counterattack_hard | 我方球门的防守反击场景，4_v_2    | 3day         |
| SMACV2   | protoss_10_vs_10           | 神族随机对战，10_vs_10           | 3day         |
| SMACV2   | terran_10_vs_10            | 人族随机对战，10_vs_10           | 3day         |
| SMACV2   | zerg_10_vs_10              | 虫族随机对战，10_vs_10           | 3day         |

在奖励正常和奖励稀疏下各跑一次，所以一共16个曲线图

### 对照算法

1. PER：主要基线，直接对经验做PER。已经实现。已经有SMAC 3s_vs_3z、SMAC 10m_vs_11m、GRF academy_corner、GRF academy_counterattack_hard在奖励正常和奖励稀疏下的数据。

   T. Schaul, J. Quan, I. Antonoglou, and D. Silver, “Prioritized Experi

   ence Replay,” in *4th International Conference on Learning Represen*

   *tations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016*, 2016.

2. DIFFER：主要基线。已实现。已经有SMAC 3s_vs_3z、SMAC 10m_vs_11m、GRF academy_corner、GRF academy_counterattack_hard在奖励正常和奖励稀疏下的数据。

   X. Hu, J. Zhao, W. Zhou, R. Feng, and H. Li, “DIFFER: Decomposing

   Individual Reward for Fair Experience Replay in Multi-Agent Rein

   forcement Learning,” in *Annual Conference on Neural Information*

   *Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA,*

   *December 10 - 16, 2023*, 2023.

3. SUPER：主要基线，已实现。已经有SMAC 3s_vs_3z、SMAC 10m_vs_11m、GRF academy_corner、GRF academy_counterattack_hard在奖励正常和奖励稀疏下的数据。

   M. Gerstgrasser, T. Danino, and S. Keren, “Selectively Sharing Ex

   periences Improves Multi-Agent Reinforcement Learning,” in *Inter*

   *national Conference on Autonomous Agents and Multiagent Systems,*

   *AAMAS 2023, London, United Kingdom, 29 May 2023 - 2 June 2023*,

   2023, pp. 2433–2435.

4. QMIX

5. 我们的方法
6. 2024-CCFA-NIPS Kaleidoscope: Learnable Masks for Heterogeneous Multi-agent Reinforcement Learning**利用个性化掩膜实现多样化参数共享**。找方文浩

   提出kaleidscope，为不同的agent维护一组公共参数和多组可学习的mask来进行参数共享。通过鼓励mask间的差异促进policy的多样性。

   代码：https://github.com/LXXXXR/Kaleidoscope



备选的2024年算法：

1. 2024-CCFA-ICML Individual Contributions as Intrinsic Exploration Scaffolds for Multi-agent Reinforcement Learning **使用个人奖励鼓励探索**
   解决：提出个体贡献奖励的内在探索框架（ICES），在全局角度评估每个agent的贡献来激励探索。ICES使用贝叶斯惊讶（用带两个编码器的CVAE实现）构造探索框架辅助训练。分离探索policy和实际policy，使探索policy可以使用全局信息训练。
   代码：https://github.com/LXXXXR/ICES.

   优势：完整代码，疑似符合PYMARL工程结构。在GRF和SMAC都跑过。看起来是最容易实现的代码

2. 2024-ICLR ATTENTION-GUIDED CONTRASTIVE ROLE REPRESENTATIONS FOR MULTI-AGENT REINFORCEMENT LEARNING **引入对比学习促进信用分配**

   从角色和agent的相关性得到启发，提出ACORM。引入互信息最大化来形式化角色表示学习，推导对比学习目标，并近似负对分布。利用attention机制在全局价值分解中学习角色特征，引导智能体在角色空间中协调来产生信用分配。

   [GitHub - NJU-RL/ACORM](https://github.com/NJU-RL/ACORM)

   优势：在GRF和SMAC都跑过。有完整代码

   劣势：代码跟pymarl不是同一个框架，移植比较费时间

4. 2024-CCFA-ICML LAGMA: LAtent Goal-guided Multi-Agent Reinforcement Learning  **潜在目标引导协作**
   解决：提出潜在目标引导的MARL（LAGMA），在潜在空间中生成目标来达到轨迹，提供一个潜在的目标一道奖励来向参考轨迹过度。使用VQ-VAE进行量化嵌入空间改造，将状态投影到量化向量空间。使用VQ密码本生成到达目标的参考轨迹。使用潜在目标来引导内在奖励的生成。
   代码：https://github.com/aailabkaist/LAGMA

   优势：代码完整，在GRF和SMAC都跑过

   劣势：代码针对每一个他用到的地图设计了专门的VAE超参数，这很可能意味着算法的鲁棒性可能像MASER一样极差，需要对每个地图单独调优。

### 时间估计

PER，DIFFER，SUPER：需要运行4+4，8次

我们的方法：需要运行8+8,16次

新增对照算法：每个需要运行8+8,16次

总共需要运行至少56-72次。按以前的经验，设一个服务器可以同时运行四次，运行一次需要17小时，则总共需要约238h-306h，即9天—14天。统计不包括成功实现PER，成功运行算法和成功移植算法到PYMARL的时间。不包括两个批次运行之间的操作时间和延误时间。

### 实验结果记录

1号：方学长服务器：121.248.201.30（同时跑四个可能暴死进程，所以同时跑三个）

2号：乔学长服务器：121.248.201.11（同时跑3个）

3号：沈学姐服务器：Todesk 774 340 177（可以同时跑3个）

**SMAC-奖励正常**

| 算法         | 2s3z               | MMM2               | 10m_vs_11m         | 8m                 |
| ------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| PER          | 0.9765258215962441 | 在跑-1号           | 0.565833893421164  | 0.867764538559205  |
| SUPER        | 0.9855769230769231 | 在跑-2号           | 0.678294553976223  | 0.924538795684027  |
| DIFFER       | 0.9765             | 0.6341463414634146 | 0.7344632768361582 | 0.9329896907216495 |
| 我们的方法   | 0.9953488372093023 | 0.6972972972972973 | 0.6727272727272727 | 在跑-12服务器      |
| kaleiscope   | 0.9095477386934674 | 在跑-服务器10      | 0.6257668711656442 | 在跑-1号           |
| ices         | 0.9711538461538461 | 0.8920454545454546 | 0.6071428571428571 | 0.9858             |
| qmix（可选） |                    |                    |                    |                    |

**SMAC-奖励稀疏**

| 算法         | 2s3z               | MMM2                | 10m_vs_11m          | 8m                 |
| ------------ | ------------------ | ------------------- | ------------------- | ------------------ |
| PER          | 在跑-3号           | 在跑-3号            | 0.20669463632415203 | 0.8486486486486486 |
| SUPER        | 在跑-2号           | 0.19473684210526315 | 0.602470292252863   | 0.740261437908496  |
| DIFFER       | 0.9509803921568627 | 在跑-2号            | 0.09571788413098237 | 0.8263157894736843 |
| 我们的方法   |                    |                     | 在跑-服务器12       | 在跑-服务器12      |
| kaleiscope   |                    |                     |                     |                    |
| ices         |                    |                     |                     |                    |
| qmix（可选） |                    |                     |                     |                    |

**GRF-奖励正常**

| 算法         | academy_corner     | academy_counterattack_hard |
| ------------ | ------------------ | -------------------------- |
| PER          | 0.509775110253782  | 0.671561295825965          |
| SUPER        | 0.742709902951855  | 0.793875645917204          |
| DIFFER       | 0.2767857142857143 | 0.09433962264150944        |
| 我们的方法   |                    |                            |
| kaleiscope   |                    |                            |
| ices         | 在跑-服务器10      |                            |
| qmix（可选） |                    |                            |

**GRF-奖励稀疏**

| 算法         | academy_corner      | academy_counterattack_hard |
| ------------ | ------------------- | -------------------------- |
| PER          | 0.692256181513049   | 0.181245637552827          |
| SUPER        | 0.713048885163406   | 0.347565307296136          |
| DIFFER       | 0.09722222222222222 | 0.2840909090909091         |
| 我们的方法   |                     |                            |
| kaleiscope   |                     |                            |
| ices         | 0.40625             | 在跑-服务器10              |
| qmix（可选） |                     |                            |

**SMACV2-奖励正常**

| 算法         | terran_10_vs_10 |      | protoss_10_vs_10 | zerg_10_vs_10 |
| ------------ | --------------- | ---- | ---------------- | ------------- |
| PER          |                 |      |                  |               |
| SUPER        |                 |      |                  |               |
| DIFFER       |                 |      |                  |               |
| 我们的方法   |                 |      |                  |               |
| kaleiscope   |                 |      |                  |               |
| ices         |                 |      |                  |               |
| qmix（可选） |                 |      |                  |               |

**SMACV2-奖励稀疏**

| 算法         | terran_10_vs_10 | protoss_10_vs_10 | zerg_10_vs_10 |
| ------------ | --------------- | ---------------- | ------------- |
| PER          |                 |                  |               |
| SUPER        |                 |                  |               |
| DIFFER       |                 |                  |               |
| 我们的方法   |                 |                  |               |
| kaleiscope   |                 |                  |               |
| ices         |                 |                  |               |
| qmix（可选） |                 |                  |               |
