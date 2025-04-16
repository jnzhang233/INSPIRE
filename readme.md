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

1. src/envs/smac_v1/official/starcraft2.py写入：

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

   

## 运行指令：

目前在12服务器运行。

differ_qmix版本，用的episode_runner:python src/main.py --config=differ_qmix --env-config=sc2 with env_args.map_name=2s3z t_max=20000



运行inspire_qmix:

python src/main.py --config=inspire_qmix --env-config=sc2 with env_args.map_name=2s3z t_max=3005000



测试项目：

1. differ_qmix_parallel版本，用的parallel_runner，可能更适合SMAC环境:python src/main.py --config=differ_qmix_parallel --env-config=sc2 with env_args.map_name=2s3z t_max=20000

   目前已被放弃使用，效果不好，时间倒是挺快。大概得10005000轮次才能稳定训练完成感觉

2. transformer做的agent网络：python src/main.py --config=differ_qmix_transformertest --env-config=sc2 with env_args.map_name=2s3z t_max=20000

   transformer层消融：python src/main.py --config=differ_qmix_transformertest --env-config=sc2 with env_args.map_name=2s3z t_max=3005000 transformer_n_layers=1

   transformer头消融：python src/main.py --config=differ_qmix_transformertest --env-config=sc2 with env_args.map_name=2s3z t_max=3005000 transformer_n_head=1

   正在测试，反正能跑？



## 改进思路

1一定要做。2可以简化，但是还用视野范围随便选的话可能被怼。4相对简单，高优先级做，可以找沈学姐。3相对简单，可以考虑做，可以找乔学长。

做到哪里算哪里

1（transformer做agent网络，生成embedding并分享）>3（per采样）>2（做视野范围（考虑引入CommomNet，Tarmmac））>4（做基于互信息的公式）.





1. 每个智能体对经验批次过transformer生成固定批次的embedding，然后分享embedding。接收端接收embedding后解压缩成新经验。新的经验可以更适合目前智能体的经验。可能可以提取和构建新的信息来帮助智能体的经验。对自己的经验可以直接用。可以削减通信频次。

   目前来看是最不靠谱的改进思路，建议回炉。

   **存在的问题：**

   1. 在工程实践中，智能体实际分享的是TD-ERROR而不是整个经验，而且一次训练用的是batch_size个轨迹的batch，并不是实时交互和分享的。
   2. 固定批次的embedding应对多样化的环境必然出现问题。需要根据agent数目和经验批次规模来建立一个计算生成embedding数目的公式
   3. 目前agent网络的输入是整个显式数据构成的episode_batch的每一步。而且input_size是锁死的，同时接收embedding和state很可能造成训练的混乱。直接使用embedding进行训练，也会导致训练的结果不能直接用于与环境的交互中，这是本末倒置的。
   4. 接收embedding后解压缩为新的经验，可行性很低。而且训练工程中，所有agent的数据都在一个批次内，真有这个需要直接把完整的数据送过去就好了。

   可能的实现方式：

   1. 构建一个Transformer做agent网络，自己的经验走全过程，别人的embedding从attention层走。

2. 只和视野范围内的共享。采样一些批次计算是不是在视野范围内。视野范围需要参数消融。可以考虑mean_shift。可以考虑改成用ROMA的角色聚类来实现共享对象的选择。可以看看谢老师发的局部观测通信方面的文章找找灵感。

   存在的问题:

   1. 目前经验的batch中不包括真正意义的智能体[x,y]坐标，所以需要另外更改实验环境才可以实现。
   2. 一次训练用的是batch_size个轨迹的batch，并不是实时交互和分享的。所以实现这个可能需要对要分享的经验逐一计算位置是否在范围内。可以试试L1距离和L2距离。
   3. 不在视野范围但是所处OBS相似的，也应该进行分享，但是本文并没有体现。

   如果用视野范围：

   1. 如果视野范围没有人怎么办。是略过还是扩大范围继续搜索。
   2. 视野范围的消融怎么做。

   如果用ROMA：

   1. 虽然代码好用，但是论述可能比较复杂

   如果用CommomNet，Tarmmac之类的：

   1. 调研一下,能不能直接用。也是视野范围基础上做的东西。可以问乔学长。

   如果用GCN做邻居选择：

   1. 目前不考虑？

3. 选取分享经验的机制。

   学长版本的机制是计算TD-ERROR是不是自己这个批次的极端值（正态分布的边缘值）并只分享极端值，这个是SUPER给出的一种策略。

   存在的问题：

   1. 我们可以考虑训练一个网络来为经验打分，判断经验是否值得分享。但是具体网络怎么设置还有待商榷。
   1. 或许可以仿照PER的采样机制，让极端值有高概率被分享，但是一般的值也有一定概率被分享。被分享的概率取决于值的概率。
   3. 是否需要对轨迹做PER

3. 共享的频次可以通过经验采样的变化熵来判断，熵越大频次越高。

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

1. 新增了config/algs/inspire_qmix.yaml。新增了src/learners/qmix_newdiffer_test.py。并在src/learners/_init_.py加入对应导入代码。

2. 开始尝试把本科毕设优化过的经验分享、经验接收代码调整一下，加入newdiffer

   a. 完成了上下限计算和经验分享的code
   
   b. 完成了经验接收的code

### stage2:尝试设计一个基于transformer的agent网络：

1. 添加了src/modules/agents/trnasformer_agent.py。并在init.py添加了导入语句

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

{'DIFFER(baseline)': 0.9861111111111112。DIFFER基线。

'DIFFER_Transformer': 0.9852216748768472。跑错了，这个还是DIFFER基线。

'INSPIRE_ESR_using_normal_distribution': 0.9863013698630136。基于正态分布的经验选择与分享算法+DIFFER效果。

目前在跑：

加了transformer的differ。transformer的transformer_n_head=1版本，transformer的transformer_n_layers=1版本
