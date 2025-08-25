# INSPIRE: Individualized and Neighbor-based Sharing Prioritized Experience Replay for Multi-Agent Reinforcement Learning

## overview
This repository contains the official implementation of INSPIRE (Individualized and Neighbor-based Sharing Prioritized Experience Replay), a framework for efficient experience exchange in sparse-reward multi-agent reinforcement learning (MARL). INSPIRE is designed to enhance training efficiency and generalization performance under sparse reward conditions by combining:

- Experience decomposition to partition team experiences into individualized replay buffers, reducing irrelevant information.

- Neighbor discovery to restrict communication to local neighborhoods, improving contextual relevance and scalability.

- Neighbor Experience Transmitter (NET) to evaluate and selectively share high-value experiences using feedback from nearby agents.

- Experience receiver filtering to retain only the most informative experiences, mitigating overfitting and noise propagation.

To evaluate effectiveness and robustness, we conduct experiments under sparse-reward settings across SMAC, SMACv2, and GRF benchmarks. Results show that INSPIRE achieves up to 13.06% higher win rates on SMAC, 12.93% on SMACv2, and 37.92% on GRF compared to five state-of-the-art baselines, while converging faster and maintaining superior sample efficiency. Extensive ablation studies further confirm the contribution of each component and the scalability of the framework.

## Instructions

We have retained the running code for INSPIRE and the baseline algorithms. The running modes are as follows:

```shell
python src/main.py --config=[Algorithm name] --env-config=[Env name] with
for example:
(INSPIRE)python src/main.py --config=inspire_qmix --env-config=sc2 with env_args.map_name=2s3z
```

The config files are all located in `src/config/algs`.

## Prerequisites

If you encounter any issues while downloading SMAC, SMACV2, and GRF, you can refer to [PymarlZoo](https://github.com/jnzhang233/PymarlZoo) or refer to the corresponding GitHub repository's issue. Additionally, apart from the basic extension package dependencies, the project can still operate normally even if individual environments are not configured.

Set up StarCraft II and SMAC:

```shell
bash install_sc2.sh
```

Set up Google Research Football: Follow the instructions in [GRF](https://github.com/google-research/football?tab=readme-ov-file#quick-start) .

Set up SMACV2: Follow the instructions in [SMACV2](https://github.com/oxwhirl/smacv2) .

Set up other packages:

```
pip install -r requirements.txt
```

## Acknowledgement

The code is implement based on the following open-source projects:

- [pymarl](https://github.com/oxwhirl/pymarl)
- [pymarl2](https://github.com/hijkzzz/pymarl2)
- [PymarlZoo](https://github.com/jnzhang233/PymarlZoo)
- [pyamrl3](https://github.com/tjuHaoXiaotian/pymarl3)








