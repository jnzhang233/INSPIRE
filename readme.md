# INSPIRE: Individualized and Shared Prioritized Experience Replay for Sparse Reward Multi-agent Reinforcement Learning

## Acknowledgement

The code is implement based on the following open-source projects:

- [pymarl](https://github.com/oxwhirl/pymarl)
- [pymarl2](https://github.com/hijkzzz/pymarl2)
- [PymarlZoo](https://github.com/jnzhang233/PymarlZoo)
- [pyamrl3](https://github.com/tjuHaoXiaotian/pymarl3)

## Installation instructions

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



## Run an experiment 

We have retained the running code for INSPIRE and the baseline algorithms. The running modes are as follows:

```shell
python src/main.py --config=[Algorithm name] --env-config=[Env name] with
for example:
(INSPIRE)python src/main.py --config=inspire_qmix --env-config=sc2 with env_args.map_name=2s3z
(Kaleiscope)python src/main.py --config=kalei/Kalei_qmix_rnn --env-config=sc2 with env_args.map_name=2s3z
(DIFFER)python src/main.py --config=differ/differ_qmix --env-config=sc2 with env_args.map_name=2s3z
(SUPER)python src/main.py --config=super_qmix --env-config=sc2 with env_args.map_name=2s3z
(PER)python src/main.py --config=per_qmix --env-config=sc2 with env_args.map_name=2s3z
```

The config files are all located in `src/config/algs`.


