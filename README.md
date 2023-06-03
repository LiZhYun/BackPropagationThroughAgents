# BPTA

Zhiyuan Li*. 

This repository implements Back-Propagatioin Through Agents(BPTA). The implementation in this repositorory is used in the paper "Back-Propagatioin Through Agents" (https://arxiv.org/abs/). This repository is heavily based on https://github.com/marlbenchmark/on-policy.

<font color="red"> All hyperparameters and training curves are reported in appendix, we would strongly suggest to double check the important factors before runing the code, such as the rollout threads, episode length, ppo epoch, mini-batches, clip term and so on.
 

## Environments supported:

- [Go-Bigger](https://github.com/opendilab/GoBigger)
- [Hanabi](https://github.com/deepmind/hanabi-learning-environment)
- [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai)
- [Google Research Football (GRF)](https://github.com/google-research/football)


## Compared Algorithms supported:

- [MAPPO](https://github.com/marlbenchmark/on-policy)
- [HAPPO](https://github.com/cyanrain7/TRPO-in-MARL)
- [Auto-Regressive MAPPO(AR-MAPPO)](https://github.com/garrett4wade/revisiting_marl)
- [Graph-Based Coordination Strategy(GCS)](https://github.com/Amanda-1997/GCS_aamas337)


## 1. Usage

All core code is located within the bta folder. The algorithms/ subfolder contains algorithm-specific code
for BPTA. 

* The envs/ subfolder contains environment wrapper implementations for the Go-Bigger, GRF, Overcooked-AI, and Hanabi. 

* Code to perform training rollouts and policy updates are contained within the runner/Compared-Algorithms/ folder - there is a runner for 
each environment. 

* Executable scripts for training with default hyperparameters can be found in the scripts/Compared-Algorithms/ folder. The files are named
in the following manner: train_algo_environment.sh. Within each file, the scenario/layout name can be altered. 
* Python training scripts for each environment can be found in the scripts/train/ folder. 

* The config.py file contains relevant hyperparameter and env settings. Most hyperparameters are defaulted to the ones
used in the paper; however, please refer to the appendix for a full list of hyperparameters used. 


## 2. Installation

 Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/). We remark that this repo. does not depend on a specific CUDA version, feel free to use any CUDA version suitable on your own computer.

``` Bash
# create conda environment
conda create -n bpta python=3.8
conda activate bpta
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

```
# install bpta package
cd BackPropagationThroughAgents
pip install -e .
```

Even though we provide requirement.txt, it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### 2.1 Go-Bigger

   

``` Bash
pip install gobigger
```

* We referred to [GoBigger-Explore](https://github.com/opendilab/Gobigger-Explore)'s handling of observations.


### 2.2 Hanabi
Environment code for Hanabi is developed from the open-source environment code, but has been slightly modified to fit the algorithms used here.  
To install, execute the following:
``` Bash
pip install cffi
cd envs/hanabi
mkdir build & cd build
cmake ..
make -j
```
Here are all hanabi [models](https://drive.google.com/drive/folders/1RIcP_rG9NY9UzaWfFsIncDcjASk5h4Nx?usp=sharing).

### 2.3 Overcooked-AI

``` Bash
pip install overcooked-ai
```

Our Overcooked experiments are conducted in five layouts from Learning Zero-Shot Cooperation with Humans, Assuming Humans Are Biased, named Asymmetric Advantages, Coordination Ring, Counter Circuit, Distant Tomato and Many Orders. These layouts are named "unident_s", "random1", "random3", "distant_tomato" and "many_orders" respectively in the code.

### 2.4 GRF

Please see the [football](https://github.com/google-research/football/blob/master/README.md) repository to install the football environment.

## 3.Train
Here we use train_football_3v1.sh as an example:
```
cd bta/scripts/temporal/football
chmod +x ./train_football_3v1.sh
./train_football_3v1.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 


## 4. Publication

If you find this repository useful, please cite our [paper](https://arxiv.org/abs/):
```
@inproceedings{
Back-Propagatioin,
title={Back-Propagatioin Through Agents},
author={Zhiyuan Li},
booktitle={},
year={2024}
}
```

