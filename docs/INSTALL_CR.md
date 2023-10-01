# Installation

This doc provides instructions to get started.

## Install CARLA

First, you have to install carla. (*The installation script has been taken from [PlanT](https://github.com/autonomousvision/plant)*)

``` bash
# 1. Clone this repository
git clone --recurse-submodules git@github.com:pranaygupta36/LAV.git
cd LAV

# 2. Setup Carla
chmod +x setup_carla.sh
./setup_carla.sh
```

## Install dependencies
* First, inside the repo, create a dedicated conda environment. Refer [here](https://www.anaconda.com/products/individual#Downloads) if you do not have conda. 

```
conda env create -f environment.yaml 
```

* Install [PyTorch](https://pytorch.org/get-started/locally/)
* Install [torch-scatter](https://github.com/rusty1s/pytorch_scatter) based on your `CUDA` and `PyTorch` versions.
* Setup [wandb](https://docs.wandb.ai/quickstart)

## Configure environment variables

* Update CARLA_ROOT ([here](setup_env_vars.sh#L10)) and LAV_ROOT ([here](setup_env_vars.sh#L19)) variables.

``` bash
chmod +x setup_env_vars.sh
./setup_env_vars.sh
```

* Download the LAV model weights from [here](https://cmu.box.com/s/09cda83xufxqfygat2hmm0e09u15tlpl)
``` bash
cp <path_to_weights.zip> <path_to_LAV>
unzip weights.zip
```

Now, you can treat this conda environment the dedicated one (by defauld named `CR-LAV-env`). 
