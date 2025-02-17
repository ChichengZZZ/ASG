## ASG(LC): Adversarial Scenario Generation for Lane-change 

## Profile 
- This project aims to extract typical critical lane change driving scenarios from real trajectory data (NGSIM+INTERACTION datasets)。
- Generating new adversarial lane changing scenarios from extracted a priori human driving scenarios。
- Generated using GAIL, contains improved methods for PPO, innovative design of reward functions, optimization of policy networks。

## Installation

### Dependent 
```shell
    pip install -r requirements.txt
```

### Highway-env Local Code Installation
```shell
    cd exp/highway
    pip install -e .
```
### stable baseline Local Code Installation

```shell
    cd exp/stable-baselines3-master
    pip install -e .
```

## 1.data processing

### 1.1 Datasets preparation

Link or copy the ngsim dataset with the Interaction dataset to the data/datasets folder, the datasets folder structure is as follows.

```
datasets
|-- Interaction
|   |-- INTERACTION-Dataset-TC-v1_0
|   |   |-- maps
|   |   |   |-- DR_USA_Intersection_MA.osm
|   |   |   `-- DR_USA_Intersection_MA.osm_xy
|   |   `-- recorded_trackfiles
|   |       `-- DR_USA_Intersection_MA
|   |           |-- vehicle_tracks_000.csv
|   |           |-- vehicle_tracks_001.csv
|   |           |-- ...
|   |           `-- vehicle_tracks_021.csv
|   |-- maps
|   |-- recorded_trackfiles
|   |   |-- DR_CHN_Merging_ZS
|   |   |-- DR_CHN_Roundabout_LN
|   |   |-- DR_DEU_Merging_MT
|   |   |-- DR_DEU_Roundabout_OF
|   |   |-- DR_USA_Intersection_EP0
|   |   |-- DR_USA_Intersection_EP1
|   |   |-- DR_USA_Intersection_GL
|   |   |-- DR_USA_Intersection_MA
|   |   |-- DR_USA_Roundabout_EP
|   |   |-- DR_USA_Roundabout_FT
|   |   |-- DR_USA_Roundabout_SR
|   |   `-- TC_BGR_Intersection_VA
|   |-- INTERACTION-Dataset-DR-v1_1.rar
|   `-- INTERACTION-Dataset-TC-v1_0.zip
`-- ngsim.csv
```


### 1.2 Scene Extraction
#### highway
```shell
    cd NGSIM_env/data
    python3 data_process.py
```

#### urban intersection

```shell
    cd NGSIM_env/data
    python3 interaction.py
```

### 1.3 Expert Trajectory Extraction
#### highway
```shell
    cd examples/data_process
    python3 naturalistic_ngsim.py
```

#### urban intersection

```shell
    cd examples/data_process
    python3 naturalistic_interaction.py
```


## 2.Natural traffic flow generation
_Parameter description_
- `--policy`: str
- `--scenario`: str, Scene types, highway / urban intersection scene.
- `--model`: str, Model name, optional: diffusion-diffusion model, gail-generative adversarial imitation learning model
- `--num-threads`: int, The total number of processes used for testing or training.
- `--num-epochs`: int, Total number of training rounds.
- `--exp-name`: str, Experiment name for multiple experiments.
- `--load-model-id`: int, Model id loaded during testing, id corresponds to epoch during training.
- `--epochs`: int, Total number of randomly selected scenarios at the time of testing.
- `--show`: Optional, rendered or not
- `--save-video`: Optional, save or not
- `--resume`: Optional, Retraining from breakpoints prevents retraining due to power failure, etc.

### 2.1 Modeling Human Driving Strategies

#### 2.1.1 highway
```shell
    cd examples/av2_model_train
    python3 train.py --policy human_driving_policy --scenario highway --model diffusion --num-epochs 1000 --num-threads 6 --exp_name exp_1 [--resume]
                                                                              gail
                                                                              ppo
```

#### 2.1.2 urban intersection
```shell
    cd examples/av2_model_train
    python3 train.py --policy human_driving_policy --scenario intersection --model diffusion --num-epochs 1000 --num-threads 6 --exp_name exp_1 [--resume]
                                                                                   gail
                                                                                   ppo
```

### 2.2 Natural traffic flow generation

#### 2.2.1 highway
```shell
    cd examples/av2_model_test
    python3 test.py --policy human_driving_policy --scenario highway --model diffusion --load-model-id 400 --num-threads 6 --exp_name exp_1 --epochs 1000 [--show] [--save-video]
                                                                             gail
                                                                             ppo
```

#### 2.2.2 urban intersection
```shell
    cd examples/av2_model_test
    python3 test.py --policy human_driving_policy --scenario intersection --model diffusion --load-model-id 400 --num-threads 6 --exp_name exp_1 --epochs 1000 [--show] [--save-video]
                                                                                  gail
                                                                                  ppo
```

## 3.Natural Adversarial Test Scenario Generation

### 3.1 Modeling Natural Adversarial Strategies

### 3.1.1 highway
```shell
    cd examples/av2_model_train
    python3 train.py --policy natural_adversarial_model --scenario highway --model ppo --num-epochs 1000 --num-threads 6 --exp_name exp_1 --supervise-model diffusion --supervise-model-id 3700 [--resume]
                                                                                   trpo                                                                      gail
                                                                                   a2c                                                    
```

### 3.1.2 urban intersection
```shell
    cd examples/av2_model_train
    python3 train.py --policy natural_adversarial_model --scenario intersection --model ppo --num-epochs 1000 --num-threads 6 --exp_name exp_1 --supervise-model diffusion --supervise-model-id 4899 [--resume]
                                                                                        trpo                                                                      gail 
                                                                                        a2c 
```

### 3.2 Natural Adversarial Test Scenario Generation

### 3.2.1 highway 
```shell
    cd examples/av2_model_test
    python3 train.py --policy natural_adversarial_model --scenario highway --model ppo --load-model-id 400 --num-threads 6 --exp_name exp_1 --supervise-model diffusion --supervise-model-id 3700 [--show] [--save-video]
                                                                                   trpo                                                                       gail
                                                                                   a2c 
```

### 3.2.2 urban intersection
```shell
    cd examples/av2_model_test
    python3 train.py --policy natural_adversarial_model --scenario intersection --model ppo --load-model-id 400 --num-threads 6 --exp_name exp_1 --supervise-model diffusion --supervise-model-id 3700 [--show] [--save-video]
                                                                                        trpo                                                                       gail
                                                                                        a2c 
```


## Result 