# FCV2X-Net

This repo is the official implementation for FCV2X-Net:
FCV2X-Net: Foresighted and Coordinated Vehicle-to-Everything Control for Joint Navigation and Signal Optimization.

<!-- ## Overall architecture
This work aims to construct a prediction framework that predicts high-resolution carbon emissions with open data of satellite images and POI. 
![Overall framework](framework.png) -->


## Data
We conduct experiments on two city-scale datasets. Summary of the datasets are presented:

| Datasets                        | Beijing_25 | Beijing_49 |
|--------------------------------|------------|------------|
| #Intersections                 | 25         | 49         |
| #Traffic signals               | 20         | 42         |
| #Three-phase traffic signals   | 16         | 29         |
| #Four-phase traffic signals    | 4          | 13         |
| #Roads                         | 74         | 152        |
| Covered Area                   | 0.702 km²  | 1.5482 km² |
| #Vehicles                      | 3210       | 4903       |
| #Agents                        | 300        | 500        |

*Table 1: The summary statistics of our datasets.*

<!-- Due to the size limit of github, we have stored the data in an anonymous google drive link: https://drive.google.com/drive/folders/1_HHa5X6nLiB4mHfEIn42jb5fwc64nf0v?usp=sharing. 
[Due to cloud storage limit, we only update the Beijing dataset. All other datasets are available through email requests.] 
Please download them and place them inside /data. -->

## Installation
### Environment
- Tested OS: Linux
- Python >= 3.9
- torch == 2.7.0
- Tensorboard


## Config 
Configs for performance reproductions on all datasets. 


### Beijing_25
```
python main_FCV2X-Net.py --data data/data_25 --cuda_id 0 --training_start 30000 --experience_threshold 400 --gamma 0.995 --reward emission --step_count 3600 --buffer_size 1000000 --dqn_type dqn --update_threshold 30 --exploration_times 6000000 --agg bgcn --junction_training_start 12000 --junction_experience_threshold 300 --intention 1 --start_lr 0.01 --lr 5e-4 --supervised_signal 1 --mean_field 1
```

### Beijing_49
```
python main_FCV2X-Net.py --data data/data_49 --cuda_id 1 --training_start 30000 --experience_threshold 800 --gamma 0.995 --reward emission --step_count 3600 --buffer_size 1000000 --dqn_type dqn --update_threshold 30 --exploration_times 6000000 --agg_type bgcn --intention 1 --start_lr 0.01 --lr 5e-4 --supervised_signal 1 --mean_field 1 --batchsize 512 --basic_update_times 3 --balancing_coef 10
```