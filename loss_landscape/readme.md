This is the starter code for homework 1 (visualizing loss landscape) and is
heavily based on https://github.com/tomgoldstein/loss-landscape.

## Setup

- See `install.sh` or `requirements.txt` for required packages.
- Tested with python3.8

## How to use?

- **Training the models**: Use `train.py` to train the model. Currently, you can
  train resnets on CIFAR 10. The usage is below.
    - Notes:
        - Use `mode` to provide if you want to test, train or both.
        - Use `statefile` to init with some set of weights (or load a pretrained
          model).
        - `remove_skip_connections` will eliminate all skip connections.
        - `skip_bn_bias` removes batch norm and bias from list of parameters
          when flattening the params/grads for projection or computing
          directions. This was used for frequent directions algorithm which is
          used to compute streaming SVD of all the gradients. Li et al. (2018)
          do not consider bias and batch norm parameters in their work.

```
usage: train.py [-h] [-D] [--seed SEED] 
                [--device DEVICE] 
                --result_folder RESULT_FOLDER 
                [--mode {test,train} [{test,train} ...]] 
                --statefile STATEFILE 
                --model {resnet20,resnet32,resnet44,resnet56} 
                [--remove_skip_connections]
                [--batch_size BATCH_SIZE] 
                [--save_strategy {epoch,init} [{epoch,init} ...]] 
                [--skip_bn_bias]
                
example:  
python train.py \
      --result_folder "results/resnet56_skip_bn_bias_remove_skip_connections/" \
      --device cuda:3 --model resnet56 \
      --skip_bn_bias -D --remove_skip_connections 
```

- **Creating direction for projection**: We need two directions (vectors) on
  which we project the project weights for different visualizations. We can
  create directions in different ways and we provide following ways:
    1. Random directions
    2. Principle vector of {w_final-w_i} where i are models saved at end of each
       epoch.
    3. (Approximate) SVD of gradients during training (computed using frequent
       direction algorithm)
        1. Gradient during all training
        2. last 10 epoch
        3. last epoch

  3 are computed during training, for 1 and 2 use `create_directions.py`

```commandline
usage example:

- creating random directions with filter normalization (the checkpoint weights are used for normalization).

python create_directions.py --statefile results/resnet20_skip_bn_bias/ckpt/200_model.pt \
    -r results/resnet20_skip_bn_bias/ --skip_bn_bias --direction_file random_directions.npz \
    --direction_style "random"  --model resnet20

- for pca direction (the statefile folder is folder of all checkpoints)
python create_directions.py \
    --statefile_folder results/resnet20_skip_bn_bias/ckpt/ \
    -r results/resnet20_skip_bn_bias --skip_bn_bias \
    --direction_file pca_directions.npz --direction_style "pca" \
    --model resnet20

```

- **Computing Optimization Trajectories**:

```commandline
python compute_trajectory.py -r results/resnet20_skip_bn_bias_remove_skip_connections/trajectories \
  --direction_file results/resnet20_skip_bn_bias_remove_skip_connections/pca_directions.npz \
  --projection_file pca_dir_proj.npz --model resnet20  --remove_skip_connections \
  -s results/resnet20_skip_bn_bias_remove_skip_connections/ckpt --skip_bn_bias \
 ```

This takes a folder of checkpoints (-s argument) and computes the projection of
$w_i-w_final$ on the direction vectors. The results are saved to the projection
file.

- **Computing loss landscapes of final models**

```commandline
python compute_loss_surface.py \
    --result_folder results/resnet20_skip_bn_bias_remove_skip_connections/loss_surface/  \
    -s results/resnet20_skip_bn_bias_remove_skip_connections/ckpt/200_model.pt \
    --batch_size 1000 --skip_bn_bias \
    --model resnet20 --remove_skip_connections \
    --direction_file results/resnet20_skip_bn_bias_remove_skip_connections/pca_directions.npz \
    --surface_file pca_dir_loss_surface.npz --device cuda:0 \
    --xcoords 51:-10:40 --ycoords 51:-10:40  
```

- **Plotting results**:
    - You can pass either trajectory file or surface file or both in the command
      below.

```
python plot.py --result_folder figures/resnet56/ \
    --trajectory_file results/resnet56_skip_bn_bias/trajectories/pca_dir_proj.npz \
    --surface_file results/resnet56_skip_bn_bias/loss_surface/pca_dir_loss_surface.npz \
    --plot_prefix resnet56_pca_dir
```

Note: The code should be executable with loss-landscape as the root folder. 

##### ADDED LATER


## Train Model
python train.py --result_folder "results/resnet20_skip_bn_bias/" --device cuda:0 --model resnet20 --skip_bn_bias -D --batch_size 128  --num_epochs 200
python train.py --result_folder "results/resnet20_skip_bn_bias_swag/" --device cuda:0 --model resnet20 --skip_bn_bias -D --batch_size 128  --num_epochs 200 --save_swag_model  
python train.py --result_folder "results/resnet20_skip_bn_bias_swag_diag/" --device cuda:0 --model resnet20 --skip_bn_bias -D --batch_size 128  --num_epochs 200 --save_swag_model --use_swag_diag_cov 

python train.py --result_folder "results/resnet20_adversarial_skip_bn_bias/" --device cuda:0 --model resnet20 --skip_bn_bias -D --batch_size 128 --num_epochs 200 --attack_type pgd
python train.py --result_folder "results/resnet20_adversarial_skip_bn_bias_swag/" --device cuda:0 --model resnet20 --skip_bn_bias -D --batch_size 128 --num_epochs 200 --attack_type pgd --save_swag_model  
python train.py --result_folder "results/resnet20_adversarial_skip_bn_bias_swag_diag/" --device cuda:0 --model resnet20 --skip_bn_bias -D --batch_size 128 --num_epochs 200 --attack_type pgd --save_swag_model --use_swag_diag_cov
 
##### Test Model #####

# Test Deterministic, Robust, Adversarial Train - Test (remove --train_data)
python test.py --result_folder results/resnet20_adversarial_skip_bn_bias_swag --device cuda:0 --model resnet20 --skip_bn_bias --ckpt_load 200 --data_root_dir results/resnet20_adversarial_skip_bn_bias_swag/adversarial_data --use_adversarial_saved_data --batch_size 1000 --train_data 

# Test Deterministic, Robust, Vanilla Train - Test (remove --train_data)
python test.py --result_folder results/resnet20_adversarial_skip_bn_bias_swag --device cuda:0 --model resnet20 --skip_bn_bias --ckpt_load 200 --batch_size 1000 --train_data 

# Test Deterministic, Vanilla, Adversarial Train - Test (remove --train_data)
python test.py --result_folder results/resnet20_skip_bn_bias_swag --device cuda:0 --model resnet20 --skip_bn_bias --ckpt_load 200 --data_root_dir results/resnet20_skip_bn_bias_swag/adversarial_data_from_last_ckpt_of_model --use_adversarial_saved_data --batch_size 1000 --train_data 

# Test Deterministic, Vanilla, Vanilla Train - Test (remove --train_data)
python test.py --result_folder results/resnet20_skip_bn_bias_swag --device cuda:0 --model resnet20 --skip_bn_bias --ckpt_load 200 --batch_size 1000 --train_data 

# Test SWAG, Robust, Adversarial Train - Test (remove --train_data)
python test.py --result_folder results/resnet20_adversarial_skip_bn_bias_swag --device cuda:0 --model resnet20 --skip_bn_bias --ckpt_load 200 --use_swag_model --data_root_dir results/resnet20_adversarial_skip_bn_bias_swag/adversarial_data --use_adversarial_saved_data --batch_size 1000 --train_data 

# Test SWAG, Robust, Vanilla Train - Test (remove --train_data)
python test.py --result_folder results/resnet20_adversarial_skip_bn_bias_swag --device cuda:0 --model resnet20 --skip_bn_bias --ckpt_load 200 --use_swag_model --batch_size 1000 --train_data 

# Test SWAG, Vanilla, Adversarial Train - Test (remove --train_data)
python test.py --result_folder results/resnet20_skip_bn_bias_swag --device cuda:0 --model resnet20 --skip_bn_bias --ckpt_load 200 --use_swag_model --data_root_dir results/resnet20_skip_bn_bias_swag/adversarial_data_from_last_ckpt_of_model --use_adversarial_saved_data --batch_size 1000 --train_data 

# Test SWAG, Vanilla, Vanilla Train - Test (remove --train_data)
python test.py --result_folder results/resnet20_skip_bn_bias_swag --device cuda:0 --model resnet20 --skip_bn_bias --ckpt_load 200 --use_swag_model --batch_size 1000 --train_data 

robust-adversarial-swag-train: 0.74
robust-adversarial-swag-test: 0.69

robust-adversarial-train: 0.77
robust-adversarial-test: 0.65


## Compute trajectory
python compute_trajectory.py -r results/resnet20_skip_bn_bias_swag_diag/trajectories --direction_file results/resnet20_skip_bn_bias_swag_diag/buffer.npy.npz --projection_file buffer_proj.npz --model resnet20  -s results/resnet20_skip_bn_bias_swag_diag/ckpt --skip_bn_bias

## Compute Loss surface
python compute_loss_surface.py --result_folder results_final/resnet20_skip_bn_bias/loss_surface/  -s results_final/resnet20_skip_bn_bias/ckpt/200_model.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results_final/resnet20_skip_bn_bias/buffer.npy.npz --device cuda:0  --xcoords 21:-2:2 --ycoords 21:-2:2 --attack_type loaded_pgd --attack_eps 0.05 --attack_alpha 0.05 --attack_iters 20 

python compute_loss_surface.py --model_folder results_final/resnet20_skip_bn_bias_swag_diag/  -s results_final/resnet20_skip_bn_bias_swag_diag/swag_ckpt/200_swag_model_diag.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results_final/resnet20_skip_bn_bias_swag_diag/buffer.npy.npz --device cuda:0  --xcoords 21:-2:2 --ycoords 21:-2:2 --use_swag_model --use_swag_diag_cov --swag_scale 1 --swag_num_samples 30

python compute_loss_surface.py --result_folder results_final/resnet20_skip_bn_bias_swag_diag/loss_surface/  -s results_final/resnet20_skip_bn_bias_swag_diag/swag_ckpt/200_swag_model_diag.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results_final/resnet20_skip_bn_bias_swag_diag/buffer.npy.npz --device cuda:0  --xcoords 21:-2:2 --ycoords 21:-2:2 --attack_type loaded_pgd --attack_eps 0.05 --attack_alpha 0.05 --attack_iters 20 --use_swag_model --use_swag_diag_cov --swag_scale 1 --swag_num_samples 30



## Plot --trajectory_file results_final/resnet20_adversarial_skip_bn_bias_pgd/trajectories/buffer_proj.npz
python plot.py --result_folder results_final/resnet20_skip_bn_bias/figures/  --surface_file results_final/resnet20_skip_bn_bias/loss_surface/buffer_loss_surface_test_loaded_pgd_eps5e-02_alpha5e-02_iters2e+01_21,-2,2.npz --plot_prefix resnet20_freq_dir_test_loaded_pgd_eps5e-02_alpha5e-02_iters2e+01_21,-2,2


# vanilla
python create_directions.py --statefile_folder results/resnet20_skip_bn_bias/ckpt/ -r results/resnet20_skip_bn_bias --skip_bn_bias --direction_file pca_directions.npz --direction_style "pca" --model resnet20
  
python compute_trajectory.py -r results/resnet20_skip_bn_bias/trajectories --direction_file results/resnet20_skip_bn_bias/pca_directions.npz --projection_file pca_proj.npz --model resnet20  -s results/resnet20_skip_bn_bias/ckpt --skip_bn_bias

python compute_loss_surface.py --model_folder results_final/resnet20_skip_bn_bias  --adversarial_folder results_final/resnet20_skip_bn_bias -s results_final/resnet20_skip_bn_bias/ckpt/200_model.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results_final/resnet20_skip_bn_bias/pca_directions.npz --device cuda:0  --xcoords 50:-30:45 --ycoords 50:-30:45 --use_attack

python compute_loss_surface.py --model_folder results_final/resnet20_skip_bn_bias  --adversarial_folder results_final/resnet20_adversarial_skip_bn_bias_pgd -s results_final/resnet20_skip_bn_bias/ckpt/200_model.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results_final/resnet20_skip_bn_bias/pca_directions.npz --device cuda:0  --xcoords 50:-30:45 --ycoords 50:-30:45 --use_attack

python plot.py --result_folder results_final/resnet20_skip_bn_bias/figures_final/  --surface_file results_final/resnet20_skip_bn_bias/loss_surface/pca_loss_surface_train_adversarial_50,-30,45_cross.npz --plot_prefix resnet20_pca_dir_train_adversarial_50,-30,45_trajectory_cross --trajectory_file results_final/resnet20_skip_bn_bias/trajectories/pca_proj.npz

python plot_3d.py --result_folder results_final/resnet20_skip_bn_bias/figures_final_linear_color_3d/  --surface_file results_final/resnet20_skip_bn_bias/loss_surface/pca_loss_surface_train_vanilla_50,-30,45.npz --plot_prefix resnet20_3d_pca_dir_train_vanilla_50,-30,45_trajectory --trajectory_file results_final/resnet20_skip_bn_bias/trajectories/pca_proj.npz --x_uplim 45 --x_lowlim -30 --y_uplim 45 --y_lowlim -30 --zlim 15


# adversarial
python compute_trajectory.py -r results_final/resnet20_adversarial_skip_bn_bias_pgd/trajectories --direction_file results_final/resnet20_adversarial_skip_bn_bias_pgd/pca_directions.npz --projection_file pca_proj.npz --model resnet20  -s results_final/resnet20_adversarial_skip_bn_bias_pgd/ckpt --skip_bn_bias

python compute_loss_surface.py --model_folder results_final/resnet20_adversarial_skip_bn_bias_pgd  --adversarial_folder results_final/resnet20_adversarial_skip_bn_bias_pgd -s results_final/resnet20_adversarial_skip_bn_bias_pgd/ckpt/200_model.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results_final/resnet20_adversarial_skip_bn_bias_pgd/pca_directions.npz --device cuda:0  --xcoords 50:-30:45 --ycoords 50:-30:45 --use_attack

python compute_loss_surface.py --model_folder results_final/resnet20_adversarial_skip_bn_bias_pgd  --adversarial_folder results_final/resnet20_skip_bn_bias -s results_final/resnet20_adversarial_skip_bn_bias_pgd/ckpt/200_model.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results_final/resnet20_adversarial_skip_bn_bias_pgd/pca_directions.npz --device cuda:0  --xcoords 50:-30:45 --ycoords 50:-30:45 --use_attack

python compute_loss_surface.py --model_folder results_final/resnet20_adversarial_skip_bn_bias_pgd  --adversarial_folder results_final/resnet20_adversarial_skip_bn_bias_pgd -s results_final/resnet20_adversarial_skip_bn_bias_pgd/ckpt/200_model.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results_final/resnet20_adversarial_skip_bn_bias_pgd/pca_directions.npz --device cuda:0  --xcoords 50:-30:45 --ycoords 50:-30:45 

python plot.py --result_folder results_final/resnet20_adversarial_skip_bn_bias_pgd/figures_final/  --surface_file results_final/resnet20_adversarial_skip_bn_bias_pgd/loss_surface/pca_loss_surface_train_adversarial_50,-30,45_cross.npz --plot_prefix resnet20_pca_dir_train_adversarial_50,-30,45_trajectory_cross --trajectory_file results_final/resnet20_adversarial_skip_bn_bias_pgd/trajectories/pca_proj.npz

python plot_3d.py --result_folder results_final/resnet20_adversarial_skip_bn_bias_pgd/figures_final_linear_color_3d/  --surface_file results_final/resnet20_adversarial_skip_bn_bias_pgd/loss_surface/pca_loss_surface_train_vanilla_50,-30,45.npz --plot_prefix resnet20_3d_pca_dir_train_vanilla_50,-30,45_trajectory --trajectory_file results_final/resnet20_adversarial_skip_bn_bias_pgd/trajectories/pca_proj.npz --x_uplim 45 --x_lowlim -30 --y_uplim 45 --y_lowlim -30 --zlim 15

# vanilla (swag)
python create_directions.py --statefile_folder results/resnet20_skip_bn_bias_swag/ckpt/ -r results/resnet20_skip_bn_bias_swag --skip_bn_bias --direction_file pca_directions.npz --direction_style "pca" --model resnet20
  
python compute_trajectory.py -r results/resnet20_skip_bn_bias_swag/trajectories --direction_file results/resnet20_skip_bn_bias_swag/pca_directions.npz --projection_file pca_proj.npz --model resnet20  -s results/resnet20_skip_bn_bias_swag/ckpt --skip_bn_bias

python compute_loss_surface.py --model_folder results/resnet20_skip_bn_bias_swag -s results/resnet20_skip_bn_bias_swag/ckpt/200_model.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results/resnet20_skip_bn_bias_swag/pca_directions.npz --device cuda:0  --xcoords 50:-30:45 --ycoords 50:-30:45 --use_attack --data_root_dir results/resnet20_skip_bn_bias_swag/adversarial_data

python compute_loss_surface.py --model_folder results/resnet20_skip_bn_bias_swag -s results/resnet20_skip_bn_bias_swag/ckpt/200_model.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results/resnet20_skip_bn_bias_swag/pca_directions.npz --device cuda:0  --xcoords 50:-30:45 --ycoords 50:-30:45 

python plot.py --result_folder results/resnet20_skip_bn_bias_swag/figures/  --surface_file results/resnet20_skip_bn_bias_swag/loss_surface/pca_loss_surface_train_adversarial_50,-30,45.npz --plot_prefix resnet20_pca_loss_surface_train_adversarial_50,-30,45 --trajectory_file results/resnet20_skip_bn_bias_swag/trajectories/pca_proj.npz

python plot.py --result_folder results/resnet20_skip_bn_bias_swag/figures/  --surface_file results/resnet20_skip_bn_bias_swag/loss_surface/pca_loss_surface_train_adversarial_50,-30,45_swag_30ns_1.0e+00s_1.0e-05vc.npz --plot_prefix resnet20_pca_loss_surface_train_adversarial_50,-30,45_swag_30ns_1.0e+00s_1.0e-05vc --trajectory_file results/resnet20_skip_bn_bias_swag/trajectories/pca_proj.npz

python plot_3d.py --result_folder results/resnet20_skip_bn_bias_swag/figures_linear_color_3d  --surface_file results/resnet20_skip_bn_bias_swag/loss_surface/pca_loss_surface_train_adversarial_50,-30,45.npz --plot_prefix resnet20_3d_pca_loss_surface_train_adversarial_50,-30,45 --trajectory_file results/resnet20_skip_bn_bias_swag/trajectories/pca_proj.npz --x_uplim 45 --x_lowlim -30 --y_uplim 45 --y_lowlim -30 --zlim 10 

# vanilla 
python create_directions.py --statefile results/resnet20_skip_bn_bias_swag_diag/ckpt/200_model.pt -r results/resnet20_skip_bn_bias_swag_diag --skip_bn_bias --direction_file random_directions.npz --direction_style "random" --model resnet20 

python compute_trajectory.py -r results/resnet20_skip_bn_bias_swag/trajectories --direction_file results/resnet20_skip_bn_bias_swag/random_directions.npz --projection_file random_proj.npz --model resnet20  -s results/resnet20_skip_bn_bias_swag/ckpt --skip_bn_bias

python compute_loss_surface.py --model_folder results/resnet20_skip_bn_bias_swag_diag -s results/resnet20_skip_bn_bias_swag_diag/ckpt/200_model.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results/resnet20_skip_bn_bias_swag_diag/pca_directions.npz --device cuda:0  --xcoords 50:-30:45 --ycoords 50:-30:45

python plot.py --result_folder results/resnet20_skip_bn_bias_swag/figures/  --surface_file results/resnet20_skip_bn_bias_swag_diag/loss_surface/pca_loss_surface_train_vanilla_5,-30,45.npz --plot_prefix resnet20_pca_loss_surface_train_vanilla_5,-30,45 --trajectory_file results/resnet20_skip_bn_bias_swag_diag/trajectories/pca_proj.npz

python plot_3d.py --result_folder results/resnet20_skip_bn_bias/figures_linear_color_3d/  --surface_file results/resnet20_skip_bn_bias/loss_surface/pca_loss_surface_train_vanilla_5,-30,45.npz --plot_prefix resnet20_3d_pca_loss_surface_train_vanilla_log --trajectory_file results/resnet20_skip_bn_bias/trajectories/pca_proj.npz --x_uplim 45 --x_lowlim -30 --y_uplim 45 --y_lowlim -30 --zlim 15

# adversarial 
python create_directions.py --statefile results/resnet20_adversarial_skip_bn_bias_swag/ckpt/200_model.pt -r results/resnet20_adversarial_skip_bn_bias_swag --skip_bn_bias --direction_file random_directions.npz --direction_style "random" --model resnet20 

python compute_trajectory.py -r results/resnet20_adversarial_skip_bn_bias_swag/trajectories --direction_file results/resnet20_adversarial_skip_bn_bias_swag/random_directions.npz --projection_file random_proj.npz --model resnet20  -s results/resnet20_adversarial_skip_bn_bias_swag/ckpt --skip_bn_bias

python compute_loss_surface.py --model_folder results/resnet20_adversarial_skip_bn_bias_swag -s results/resnet20_adversarial_skip_bn_bias_swag/swag_ckpt/200_swag_model.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results/resnet20_adversarial_skip_bn_bias_swag/random_directions.npz --device cuda:0  --xcoords 50:-1:1 --ycoords 50:-1:1 --use_swag_model --swag_num_samples 30 --swag_scale 1.0 --use_attack --data_root_dir results/resnet20_adversarial_skip_bn_bias_swag/adversarial_data

python compute_loss_surface.py --model_folder results/resnet20_adversarial_skip_bn_bias_swag -s results/resnet20_adversarial_skip_bn_bias_swag/ckpt/200_model.pt --batch_size 1000 --skip_bn_bias --model resnet20  --direction_file results/resnet20_adversarial_skip_bn_bias_swag/pca_directions.npz --device cuda:0  --xcoords 50:-30:45 --ycoords 50:-30:45 --use_attack --data_root_dir results/resnet20_adversarial_skip_bn_bias_swag/adversarial_data

python plot.py --result_folder results/resnet20_adversarial_skip_bn_bias_swag/figures/  --surface_file results/resnet20_adversarial_skip_bn_bias_swag/loss_surface/pca_loss_surface_train_vanilla_50,-30,45_swag_30ns_1.0e+00s_1.0e-05vc.npz --plot_prefix resnet20_pca_loss_surface_train_vanilla_50,-30,45_swag_30ns_1.0e+00s_1.0e-05vc --trajectory_file results/resnet20_adversarial_skip_bn_bias_swag/trajectories/pca_proj.npz

python plot_3d.py --result_folder results/resnet20_adversarial_skip_bn_bias_swag/figures_linear_color_3d/  --surface_file results/resnet20_adversarial_skip_bn_bias_swag/loss_surface/pca_loss_surface_train_vanilla_50,-30,45.npz --plot_prefix resnet20_3d_pca_loss_surface_train_vanilla_50,-30,45 --x_uplim 45 --x_lowlim -30 --y_uplim 45 --y_lowlim -30 --zlim 10 