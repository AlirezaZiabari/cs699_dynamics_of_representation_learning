#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:1

# inputs to slurm (should be exported as enviroment variables)
: "${USERNAME:=}" #put your username here
: "${log_file:=log.txt}"
: "${Q:=gpu}"
: "${TIME:=48:00:00}"
: "${MEM:=32GB}"

: "${env_dir:=/scratch/$USERNAME/drl_venv/bin/activate}"
: "${code_dir:=/scratch/$USERNAME/cs699_dynamics_of_representation_learning/loss_landscape}"

: "${result_folder:=$code_dir/results/resnet20_skip_bn_bias}"
: "${log_dir:=$/scratch/$USERNAME/logs_python}"

: "${device:="cuda"}"
: "${model:="resnet20"}"
: "${attack_type:="pgd"}"
: "${batch_size:=128}"

# For discovery
module purge
module load gcc/8.3.0
module load cuda/11.0.2
module load cudnn/8.0.2-10.1
module load nvidia-hpc-sdk
# module spider xvfb

mkdir -p $log_dir

source $env_dir
export MPLCONFIGDIR=/scratch/$USERNAME/python-cache/mpl-cache
echo Log file: ${log_dir}/${log_file}
cd $code_dir

python -u train.py --batch_size $batch_size --result_folder $result_folder --device $device --model $model --skip_bn_bias -D | tee -a ${log_dir}/${log_file} 
