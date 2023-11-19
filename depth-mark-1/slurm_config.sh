#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=depth_1
#SBATCH --time=50:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --account=teach0013
#sBATCH --exclusive
module load miniconda/3 -q
module load cuda/11.4 -q

conda activate check

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
pip3 install hydra-core -q
pip3 install numpy -q 
pip3 install matplotlib -q
pip3 install pillow -q
python train_dis.py  1>out.txt 2>err.txt 
