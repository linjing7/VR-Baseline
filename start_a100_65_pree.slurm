#!/bin/bash
#SBATCH -J gopro_test
#SBATCH -p cvr-preempted
#SBATCH -N 1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:hgx:4
#SBATCH --mem 500GB
#SBATCH --time=10-00:00:00
#SBATCH -o gopro_test.log
bash tools/dist_train.sh configs/FGST_deblur_gopro_test.py 4