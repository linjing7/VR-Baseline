#!/bin/bash
#SBATCH -J gopro
#SBATCH -p cvr
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:hgx:4
#SBATCH --mem 500GB
#SBATCH --time=10-00:00:00
#SBATCH -o dvd_retrain.log
#SBATCH --qos=preemptive
#SBATCH --exclude=hgx042
bash tools/dist_train.sh configs/FGST_deblur_gopro_test.py 4
