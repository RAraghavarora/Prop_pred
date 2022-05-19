#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH -J git
#SBATCH --mem-per-gpu=2500MB
git pull origin main
