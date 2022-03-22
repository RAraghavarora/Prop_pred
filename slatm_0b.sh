#!/bin/bash
#SBATCH --time=156:00:00
#SBATCH --partition=gpu2                        # specify ml partition or gpu2 partition
#SBATCH --gres=gpu:2                      # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --nodes=1                        # request 1 node
#SBATCH --ntasks=6
#SBATCH -J ra-eq
#SBATCH --output=slatm0b.out
#SBATCH --error=slatm0b.err
#SBATCH -A p_biomolecules
#SBATCH --mail-type=all
#SBATCH        --mail-user=reepicheep_logs@protonmail.com
#SBATCH --mem-per-gpu=16000MB
ulimit -s unlimited
echo Starting Program
module purge                                 # purge if you already have modules loaded
module load modenv/scs5
module load Python/3.6.4-intel-2018a
. /home/medranos/vdftb20/bin/activate
module load cuDNN/8.0.4.30-CUDA-11.1.1

work=/scratch/ws/1/medranos-DFTBprojects/raghav/Prop_pred
python3 $work/slatm_split_actual_0b.py

echo "training is over :-)"
EXTSTAT=$?
exit $EXTSTAT
