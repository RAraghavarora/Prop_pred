#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --partition=haswell
#SBATCH -J bob_hp_qm9
#SBATCH --output=bob.out
#SBATCH --error=bob.err
#SBATCH -A p_biomolecules
#SBATCH -N 4

#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=all
#SBATCH        --mail-user=reepicheep_logs@protonmail.com
#SBATCH --mem-per-cpu=6000MB


ulimit -s unlimited
echo Starting Program
module purge                                 # purge if you already have modules loaded
module load modenv/scs5
module load Python/3.6.4-intel-2018a
. /home/medranos/vdftb20/bin/activate
module load cuDNN/8.0.4.30-CUDA-11.1.1
echo "training starts"

work=/scratch/ws/1/medranos-DFTBprojects/raghav/Prop_pred
python3 $work/bob_split_hp.py

echo "training is over :-)"
EXTSTAT=$?
exit $EXTSTAT
