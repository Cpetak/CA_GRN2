#!/bin/sh

#Specify a partition
#SBATCH --partition=short
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=1
# Request 3 cores per task
#SBATCH --cpus-per-task=8
# Request memory
#SBATCH --mem=40G
# Run for five minutes
#SBATCH --time=03:00:00
# Name job
#SBATCH --job-name=SbatchJob
# Name output file
#SBATCH --output=%x_%j.out

# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}

# Executable section: echoing some Slurm data
echo "Starting sbatch script myscript.sh at:`date`"

cd /users/c/p/cpetak/CA_GRN2

echo "rule: $1"
echo "exp_type: $2"
echo "candidate_idx: $3"

python noisify_grn.py --rule $1 --exp_type $2 --candidate_idx $3