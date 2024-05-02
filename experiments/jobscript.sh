#!/bin/zsh
#SBATCH --job-name=adv_training_test
#SBATCH --time=00:30:00
#SBATCH --err ./logs/errors.err
#SBATCH --out ./logs/output.out
#SBATCH --mem-per-cpu=3900M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aaronberger@hotmail.de

export CONDA_ROOT=$HOME/miniconda3 
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

conda activate sandbox
python -u test_run.py