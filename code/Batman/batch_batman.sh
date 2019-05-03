#!/bin/bash
#SBATCH --job-name=gen_batman
#SBATCH --workdir=/common/contrib/classroom/ast520/tess_batman/code/Batman/
#SBATCH --output=/common/contrib/classroom/ast520/tess_batman/data/log/batman%A.log
#SBATCH --time=01:00:00
#SBATCH --mem=10000
#SBATCH --account=ast520-spr19

# 2500 curves ran in 2 minutes
# 100x(2500) curves in about 2 hours (gave it 4 because monsoon cores slower)
# 250000 curves ran in 10 minutes, used 8 GB RAM

TBP=/common/contrib/classroom/ast520/tess_batman
echo Starting
module load anaconda
source activate batman

cmd="python $TBP/code/Batman/batman_monsoon.py $TBP/code/Batman/param.txt $TBP/data/"
echo running command: $cmd
srun $cmd
echo Finished

