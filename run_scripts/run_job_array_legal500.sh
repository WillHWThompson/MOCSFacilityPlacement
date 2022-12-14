#!/bin/bash
#SBATCH --partition=bluemoon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=30:00:00
#SBATCH --job-name=fc10kil
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.error
#SBATCH --array=1-500%10


START=9750
SLURM_ARRAY_TASK_ID=$1
TOTAL_STEPS=$((START+SLURM_ARRAY_TASK_ID))
echo "TOTAL STEPS: $TOTAL_STEPS"
source ~/.bashrc
conda init
conda activate mocs_env

cd /users/w/t/wthomps3/CSDS/year1/MOCS/fac_place/MOCSFacilityPlacement

python3 /gpfs1/home/w/t/wthomps3/CSDS/year1/MOCS/fac_place/MOCSFacilityPlacement/run_model.py --num_steps 1 --out_path /gpfs1/home/w/t/wthomps3/CSDS/year1/MOCS/fac_place/MOCSFacilityPlacement/output/500_runs_output --no-legal_states_only 
#python3 /gpfs1/home/w/t/wthomps3/CSDS/year1/MOCS/fac_place/MOCSFacilityPlacement/run_model.py --num_steps $START+$SLURM_ARRAY_TASK_ID  --out_path /gpfs1/home/w/t/wthomps3/CSDS/year1/MOCS/fac_place/MOCSFacilityPlacement/output --no-legal_states_only 
