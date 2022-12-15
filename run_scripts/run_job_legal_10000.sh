#!/bin/bash
#SBATCH --partition=bluemoon
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem=1G
#SBATCH --time=30:00:00
#SBATCH --job-name=fc10kl
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.error
# Receive emails when job begins and ends or fails


source ~/.bashrc
conda init
conda activate mocs_env

python3 /gpfs1/home/w/t/wthomps3/CSDS/year1/MOCS/fac_place/MOCSFacilityPlacement/run_model.py --num_steps 10000 --out_path /gpfs1/home/w/t/wthomps3/CSDS/year1/MOCS/fac_place/MOCSFacilityPlacement/output --legal_states_only 
