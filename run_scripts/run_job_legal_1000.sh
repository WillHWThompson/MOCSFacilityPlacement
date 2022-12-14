#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=2
#SBATCH --ntasks=48
#SBATCH --mem=1G
#SBATCH --time=0:30:00
#SBATCH --job-name=fac_place_test
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.error
# Receive emails when job begins and ends or fails

source ~/.bashrc
conda init
conda activate mocs_env

python3 /gpfs1/home/w/t/wthomps3/CSDS/year1/MOCS/fac_place/MOCSFacilityPlacement/run_model.py --num_steps 1000 --out_path /gpfs1/home/w/t/wthomps3/CSDS/year1/MOCS/fac_place/MOCSFacilityPlacement/output
