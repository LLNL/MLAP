#!/bin/sh
#SBATCH -J wildfire_data
#SBATCH -N 1
#SBATCH --partition=quartz
#SBATCH -t 24:00:00
#SBATCH -A gsmisc
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH --export=ALL

#Display job info
date
echo "Job id = $SLURM_JOBID"
hostname

export python_script=$1
export input_json=$2
echo "Running the python script: ${python_script}"
echo "...Using the input json file: ${input_json}"

# RUN THIS SCRIPT FROM A conda ENVIRONMENT

#source $HOME/.bashrc
#conda_invoke
#conda activate py3_ml
#source $HOME/VirtualEnv/py3_ml_wind/bin/activate

python $python_script $input_json

#conda deactivate
#conda deactivate

echo "Done running the python script with input json file"
