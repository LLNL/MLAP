#!/bin/sh
#SBATCH -J wildfire_prep
#SBATCH -N 1
#SBATCH --partition=quartz
#SBATCH -t 24:00:00
#SBATCH -A cr4ns
##SBATCH -A gsmisc
#SBATCH -p pbatch
##SBATCH --mail-type=ALL
#SBATCH --export=ALL

#Display job info
date
echo "Job id = $SLURM_JOBID"
hostname

export python_script=$1
export json_extract_data=$2
export json_prepare_data=$3
#export json_train_model=$4
echo "Running the python script: ${python_script}"
echo "... Using the input json files for: "
echo "... ... Extracting Data: ${json_extract_data}"
echo "... ... Preparing Data:  ${json_prepare_data}"
#echo "... ... Training Model:  ${json_train_model}"

# RUN THIS SCRIPT FROM A conda ENVIRONMENT

#source $HOME/.bashrc
#conda_invoke
#conda activate py3_ml
#source $HOME/VirtualEnv/py3_ml_wind/bin/activate

python $python_script $json_extract_data $json_prepare_data

#conda deactivate
#conda deactivate

echo "Done running the python script with input json file"
