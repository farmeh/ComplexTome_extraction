#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH -p gpu
###SBATCH -p gputest
#SBATCH -t 02:00:00
###SBATCH -t 00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
### update with your project number
#SBATCH --account=
#SBATCH -o puhtilogs/%j.out
#SBATCH -e puhtilogs/%j.err

#get current working directory
DIR=$(pwd)

# delete job file on exit
function on_exit {
    rm -f jobs/$SLURM_JOBID
}
trap on_exit EXIT

# check number of input params
if [ "$#" -ne 2 ]; then
    echo "incorrect params"
    exit 1
fi

folder_index="$1"
inputs_dir="$2"
# create/recreate output directory
rm -rf "outputs/output_${folder_index}"
mkdir "outputs/output_${folder_index}"

# set execution environment
module purge
module load tensorflow/2.8
export TRANSFORMERS_CACHE="${DIR%/*/*}/transformers_cache/"
export TOKENIZERS_PARALLELISM="false"

# run the large scale pipeline

python run_ls_pipeline.py \
    --configs_file_path "${DIR}/ComplexTome_configs.json" \
    --log_file_path "${DIR}/logs/${folder_index}.log" \
    --model_folder_path "${DIR}/the_best_model" \
    --input_folder_path "${DIR}/${inputs_dir}/input_${folder_index}" \
    --output_folder_path "${DIR}/outputs/output_${folder_index}"
