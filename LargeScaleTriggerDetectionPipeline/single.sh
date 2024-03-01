#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
### update your project number
#SBATCH --account=
###update where cluster output log and error log should go
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
module load pytorch/1.12
export TRANSFORMERS_CACHE="${DIR%/*/*}/transformers_cache/"
export TOKENIZERS_PARALLELISM="false"

# run the large scale pipeline

python run_ls_pipeline.py \
    --configs_file_path "${DIR}/ComplexTome_configs.json" \
    --pretrained_model_path "${DIR}/the_best_model" \
    --log_file_path "${DIR}/logs/${folder_index}.log" \
    --input_folder_path "${DIR}/${inputs_dir}/${folder_index}" \
    --output_folder_path "${DIR}/outputs/output_${folder_index}"
