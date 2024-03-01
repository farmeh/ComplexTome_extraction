#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH -p gpu
###SBATCH -p gputest
#SBATCH -t 15:00:00
###SBATCH -t 00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
### update your project number
#SBATCH --account=
###assuming the script is run from the current directory
#SBATCH -o OUTPUTS/clusterlogs/%j.out
#SBATCH -e OUTPUTS/clusterlogs/%j.err

#folders and paths ...
DIR=$(pwd)
export CT_REL_FOLDERPATH=$(pwd)
model_address="$DIR/MODEL/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf/"
train_set_address="$DIR/splits/train-set/"
devel_set_address="$DIR/splits/dev-set/"
preds_model_output_address="$DIR/OUTPUTS/preds/$SLURM_JOBID"
logfile_address="$DIR/OUTPUTS/logs/${SLURM_JOBID}.log"

echo model_address=$model_address
echo devel_set_address=$devel_set_address
echo preds_model_output_address=$preds_model_output_address
echo logfile_address=$logfile_address

#create a prediction folder
mkdir -p $preds_model_output_address

# delete job file on exit
function on_exit {
    rm -f "$DIR/OUTPUTS/jobs/$SLURM_JOBID"
}
trap on_exit EXIT

# check number of input params
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 random_seed_index"
    exit 1
fi

random_seed_index="$1"

# set execution environment
module purge
module load tensorflow/2.8
#update transformers_cache_directory if necessary. Currently, this points to the directory where the ComplexTome_extraction directory is located
export TRANSFORMERS_CACHE= ...
#export TRANSFORMERS_CACHE="${DIR%/*/*}/transformers_cache/"
export TOKENIZERS_PARALLELISM="false"

# run the training pipeline
python3 ct_train_pipeline.py \
    --random_seed_index "$random_seed_index" \
    --model_address "$model_address" \
    --train_set_address "$train_set_address" \
    --devel_set_address "$devel_set_address" \
    --preds_model_output_address "$preds_model_output_address" \
    --logfile_address "$logfile_address"
