# Code for running prediction with the best model

After having trained a model using the code in the `TrainRelationExtractionSystem` directory one can proceed with predictions on a small sample of the literature or even the entire literature. 

## Prediction on sample data

If you haven't done so already you will need to clone the repository to your cluster drive space. Assume you copy to your home directory $HOME

You must first do a git clone:
```
cd $HOME
git clone git@github.com:farmeh/ComplexTome_extraction.git
```

Then you need to either use a model you trained yourself using the code in `TrainRelationExtractionSystem` or download the best model we have pretrained from Zenodo, by running: 
```
bash get_model.sh
```
This code first ceate all necessary folders and then downloads the [RoBERTa-large-PM-M3-Voc model](https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz) (pre-trained RoBERTa model on PubMed and PMC and MIMIC-III with a BPE Vocab learnt from PubMed),
which is used by our system and extract it into the `original_model` folder: `$HOME/ComplexTome_extraction/LargeScaleRelationExtractionPipeline/original_model/`.
In case this fails, you can manually download the pre-trained model from [here](https://github.com/facebookresearch/bio-lm/blob/main/README.md) and extract it to the model folder.

If everything goes right, then the model should be here:`$HOME/ComplexTome_extraction/LargeScaleRelationExtractionPipeline/original_model/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf`
If not, make sure to download the model manually and place it correctly into that folder.

Then the code will download the trained model **weights** from Zenodo and extract it in `the_best_model` directory. 
It then executes `python rewite_model_address.py` to add `original_model` path to `the_best_model/info.json` file.
and if that line fails, try with `python3 rewite_model_address.py`.


Then in order to run prediction you need to run (after updating it with needed information such as you account information on the cluster, etc):
```
sbatch single.sh 1 sample_data
```

This uses the data in the `sample_data` directory, the best model we have used for relation extraction of Complex Formation relationships for STRING v12 and outputs the results in a tab-delimited file with 5 columns: PubMed Identifier, Entity ID1, Entity ID2, prediction (positive or negative) and a list of the positive and negative score coming from the relation extraction model. 

## Large scale run on the entire literature

To replicate the large scale run in the entire literature you first need to get all the data from Zenodo:
```
get_data.sh
```
Then we recommend running the shell scripts that submits the jobs inside a screen, since it goes through all input directories and starts a gpu job per directory: 
```
screen -S mygrid
bash grid_screen.sh
```

To detach from the screen press `Ctrl+A+D` and to resume the screen `screen -r mygrid`. Finally, to completely terminate the screen make sure you are connected in the same login node in which you have started the screen and press `Ctrl+D`.

## Technical consideration on Puhti

You need to install spacy, scispacy, and the `en_core_sci_sm` model, a full spaCy pipeline for biomedical data, under your user.

```
module purge
module load tensorflow/2.8
python -m pip install --user spacy==2.3.2
python -m pip install --user scispacy==0.2.5
python -m pip install --user https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz
python -m pip install --user torch==1.9.1
```


