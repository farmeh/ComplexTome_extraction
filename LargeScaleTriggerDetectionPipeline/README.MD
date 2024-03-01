# Code for running the trigger detection pipeline

## Preparing the system
If you haven't done so already you will need to clone the repository to your cluster drive space. Assume you copy to your home directory `$HOME`

You must first do a git clone:

```
cd $HOME
git clone git@github.com:farmeh/ComplexTome_extraction.git
```

Then you need to download the best PyTorch based-model we have pretrained from Zenodo, by running: 
```
bash get_model.sh
```

This will download the model from Zenodo and extract it in `the_best_model` directory. On top of that it will create the necessary directories for successfully running the job.

Then to run the pipeline on a sample dataset you need to run:

```
sbatch single.sh 1 sample_data
```

This uses the data in the `sample_data` directory, the best model we have used for relation extraction of Complex Formation relationships for trigger detection and outputs the results in a tab-delimited file with 9 columns: PubMed id, Entity ID1, Entity ID2, Negative logit for complex formation relationship, Positive logit for complex formation relationship, trigger score, start offset, end offset, trigger word match.

## Large scale run on the entire literature

To replicate the large scale run in the entire literature you first need to get all the data from Zenodo:
```
bash get_data.sh
```
Then we recommend running the shell script that submits the jobs inside a screen, since it goes through all input directories and starts a gpu job per directory: 
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
```

