#!/bin/bash
mkdir -p original_model
mkdir -p the_best_model
mkdir -p output
mkdir -p jobs
mkdir -p logs
mkdir -p preds
mkdir -p puhtilogs

wget 'https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz' -O model.tar.gz
tar -xvf model.tar.gz -C ./original_model
rm model.tar.gz

wget 'https://zenodo.org/records/8139717/files/relation_extraction_string_v12_best_model.tar.gz?download=1' -O model.tar.gz
tar -xvf model.tar.gz -C ./the_best_model
rm model.tar.gz

python rewite_model_address.py
