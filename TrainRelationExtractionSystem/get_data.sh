#!/bin/bash
wget 'https://zenodo.org/records/8139717/files/ComplexTome.tar.gz?download=1' -O data.tar.gz
tar -xvf data.tar.gz -C .
rm data.tar.gz
mv physical-interaction-corpus/splits splits
rm -rf physical-interaction-corpus
rm -rf splits/test-set
mkdir -p MODEL
mkdir -p OUTPUTS
mkdir -p OUTPUTS/jobs
mkdir -p OUTPUTS/logs
mkdir -p OUTPUTS/preds
mkdir -p OUTPUTS/clusterlogs

wget 'https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz' -O model.tar.gz
tar -xvf model.tar.gz -C ./MODEL
rm model.tar.gz





