#!/bin/bash
mkdir -p the_best_model
mkdir -p outputs
mkdir -p jobs
mkdir -p logs
mkdir -p preds
mkdir -p puhtilogs

wget 'https://zenodo.org/records/10693924/files/relation_extraction_for_trigger_detection_best_model.tar.gz?download=1' -O model.tar.gz
tar -xvf model.tar.gz -C ./the_best_model
rm model.tar.gz
