#!/bin/bash
wget 'https://zenodo.org/records/10693924/files/combined_input_for_triggers.tar.gz?download=1' -O input.tar.gz
tar -xvf input.tar.gz -C .
rm input.tar.gz
mv brat-outputs-all-nes inputs
