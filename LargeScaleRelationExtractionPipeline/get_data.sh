#!/bin/bash
wget 'https://zenodo.org/api/records/8139717/files/combined_input_for_re.tar.gz/content' -O input.tar.gz
tar -xvf input.tar.gz -C .
rm input.tar.gz
