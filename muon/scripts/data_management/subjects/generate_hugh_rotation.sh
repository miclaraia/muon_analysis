#!/bin/bash

NAME="hugh_rotation"

python $MUON/muon/scripts/data_management/subjects/generate_data_split.py \
    --subject_data $MUOND/subjects/subject_data_v3.hdf5 \
    --label_name hugh \
    --splits_out $MUOND/subjects/split_${NAME}.json \
    --xy_out $MUOND/subjects/split_${NAME}_xy.pkl \
    --rotation
