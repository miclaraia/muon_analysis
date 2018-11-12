#!/bin/bash

python $MUON/muon/scripts/data_management/subjects/generate_data_split.py \
    $MUOND/subjects/subject_data_v3.hdf5 \
    volunteer_majority \
    $MUOND/subjects/tt_split_volunteer_majority_all.json
