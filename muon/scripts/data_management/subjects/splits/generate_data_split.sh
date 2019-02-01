#!/bin/bash

python $MUON/muon/scripts/data_management/subjects/generate_data_split.py \
  $MUOND/subjects/subject_data_v3.hdf5 hugh $MUOND/subjects/tt_split_hugh.json
