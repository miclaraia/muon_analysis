#!/bin/bash

python $MUON/muon/scripts/data_management/subjects/dump_splits.py \
  #--rotation \
  $MUOND/subjects/tt_split_hugh.json \
  $MUOND/subjects/subject_data_v3.hdf5 \
  hugh \
  $MUOND/subjects/tt_split_hugh_xy.pkl
