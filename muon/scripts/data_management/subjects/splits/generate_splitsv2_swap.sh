#!/bin/bash
set -euxo pipefail

#NAME="v2_rotation"
#cd $MUON/muon/scripts/data_management/subjects
#python generate_splits.py \
    #--subject_data $MUOND/subjects/subject_data_v3.hdf5 \
    #--train_label_name volunteer_majority \
    #--true_label_name hugh \
    #--splits_out $MUOND/subjects/split_${NAME}.json \
    #--xy_out ${MUOND}/subjects/split_${NAME}_xy.pkl \
    #--train 0.9 --train_dev 0.1 --valid 0.5 --test 0.5 \
    #--train_rotation

#NAME="v2_norotation"
#python generate_splits.py \
    #--subject_data $MUOND/subjects/subject_data_v3.hdf5 \
    #--train_label_name volunteer_majority \
    #--true_label_name hugh \
    #--splits_out $MUOND/subjects/split_${NAME}.json \
    #--xy_out ${MUOND}/subjects/split_${NAME}_xy.pkl \
    #--train 0.9 --train_dev 0.1 --valid 0.5 --test 0.5 \

NAME="v2_swap_rotation"
cd $MUON/muon/scripts/data_management/subjects
python generate_splits.py \
    --subject_data $MUOND/subjects/subject_data_v3.hdf5 \
    --train_label_name swap \
    --true_label_name hugh \
    --splits_out $MUOND/subjects/split_${NAME}.json \
    --xy_out ${MUOND}/subjects/split_${NAME}_xy.pkl \
    --train 0.9 --train_dev 0.1 --valid 0.5 --test 0.5 \
    --train_rotation

NAME="v2_swap_norotation"
python generate_splits.py \
    --subject_data $MUOND/subjects/subject_data_v3.hdf5 \
    --train_label_name swap \
    --true_label_name hugh \
    --splits_out $MUOND/subjects/split_${NAME}.json \
    --xy_out ${MUOND}/subjects/split_${NAME}_xy.pkl \
    --train 0.9 --train_dev 0.1 --valid 0.5 --test 0.5 \
