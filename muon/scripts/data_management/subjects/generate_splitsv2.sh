#!/bin/bash

# python $MUON/muon/scripts/data_management/subjects/generate_data_split.py \
#     $MUOND/subjects/subject_data_v3.hdf5 \
#     volunteer_majority \
#     $MUOND/subjects/tt_split_volunteer_majority_all.json
# 
# 
# 
# @click.option('--subject_data', required=True)
# @click.option('--train_labels', required=True)
# @click.option('--true_labels', required=True)
# @click.option('--output_file', required=True)
# @click.option('--train_rotation', is_flag=True)
# @click.option('--true_rotation', is_flag=True)
# def main(subject_data, train_labels, true_labels, output_file,
#          train_rotation, true_rotation):

NAME="v2_norotation"

cd $MUON/muon/scripts/data_management/subjects
python generate_splits.py \
    --subject_data $MUOND/subjects/subject_data_v3.hdf5 \
    --train_label_name volunteer_majority \
    --true_label_name hugh \
    --splits_out $MUOND/subjects/split_${NAME}.json \
    --xy_out ${MUOND}/subjects/split_${NAME}_xy.pkl \
    --train 0.9 --train_dev 0.1 --valid 0.75 --test 0.25 \
    --train_rotation \
    --true_rotation

NAME="v2_rotation"
python generate_splits.py \
    --subject_data $MUOND/subjects/subject_data_v3.hdf5 \
    --train_label_name volunteer_majority \
    --true_label_name hugh \
    --splits_out $MUOND/subjects/split_${NAME}.json \
    --xy_out ${MUOND}/subjects/split_${NAME}_xy.pkl \
    --train 0.9 --train_dev 0.1 --valid 0.75 --test 0.25
