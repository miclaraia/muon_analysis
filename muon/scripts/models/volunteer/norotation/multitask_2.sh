#!/bin/bash
set -euo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"
SCRIPT="${MUON}/muon/scripts/models/"

SPLITS_FILE=${MUOND}/subjects/split_volunteer_norotation_xy.pkl
MODEL_NAME="volunteer/norotation/multitask_1"
SOURCE_DIR=${MUOND}/clustering_models/dec/dec_no_labels

python ${SCRIPT}/multitask_run.py \
    --splits_file ${SPLITS_FILE} \
    --source_dir ${SOURCE_DIR} \
    --model_name ${MODEL_NAME} \
    --lr 0.001 \
    --maxiter 100 \
    --gamma 0.7

