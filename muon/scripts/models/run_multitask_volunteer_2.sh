#!/bin/bash
set -euo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

SAVE_DIR=$MUOND/clustering_models/run_volunteer_2
SPLITS_FILE=${MUOND}/subjects/tt_split_volunteer_majority_all_xy.pkl
AE_WEIGHTS=${MUOND}/clustering_models/dec/dec_no_labels/ae_weights.h5
DEC_WEIGHTS=${MUOND}/clustering_models/dec/dec_no_labels/DEC_model_final.h5

mkdir ${SAVE_DIR}
cp ${AE_WEIGHTS} ${SAVE_DIR}
cp ${DEC_WEIGHTS} ${SAVE_DIR}

python ${HERE}/multitask_run.py \
    --maxiter=80 \
    --gamma=.5 \
    ${SPLITS_FILE} ${SAVE_DIR}
