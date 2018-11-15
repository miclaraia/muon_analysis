#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

MULTITASK_DIR=$MUOND/clustering_models/run_volunteer_1
SAVE_DIR=${MULTITASK_DIR}/run_redec_1
SPLITS_FILE=${MUOND}/subjects/tt_split_volunteer_majority_all_xy.pkl
AE_WEIGHTS=${MULTITASK_DIR}/ae_weights.h5
DEC_WEIGHTS=${MULTITASK_DIR}/DEC_model_final.h5

mkdir ${SAVE_DIR}
cp ${AE_WEIGHTS} ${SAVE_DIR}
cp ${DEC_WEIGHTS} ${SAVE_DIR}

python ${HERE}/redec_run.py \
    --epochs 10 \
    ${SPLITS_FILE} ${MULTITASK_DIR} ${SAVE_DIR}
