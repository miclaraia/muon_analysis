#!/bin/bash
set -euo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

SPLITS_FILE=${MUOND}/subjects/tt_split_hugh_xy.pkl
MODEL_NAME=run_multitask_7
SOURCE_DIR=${MUOND}/clustering_models/dec/dec_no_labels

if [ -d ${SAVE_DIR} ]; then
    echo "Save dir already exists!"
    exit 1
fi

mkdir -p ${SAVE_DIR}

python ${HERE}/multitask_run.py \
    --splits_file ${SPLITS_FILE} \
    --source_dir ${SOURCE_DIR} \
    --model_name ${MODEL_NAME}
    --maxiter 80 \
    --gamma 1.0

