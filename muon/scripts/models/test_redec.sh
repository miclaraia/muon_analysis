#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

SPLITS_FILE=${MUOND}/subjects/tt_split_hugh_xy.pkl
SAVE_DIR=$MUOND/clustering_models/run_redec_test
SOURCE_DIR=$MUOND/clustering_models/run_multitask_4_1

if [ -d ${SAVE_DIR} ]; then
    echo "Save dir already exists!"
    exit 1
fi

mkdir -p ${SAVE_DIR}

python ${HERE}/redec_run.py \
    --splits_file ${SPLITS_FILE} \
    --source_dir ${SOURCE_DIR} \
    --save_dir ${SAVE_DIR} \
    --epochs 10

