#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"
SCRIPT="${MUON}/muon/scripts/models/"

MODEL_NAME="volunteer/rotation/redec_1"
MULTITASK_DIR=${MUOND}/clustering_models/volunteer/rotation/$1

python ${SCRIPT}/redec_run.py \
    --source_dir ${MULTITASK_DIR} \
    --model_name ${MODEL_NAME} \
    --epochs 200 \
