#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"
SCRIPT="${MUON}/muon/scripts/models/"

MODEL_NAME="decv2/norotation/multitask_1"
SOURCE_DIR=${MUOND}/clustering_models/decv2/norotation/$1
shift
NAME="$@"

python ${SCRIPT}/multitask_run.py \
    --name "${NAME}" \
    --source_dir ${SOURCE_DIR} \
    --model_name ${MODEL_NAME} \
    --maxiter 100 \
    --gamma 1.0

