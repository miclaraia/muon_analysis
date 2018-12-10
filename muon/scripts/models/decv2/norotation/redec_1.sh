#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"
SCRIPT="${MUON}/muon/scripts/models/"

MODEL_NAME="decv2/norotation/redec_1"
SOURCE_DIR=${MUOND}/clustering_models/decv2/norotation/$1
shift
NAME="$@"

python ${SCRIPT}/redec_run.py \
    --name "${NAME}" \
    --source_dir ${SOURCE_DIR} \
    --model_name ${MODEL_NAME} \
    --epochs 200 \
