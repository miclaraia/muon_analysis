#!/bin/bash
set -euo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"
SCRIPT="${MUON}/muon/scripts/models/"

SPLITS_FILE=${MUOND}/subjects/split_v2_rotation_xy.pkl
MODEL_NAME="decv2/rotation/dec_1"

python ${SCRIPT}/decv2_run.py \
    --splits_file ${SPLITS_FILE} \
    --model_name ${MODEL_NAME} \
    --maxiter 20000 \
