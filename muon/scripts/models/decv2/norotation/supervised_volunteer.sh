#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"
SCRIPT="${MUON}/muon/scripts/models/"

SPLITS_FILE=${MUOND}/subjects/split_v2_volunteer_norotation_xy.pkl
MODEL_NAME="decv2/norotation/supervised_volunteer_1"
NAME="$@"

python ${SCRIPT}/supervised_run.py \
    --name "${NAME}" \
    --splits_file ${SPLITS_FILE} \
    --model_name ${MODEL_NAME} \
    --epochs 100 \


