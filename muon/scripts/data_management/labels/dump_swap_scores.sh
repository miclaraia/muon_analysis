#!/bin/bash
set -euvxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

DATA="${MUOND}/zooniverse/MH2"

python ${HERE}/../../mh2_run_swap.py \
    --export ${DATA}/muon-hunters-2-dot-0-classifications.csv \
    --images ${DATA}/image_structure/production_data.h5 \
    --score_output ${DATA}/swap_scores.csv \
    --label_output ${DATA}/swap_labels.csv \
    --golds ${DATA}/mh2_golds.csv \
