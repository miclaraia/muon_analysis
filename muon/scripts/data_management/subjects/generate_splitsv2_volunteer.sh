#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"
SCRIPT="${MUON}/muon/scripts/models/"


python ${HERE}/translate_splits.py \
    --subject_file ${MUOND}/subjects/subject_data_v3.hdf5 \
    --splits_source ${MUOND}/subjects/split_v2_swap_norotation.json \
    --xy_source ${MUOND}/subjects/split_v2_swap_norotation_xy.pkl \
    --xy_out ${MUOND}/subjects/split_v2_volunteer_norotation_xy.pkl \
    --label_name volunteer_majority
