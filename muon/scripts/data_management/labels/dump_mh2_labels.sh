#!/bin/bash
set -euvxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

DATA="${MUOND}/zooniverse/MH2"

python ${HERE}/../../mh2_dump_volunteer_labels.py \
    --export ${DATA}/muon-hunters-2-dot-0-classifications.csv \
    --images ${DATA}/image_structure/production_data.h5 \
    --majority_output ${DATA}/majority_labels.csv \
    --fraction_output ${DATA}/fraction.csv \
    --first_output ${DATA}/first_label.csv

python ${HERE}/../../mh2_dump_volunteer_labels.py \
    --export ${DATA}/muon-hunters-2-dot-0-classifications.csv \
    --images ${DATA}/image_structure/production_data.h5 \
    --majority_output ${DATA}/majority_labels0.6.csv \
    --threshold 0.6

python ${HERE}/../../mh2_dump_volunteer_labels.py \
    --export ${DATA}/muon-hunters-2-dot-0-classifications.csv \
    --images ${DATA}/image_structure/production_data.h5 \
    --majority_output ${DATA}/majority_labels0.4.csv \
    --threshold 0.4

python ${HERE}/../../mh2_dump_volunteer_labels.py \
    --export ${DATA}/muon-hunters-2-dot-0-classifications.csv \
    --images ${DATA}/image_structure/production_data.h5 \
    --majority_output ${DATA}/majority_labels0.2.csv \
    --threshold 0.2
    
