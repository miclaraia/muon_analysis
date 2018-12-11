#!/bin/bash
set -euo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"
SCRIPT="${MUON}/muon/scripts/models/"

A=$1
shift
N="$(python -c "for i in '${1}'.split(','): print(i)")"
echo "Running iterations $N"
shift
NAME="$@"

get_date() {
    echo $(date +"%Y-%m-%dT%H:%M:%S")
}

SPLITS_FILE=${MUOND}/subjects/split_v2_swap_norotation_xy.pkl


for i in ${N}; do
    MODEL_NAME="decv2/norotation/paper_runs_${A}/run_${i}"
    ROOT_SAVE_DIR=${MUOND}/clustering_models/${MODEL_NAME}
    echo "MODEL_NAME=$MODEL_NAME"
    echo "ROOT_SAVE_DIR=$ROOT_SAVE_DIR"
    echo

    SAVE_DIR=${ROOT_SAVE_DIR}/dec-$(get_date)
    echo "Starting run dec"
    echo "SAVE_DIR=${SAVE_DIR}"
    python ${SCRIPT}/decv2_run.py \
        --name "${NAME} ${i} dec" \
        --splits_file ${SPLITS_FILE} \
        --save_dir ${SAVE_DIR} \
        --model_name None \
        --maxiter 20000 \
        --n_clusters 10

    SOURCE_DIR=${SAVE_DIR}
    SAVE_DIR=${ROOT_SAVE_DIR}/multitask-$(get_date)
    echo "Starting run multitask"
    echo "SAVE_DIR=${SAVE_DIR}"
    echo "SOURCE_DIR=${SOURCE_DIR}"
    python ${SCRIPT}/multitask_run.py \
        --name "${NAME} ${i} multitask" \
        --source_dir ${SOURCE_DIR} \
        --save_dir ${SAVE_DIR} \
        --model_name None \
        --maxiter 200 \
        --gamma 1.0 \
        --patience 10

    SOURCE_DIR=${SAVE_DIR}
    SAVE_DIR=${ROOT_SAVE_DIR}/redec-$(get_date)
    echo "Starting run redec"
    echo "SAVE_DIR=${SAVE_DIR}"
    echo "SOURCE_DIR=${SOURCE_DIR}"
    python ${SCRIPT}/redec_run.py \
        --name "${NAME} ${i} redec" \
        --source_dir ${SOURCE_DIR} \
        --save_dir ${SAVE_DIR} \
        --model_name None \
        --epochs 200 
done
