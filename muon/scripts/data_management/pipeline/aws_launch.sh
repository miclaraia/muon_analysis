#!/bin/bash
set -euvxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"



POS=()
while [[ $# -gt 0 ]]; do
    key="$1"

    case ${key} in
        --data)
            DATA=$2
            shift 2
            ;;
        --name)
            NAME=$2
            shift 2
            ;;
        -h|--host)
            HOST=$2
            shift 2
            ;;
        -d|--device)
            DEVICE=$2
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

DATA="\${MUOND}/$(python -c "import os.path; print(os.path.relpath('${DATA}', '${MUOND}'))")"
SAVEDIR="\${MUOND}/clustering_models/project/${NAME}-$(date --iso-8601=s)"

${MUON}/muon/scripts/aws_deploy/aws.sh \
    -h ${HOST} \
    -d ${DEVICE} \
    launch --script ${HERE}/train_model.py \
    "--database_file ${DATA} \
    --name ${NAME} \
    --save_dir ${SAVEDIR}"


