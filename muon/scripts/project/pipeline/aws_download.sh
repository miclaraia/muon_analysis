#!/bin/bash
set -euvxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"


POS=()
while [[ $# -gt 0 ]]; do
    key="$1"

    case ${key} in
        -h|--host)
            HOST=$2
            shift 2
            ;;
        -d|--device)
            DEVICE=$2
            shift 2
            ;;
        --name)
            NAME=$2
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

${MUON}/muon/scripts/aws_deploy/aws.sh \
    -h ${HOST} \
    -d ${DEVICE} \
    download_run project/${NAME}
