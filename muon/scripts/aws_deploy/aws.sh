#!/bin/bash
set -euvxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

get_host() {
    if [[ "$1" == "gpu" ]]; then
        HOST="ec2-3-17-23-132.us-east-2.compute.amazonaws.com"
        TYPE="ubuntu"
    elif [[ "$1" == "setup" ]]; then
        HOST="ec2-18-188-95-229.us-east-2.compute.amazonaws.com"
        TYPE="ubuntu"
    elif [[ "$1" == "gpu2" ]]; then
        HOST="ec2-3-17-29-96.us-east-2.compute.amazonaws.com"
        TYPE="ubuntu"
    fi
    echo $HOST
}

get_device() {
    if [[ "$1" == "1" ]]; then
        DEVICE="96d0c028-399a-4559-967f-ebaa5177d87e"
    elif [[ "$1" == "2" ]]; then
        DEVICE="bb991a07-6876-4d88-9718-9cd930b5b011"
    fi
}


POS=()
while [[ $# -gt 0 ]]; do
    key="$1"

    case ${key} in
        -h|--host)
            get_host $2
            shift 2
            ;;
        --device)
            get_device $2
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

COMMAND=$1
shift

SSH=$(${HERE}/ssh.sh -h ${HOST} -t ${TYPE} -p)
case ${COMMAND} in
    connect)
        ${HERE}/ssh.sh -h ${HOST} -t ${TYPE} -T
        ;;
    list_models)
        $SSH << EOF
            set -vx
            source \$HOME/muon.env
            ls \$MUOND/clustering_models/${1:-}
EOF
        ;;
    setup_env)
        ${HERE}/upload_code.sh -h ${HOST} -t ${TYPE} --device ${DEVICE} -c
        ;;
    setup_python)
        ${HERE}/upload_code.sh -h ${HOST} -t ${TYPE} --device ${DEVICE} \
            -c --python
        ;;
    mount)
        $SSH << EOF
            if [ -z "\$(sudo blkid | grep ${DEVICE})" ]; then
                echo "Couldn't find device ${DEVICE}"
                echo \$(sudo blkid)
                exit 1
            fi
            sudo mkdir -p /mnt/muon
            sudo mount -U ${DEVICE} /mnt/muon
            if [ ! -d /mnt/muon/data ]; then
                mkdir /mnt/muon/data
            fi
            sudo chmod 777 /mnt/muon
EOF
        ;;
    launch)
        ${HERE}/upload_code.sh -h ${HOST} -t ${TYPE}
        ${HERE}/launch_run.sh -h ${HOST} -t ${TYPE} $@
        ;;
    exit)
        ${HERE}/ssh.sh -h ${HOST} -t ${TYPE} exit
        ;;

    upload_data)
        cd $MUON
        $MUON/run upload
        ssh zoo << EOF
            source \$HOME/.profile
            \${MUON}/muon/scripts/aws_deploy/upload_data.sh \
                -h ${HOST} -t ${TYPE} --device ${DEVICE}
EOF
        ;;
    download_run)
        cd $MUON
        $MUON/run upload
        ssh zoo << EOF
            source \$HOME/.profile
            \${MUON}/muon/scripts/aws_deploy/download_run.sh \
                -h ${HOST} -t ${TYPE} $@
EOF
        ;;
            


esac



