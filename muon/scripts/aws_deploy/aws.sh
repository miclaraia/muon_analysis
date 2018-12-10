#!/bin/bash
set -euvxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

get_host() {
    if [[ "$1" == "gpu" ]]; then
        HOST="ec2-3-17-27-155.us-east-2.compute.amazonaws.com"
        TYPE="ubuntu"
    elif [[ "$1" == "setup" ]]; then
        HOST="ec2-18-188-95-229.us-east-2.compute.amazonaws.com"
        TYPE="ubuntu"
    elif [[ "$1" == "gpu2" ]]; then
        HOST="ec2-18-224-39-95.us-east-2.compute.amazonaws.com"
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
        -d|--device)
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

    transfer)
        $MUON/run upload
        HOST1=$(get_host $1)
        HOST2=$(get_host $2)
        shift 2

        cd $MUON
        ssh zoo << EOF
            source \$HOME/.profile
            set -euxo pipefail
            cd \$MUON/muon/scripts/aws_deploy
            SSH1=\$(./ssh.sh -h ${HOST1} -t ${TYPE} -p)
            SSH1_PRE="\$(python3 -c "print(' '.join('\${SSH1}'.split(' ')[:-1]))")"
            HOST1="\$(python3 -c "print('\${SSH1}'.split(' ')[-1])")"
            SSH2=\$(./ssh.sh -h ${HOST2} -t ${TYPE} -p)
            SSH2_PRE="\$(python3 -c "print(' '.join('\${SSH2}'.split(' ')[:-1]))")"
            HOST2="\$(python3 -c "print('\${SSH2}'.split(' ')[-1])")"
            cd /tmp
            rsync -rPv -e "\$SSH1_PRE" \
                \$HOST1:/mnt/muon/data/clustering_models/$1/. muon_transfer
            rsync -rPv -e "\$SSH2_PRE" \
                muon_transfer/. \$HOST2:/mnt/muon/data/clustering_models/$1
            rm -R muon_transfer
EOF

esac



